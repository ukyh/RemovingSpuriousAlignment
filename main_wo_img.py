import os
import sys
import random
import json
import itertools
import numpy as np
from logging import getLogger

import torch
import torch.nn as nn

from speaksee import evaluation
from tools.utils import decorate_logger, bow_eval
from arg_parser import get_args
from data_loader_wo_img import get_dataloader
from nns.initializer import init_model_


logger = getLogger()


def get_model(args, embs, embs_obj):
    from nns.model_wo_img import Model
    model = Model(args, embs, embs_obj, vocab)
    init_model_(model, args.init_method)
    logger.debug(model)

    # load pre-trained parameters
    if args.analysis:
        model.load_state_dict(torch.load(args.param_path))
        logger.info("Loaded pre-trained model: {}".format(args.param_path))

    # to GPU
    if args.device != "cpu":
        model.to(args.device)

    return model


def get_gen_criterion(args):
    gen_criterion = nn.CrossEntropyLoss()
    pos_gate_weight = torch.FloatTensor([args.pos_gate_weight])
    gate_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_gate_weight)

    # to GPU
    if args.device != "cpu":
        gen_criterion.to(args.device)
        gate_criterion.to(args.device)

    return gen_criterion, gate_criterion


def get_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "AMSGrad":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    else:
        raise KeyError("Invalid optimizer: {}".format(args.optimizer))

    return optimizer


def train_loop(args, model, gen_criterion, gate_criterion, optimizer):
    GenLoss, GateLoss = [], []
    optimizer.zero_grad()
    model.train()
    for i, batch in enumerate(train_loader):
        pair_preds, pair_gts, pair_gate_preds, pair_gate_label, \
            obj1_preds, obj1_gts, obj1_gate_preds, obj1_gate_label, \
            obj2_preds, obj2_gts, obj2_gate_preds, obj2_gate_label \
            = model(batch, "train")
        genloss = gen_criterion(pair_preds, pair_gts) \
            + gen_criterion(obj1_preds, obj1_gts) \
            + gen_criterion(obj2_preds, obj2_gts)
        gateloss = gate_criterion(pair_gate_preds, pair_gate_label) \
            + gate_criterion(obj1_gate_preds, obj1_gate_label) \
            + gate_criterion(obj2_gate_preds, obj2_gate_label)
        loss = genloss
        if args.use_pseudoL:
            loss = loss + gateloss * args.loss_weight
        
        loss = loss / args.accumulation_size
        loss.backward()
        if (i + 1) % args.accumulation_size == 0:
            if args.grad_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip, norm_type=2
                )
            optimizer.step()
            optimizer.zero_grad()

            GenLoss.append(genloss.item() / 3)
            GateLoss.append(gateloss.item() / 3)
        
        # free GPU memory
        del loss, genloss, gateloss, \
            pair_preds, pair_gts, pair_gate_preds, pair_gate_label, \
            obj1_preds, obj1_gts, obj1_gate_preds, obj1_gate_label, \
            obj2_preds, obj2_gts, obj2_gate_preds, obj2_gate_label
        torch.cuda.empty_cache()

    GenLoss = sum(GenLoss) / len(GenLoss)
    GateLoss = sum(GateLoss) / len(GateLoss)
    logger.info("Train\tGenLoss:{} GateLoss:{}".format(GenLoss, GateLoss))


def str_decode(pred_seqs, show_full=False):
    pred_strs = []
    for seq in pred_seqs:
        tmp_seq = []
        for tok_id in seq:
            if tok_id == vocab.eos:
                break
            tmp_seq.append(vocab.i2t(tok_id))
        if show_full:
            tmp_seq = " ".join(tmp_seq)
        else:
            tmp_seq = " ".join([k for k, _ in itertools.groupby(tmp_seq)])  # remove simple repetition
        pred_strs.append(tmp_seq)

    return pred_strs    # ["seq" * batch]


def metric_eval(gen, gts, objs, is_test):
    # NOTE: tokenizer removes puctuations and lowers the case
    gen = evaluation.PTBTokenizer.tokenize(gen) # {0:["gen0"], ...}
    gts = evaluation.PTBTokenizer.tokenize(gts) # {0:["gt0-0", ..., "gt0-4"], ...}
    score_dict = {}
    bleu, _ = evaluation.Bleu(n=4).compute_score(gts, gen)
    method = ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]
    for metric, score in zip(method, bleu):
        score_dict[metric] = score
    cider, _ = evaluation.Cider().compute_score(gts, gen)
    score_dict["CIDEr"] = cider
    if is_test: # to save time
        meteor, _ = evaluation.Meteor().compute_score(gts, gen)
        score_dict["METEOR"] = meteor
        rouge, _ = evaluation.Rouge().compute_score(gts, gen)
        score_dict["ROUGE_L"] = rouge
        spice, _ = evaluation.Spice().compute_score(gts, gen)
        score_dict["SPICE"] = spice
    gen_det, gts_det = {}, {}
    gen_other, gts_other = {}, {}
    for i, det in enumerate(objs):
        gen_det[i] = set(gen[i][0].split()) & det   # {i:{set}}
        gts_det[i] = [set(s.split()) & det for s in gts[i]] # {i:[{set1}, ...]}
        gen_other[i] = set(gen[i][0].split()) - det
        gts_other[i] = [set(s.split()) - det for s in gts[i]]
    prec_det, rec_det, f1_det = bow_eval(gts_det, gen_det)
    prec_other, rec_other, f1_other = bow_eval(gts_other, gen_other)
    score_dict["Precision-Detected"] = prec_det
    score_dict["Recall-Detected"] = rec_det
    score_dict["F1-Detected"] = f1_det
    score_dict["Precision-Others"] = prec_other
    score_dict["Recall-Others"] = rec_other
    score_dict["F1-Others"] = f1_other
    
    return score_dict   # {"metric":score, ...}


def val_loop(args, model):
    gen, gts, ids, objs = [], [], [], []
    model.eval()
    for batch in val_loader:
        caps, img_feats, num_objs, img_ids, detected = batch
        caps = [cap.split("\t") for cap in caps]    # [["gt1", ..., "gt5"] * batch]
        with torch.no_grad():
            pred_seqs = model((img_feats, num_objs), "val")
        pred_strs = str_decode(pred_seqs, show_full=False) # ["seq" * batch]
        gen.extend(pred_strs)
        gts.extend(caps)
        ids.extend(img_ids)
        objs.extend(detected)
    score_dict = metric_eval(gen, gts, objs, is_test=False)
    logger.info("Val\t{}".format(" ".join([":".join([k, str(v)]) for k, v in score_dict.items()])))

    return score_dict["CIDEr"]


def test_loop(args, model):
    gen, gts, ids, objs = [], [], [], []
    model.eval()
    for batch in test_loader:
        caps, img_feats, num_objs, img_ids, detected = batch
        caps = [cap.split("\t") for cap in caps]
        with torch.no_grad():
            pred_seqs = model((img_feats, num_objs), "test")
        pred_strs = str_decode(pred_seqs, show_full=False)
        gen.extend(pred_strs)
        gts.extend(caps)
        ids.extend(img_ids)
        objs.extend(detected)
    if args.analysis:
        out = [{"image_id":i, "caption":g} for i, g in zip(ids, gen)]
        with open(args.gen_path, "w", encoding="utf-8") as outfile:
            json.dump(out, outfile, indent=4)
    score_dict = metric_eval(gen, gts, objs, is_test=True)
    
    return score_dict


def analysis_loop(args, model):
    model.eval()
    for batch in val_loader:
        gen, gts, ids, objs = [], [], [], []
        caps, img_feats, num_objs, img_ids, detected = batch
        caps = [cap.split("\t") for cap in caps]
        with torch.no_grad():
            pred_seqs, img_norm_list, seq_norm_list, \
                gate_val_list, pred_cand_list, cand_prob_list \
                = model.analyze((img_feats, num_objs), "analysis")
        pred_strs = str_decode(pred_seqs, show_full=True)
        gen.extend(pred_strs)
        gts.extend(caps)
        ids.extend(img_ids)
        objs.extend(detected)
        for gen_i, gts_i, i, obj, inorm, snorm, gate, cand, prob in zip(gen, gts, ids, objs, img_norm_list, seq_norm_list, gate_val_list, pred_cand_list, cand_prob_list):
            print("## image:", i)
            print("## detected:", obj)
            print("## gen:", gen_i)
            print("## gts:", gts_i)
            for tok_t, snorm_t, gate_t, cand_t, prob_t in zip(gen_i.split(), snorm, gate, cand, prob):
                print(
                    "{} (gate:{:.4f} image_norm:{:.4f} seq_norm:{:.4f}) ".format(tok_t, gate_t, inorm, snorm_t)
                    + " ".join(["{}-{:.4f}".format(vocab.i2t(c), p) for c, p in zip(cand_t, prob_t)])
                )
            print()


def main(args):
    logger.info("Start main")

    # set gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.device != "cpu":
        args.device = "cuda"

    # fix seed
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    # get data
    global train_loader, val_loader, test_loader, vocab
    train_loader, val_loader, test_loader, \
        vocab, embs, embs_obj = get_dataloader(args)

    # get model and set optimizer
    model = get_model(args, embs, embs_obj)
    gen_criterion, gate_criterion = get_gen_criterion(args)
    optimizer = get_optimizer(args, model)

    # run train and eval
    if not args.analysis:
        logger.info("Start training")
        best_score = 0
        best_epoch = 0
        duration = 0
        for epoch in range(args.epoch_size):
            logger.info("Epoch: {}".format(epoch))
            train_loop(args, model, gen_criterion, gate_criterion, optimizer)
            val_score = val_loop(args, model)
            if val_score > best_score:
                test_score = test_loop(args, model)
                best_score = val_score
                best_epoch = epoch
                duration = 0
                if args.save:
                    torch.save(model.state_dict(), args.param_path)
                    logger.info("Saved model in epoch {}".format(epoch))
            else:
                duration += 1
                if duration > args.early_stop:
                    logger.info("Early stop in epoch {}".format(epoch))
                    break
        logger.info("Best epoch: {}".format(best_epoch))
        logger.info("Best validation score: {}".format(best_score))
        logger.info("Test\t{}".format(" ".join([":".join([k, str(v)]) for k, v in test_score.items()])))
    # run analysis
    else:
        logger.info("Start analysis on {}".format(args.param_path))
        analysis_loop(args, model)
        test_score = test_loop(args, model)
        logger.info("Test\t{}".format(" ".join([":".join([k, str(v)]) for k, v in test_score.items()])))

    logger.info("Finish main")


if __name__ == "__main__":
    args = get_args()
    logger = decorate_logger(args, logger)
    logger.info(args)
    logger.info(" ".join(sys.argv))
    main(args)

