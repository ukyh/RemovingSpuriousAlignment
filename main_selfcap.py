import os
import sys
import random
import json
import itertools
import numpy as np
from logging import getLogger

import torch

from tools.utils import decorate_logger
from arg_parser import get_args
from data_loader_selfcap import get_dataloader
from nns.initializer import init_model_
from nns.model import Model


logger = getLogger()


def get_model(args, embs):
    # from nns.model import Model
    model = Model(args, embs, vocab)
    init_model_(model, args.init_method)
    logger.info(model)

    # load pre-trained parameters
    model.load_state_dict(torch.load(args.param_path))
    logger.info("Loaded pre-trained model: {}".format(args.param_path))

    # to GPU
    if args.device != "cpu":
        model.to(args.device)

    return model


def train_loop(args, model):
    gen, gts, ids, objs = list(), list(), list(), list()
    model.eval()
    for batch in train_loader:
        caps, box_feats, img_ids, detected = batch
        caps = [cap.split('\t') for cap in caps]
        with torch.no_grad():
            pred_seqs = model(box_feats, 'test')
        pred_strs = str_decode(pred_seqs, show_full=False)
        for pred, cap, img, det in zip(pred_strs, caps, img_ids, detected):
            unigrams = set(pred.split())
            intersect_norm = set([
                train_loader.dataset.sin2plu[i] if i in train_loader.dataset.sin2plu else i
                for i in unigrams & det
            ])
            det_norm = set([
                train_loader.dataset.sin2plu[i] if i in train_loader.dataset.sin2plu else i
                for i in det
            ])
            if len(intersect_norm) >= args.min_intersect or (len(intersect_norm) >= len(det_norm) and len(det_norm) >= 1):
                gen.append(pred)
                gts.append(cap)
                ids.append(img)
                objs.append(det)
    out = [{"image_id":i, "caption":g} for i, g in zip(ids, gen)]
    with open(args.gen_path, "w", encoding="utf-8", errors="ignore") as outfile:
        json.dump(out, outfile, indent=4)


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
    global train_loader, vocab
    train_loader, vocab, embs = get_dataloader(args)

    # get model
    model = get_model(args, embs)

    # run
    logger.info("Generate captions for training images")
    train_loop(args, model)

    logger.info("Finish main")


if __name__ == "__main__":
    args = get_args()
    logger = decorate_logger(args, logger)
    logger.info(args)
    logger.info(" ".join(sys.argv))
    main(args)

