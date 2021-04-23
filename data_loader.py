# Based on: https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py

import torch
import torch.utils.data as data

import os
import random
import numpy as np
import json
import h5py
from collections import Counter
from logging import getLogger

from tools.utils import Vocab


logger = getLogger(__name__)


######################################################################
## PREPROCESS DATA
######################################################################

attr_img_dict = {}
rel_img_dict = {}
val_test_dict = {}
train_feat_hdf5 = {}
val_feat_hdf5 = {}
test_feat_hdf5 = {}
sin2plu = {}
val_obj_dict = {}
test_obj_dict = {}
def load_data(args):
    global attr_img_dict, rel_img_dict, val_test_dict, \
        train_feat_hdf5, val_feat_hdf5, test_feat_hdf5, \
        sin2plu, val_obj_dict, test_obj_dict
    with open(os.path.join(args.data_dir, args.attr_img_path)) as f:
        attr_img_dict = json.load(f)
        logger.info("Loaded {}".format(args.attr_img_path))
    with open(os.path.join(args.data_dir, args.rel_img_path)) as f:
        rel_img_dict = json.load(f)
        logger.info("Loaded {}".format(args.rel_img_path))
    with open(os.path.join(args.data_dir, args.val_test_path)) as f:
        val_test_dict = json.load(f)
        logger.info("Loaded {}".format(args.val_test_path))
    train_feat_hdf5 = h5py.File(os.path.join(args.img_dir, args.train_feat_path), "r")
    logger.info("Loaded {}".format(args.train_feat_path))
    val_feat_hdf5 = h5py.File(os.path.join(args.img_dir, args.val_feat_path), "r")
    logger.info("Loaded {}".format(args.val_feat_path))
    test_feat_hdf5 = h5py.File(os.path.join(args.img_dir, args.test_feat_path), "r")
    logger.info("Loaded {}".format(args.test_feat_path))
    with open(os.path.join(args.data_dir, args.plural_path), encoding="utf-8") as f:
        sin2plu = json.load(f)
        logger.info("Loaded plural dict")
    with open(os.path.join(args.data_dir, args.val_obj_path)) as f:
        val_obj_dict = json.load(f)
        logger.info("Loaded {}".format(args.val_obj_path))
    with open(os.path.join(args.data_dir, args.test_obj_path)) as f:
        test_obj_dict = json.load(f)
        logger.info("Loaded {}".format(args.test_obj_path))


vocab = Vocab()
embs = []
def load_vocab_emb(args):
    global vocab, embs
    if args.analysis:
        with open(args.t2i_path) as f:
            vocab.t2i_dict = json.load(f)
        vocab.i2t_dict = {int(i):t for t, i in vocab.t2i_dict.items()}
        embs = np.random.uniform(
            low=-0.08, high=0.08, size=(len(vocab), args.dim_tok)
        ).astype(np.float32)
        logger.info('Loaded the saved vocab: {}'.format(len(vocab)))
    else:
        _vocab = []
        if args.corpus == 'ss':
            # load the vocab of Feng et al., 2019
            with open(os.path.join(args.data_dir, args.vocab_path)) as f:
                for line in f:
                    _vocab.append(line.rstrip().split()[0])
                _vocab = set(_vocab) - {'<S>', '</S>'}
        else:
            doc = []
            # iterate attr
            with open(os.path.join(args.data_dir, args.attr_path)) as f:
                _attr_dict = json.load(f)
                for obj, cap_idx_list in _attr_dict.items(): # attr_dict {obj: [[text, [attr_indexes], [attr_indexes_no_obj]], ...]}
                    for v in obj.split():   # to deal with compounds
                        _vocab.append(v)
                    for cap, *_ in cap_idx_list:
                        doc.extend(cap.rstrip().split())
            # iterate rel
            with open(os.path.join(args.data_dir, args.rel_path)) as f:
                _rel_dict = json.load(f)
                for pair, cap_idx_list in _rel_dict.items(): # {"obj1[\t]obj2": [[text, [rel_indexes], [rel_indexes_no_obj]], ...]}
                    obj1, obj2 = pair.split("\t")
                    if obj1 in _attr_dict and obj2 in _attr_dict:
                        for v in pair.split():
                            _vocab.append(v)
                        for cap, *_ in cap_idx_list:
                            doc.extend(cap.rstrip().split())
            # frequency filter
            count = Counter(doc)
            logger.info("Vocab before min_freq={} filter: {}".format(args.min_freq, len(count)))
            for token, freq in count.most_common(None):
                if freq < args.min_freq:
                    break
                _vocab.append(token)
            _vocab = set(_vocab)

        for token in _vocab:
            vocab.t2i_dict[token] = len(vocab.t2i_dict)
            vocab.i2t_dict[len(vocab.i2t_dict)] = token
        embs = np.random.uniform(
            low=-0.08, high=0.08, size=(len(vocab), args.dim_tok)
        ).astype(np.float32)
        logger.info("Created vocab: {}".format(len(vocab)))
        if args.save:
            with open(args.t2i_path, "w", encoding="utf-8") as outfile:
                json.dump(vocab.t2i_dict, outfile, indent=4)
            logger.info("Saved vocab")


# {obj:[[[cap_idxs], [path_idxs]], ...]}
# {pair:[[[cap_idxs], [path_idxs]], ...]}
attr_dict = {}
rel_dict = {}
def load_train_text(args):
    global attr_dict, rel_dict
    with open(os.path.join(args.data_dir, args.attr_path)) as f:
        _attr_dict = json.load(f)
        for obj, sents in _attr_dict.items():
            attr_dict[obj] = []
            for items in sents:
                cap, idxs, idxs_no_obj = items
                tmp_cap = []
                for token in cap.split():
                    tmp_cap.append(vocab.t2i(token))
                tmp_cap.append(vocab.eos)
                if tmp_cap.count(vocab.unk) <= len(tmp_cap) * 0.15:    # remove sentences with too much <unk>
                    attr_dict[obj].append([tmp_cap, idxs, idxs_no_obj]) # [[[cap_idxs], [path_idxs], [path_idxs_no_obj]], ...]
            if len(attr_dict[obj]) < 1:
                del attr_dict[obj]                              # remove obj with no element
    logger.info("Loaded and converted sentences of {}".format(args.attr_path))
    with open(os.path.join(args.data_dir, args.rel_path)) as f:
        _rel_dict = json.load(f)
        count = 0
        for pair, sents in _rel_dict.items():
            obj1, obj2 = pair.split("\t")
            if obj1 in attr_dict and obj2 in attr_dict: # to avoid out-of-index attr
                if count == args.max_data:
                    break
                rel_dict[pair] = []
                for items in sents:
                    cap, idxs, idxs_no_obj = items
                    tmp_cap = []
                    for token in cap.split():
                        tmp_cap.append(vocab.t2i(token))
                    tmp_cap.append(vocab.eos)
                    if tmp_cap.count(vocab.unk) <= len(tmp_cap) * 0.15:    # remove sentences with too much <unk>
                        rel_dict[pair].append([tmp_cap, idxs, idxs_no_obj])    # [[[cap_idxs], [path_idxs], [path_idxs_no_obj]], ...]
                if len(rel_dict[pair]) < 1:
                    del rel_dict[pair]                         # remove pair with no element
                else:
                    count += 1
    logger.info("Loaded and converted sentences of {}".format(args.rel_path))
    logger.info("Loaded pairs: {}".format(len(rel_dict)))


def load_obj_class():
    global vocab
    obj_class = []
    for sin, plu in sin2plu.items():
        obj_class.append(vocab.t2i(sin))
        obj_class.append(vocab.t2i(plu))
    vocab.obj_class = set(obj_class) - {vocab.unk}  # remove <unk> from object class
    logger.info("Loaded objects (singular + plural): {}".format(len(vocab.obj_class)))




######################################################################
## DATA LOADER
######################################################################

class TrainDataset(data.Dataset):
    """ Custom data.Dataset compatible with data.DataLoader. """
    def __init__(self):
        self.attr_dict = attr_dict
        self.rel_dict = rel_dict
        self.attr_img_dict = attr_img_dict
        self.rel_img_dict = rel_img_dict
        self.img_feat_hdf5 = train_feat_hdf5
        self.pair_idxs = {i:pair for i, pair in enumerate(rel_dict.keys())}
        self.num_total_pairs = len(self.pair_idxs)

    def __getitem__(self, index):
        """ Returns pair-wise features in list. """
        # text features
        pair = self.pair_idxs[index]
        obj1, obj2 = pair.split("\t")
        pair_items = self.rel_dict[pair]
        obj1_items = self.attr_dict[obj1]
        obj2_items = self.attr_dict[obj2]
        pair_rand_idx = random.randint(0, len(pair_items) - 1)
        obj1_rand_idx = random.randint(0, len(obj1_items) - 1)
        obj2_rand_idx = random.randint(0, len(obj2_items) - 1)
        pair_texts, pair_spans, pair_spans_no_obj = pair_items[pair_rand_idx]
        pair_texts = torch.LongTensor(pair_texts)                 # (slen)
        pair_spans = torch.LongTensor(pair_spans)                 # (idxs)
        pair_spans_no_obj = torch.LongTensor(pair_spans_no_obj)   # (idxs)
        obj1_texts, obj1_spans, obj1_spans_no_obj = obj1_items[obj1_rand_idx]
        obj1_texts = torch.LongTensor(obj1_texts)
        obj1_spans = torch.LongTensor(obj1_spans)
        obj1_spans_no_obj = torch.LongTensor(obj1_spans_no_obj)
        obj2_texts, obj2_spans, obj2_spans_no_obj = obj2_items[obj2_rand_idx]
        obj2_texts = torch.LongTensor(obj2_texts)
        obj2_spans = torch.LongTensor(obj2_spans)
        obj2_spans_no_obj = torch.LongTensor(obj2_spans_no_obj)

        # image features
        pair_img_items = self.rel_img_dict[pair]   # ["image_id", ...]
        obj1_img_items = self.attr_img_dict[obj1]
        obj2_img_items = self.attr_img_dict[obj2]
        pair_img_rand_idx = random.randint(0, len(pair_img_items) - 1)
        obj1_img_rand_idx = random.randint(0, len(obj1_img_items) - 1)
        obj2_img_rand_idx = random.randint(0, len(obj2_img_items) - 1)
        pair_feats = torch.FloatTensor(self.img_feat_hdf5[pair_img_items[pair_img_rand_idx]][()])   # (1, dim)
        obj1_feats = torch.FloatTensor(self.img_feat_hdf5[obj1_img_items[obj1_img_rand_idx]][()])
        obj2_feats = torch.FloatTensor(self.img_feat_hdf5[obj2_img_items[obj2_img_rand_idx]][()])

        return pair_texts, pair_spans, pair_spans_no_obj, \
            obj1_texts, obj1_spans, obj1_spans_no_obj, \
            obj2_texts, obj2_spans, obj2_spans_no_obj, \
            pair_feats, obj1_feats, obj2_feats

    def __len__(self):
        return self.num_total_pairs


def collate_train(data):
    """ Creates train mini-batch tensors.
    NOTE: Training captions already have '<eos>' at the end and it is also reflected in length
    NOTE: Span index starts from 0

    Args:
        data: List of a tuple (pair_texts, pair_spans, pair_spans_no_obj, obj1_texts, obj1_spans, obj1_spans_no_obj, obj2_texts, obj2_spans, obj2_spans_no_obj, pair_feats, obj1_feats, obj2_feats),
              made by [self.dataset[i] for i in indexes].
        pair(obj)_texts: (slen)
        pair(obj)_spans: (idxs)
        pair(obj)_spans_no_obj: (idxs)
        pair(obj)_feats : (1, dim)

    Returns:
        Padded tensors.
        Sequence tensor is `sorted` by length.
        pair(obj)_texts: (batch, max_slen)
        pair(obj)_lengths: (batch)
        pair(obj)_spans: (batch, max_slen)
        pair(obj)_spans_no_obj: (batch, max_slen)
        pair(obj)_feats : (batch, 1, dim)
    """
    def _merge_text(texts):
        lengths = [len(s) for s in texts]
        padded_texts = torch.zeros(len(lengths), max(lengths)).long()
        for i, text in enumerate(texts):
            end = lengths[i]
            padded_texts[i, :end] = text[:end]
        return padded_texts, lengths

    def _merge_span(spans, lengths):
        padded_spans = torch.zeros(len(lengths), max(lengths)).long()
        for i, span in enumerate(spans):
            padded_spans[i, span] = 1   # padded_spans[i, [idxs]]
        return padded_spans.bool()

    # seperate items
    pair_texts, pair_spans, pair_spans_no_obj, \
        obj1_texts, obj1_spans, obj1_spans_no_obj, \
        obj2_texts, obj2_spans, obj2_spans_no_obj, \
        pair_feats, obj1_feats, obj2_feats \
        = zip(*data)    # (tensor * batch) * item_num
    
    # merge with paddings
    pair_texts, pair_lengths = _merge_text(pair_texts)
    obj1_texts, obj1_lengths = _merge_text(obj1_texts)
    obj2_texts, obj2_lengths = _merge_text(obj2_texts)
    pair_spans = _merge_span(pair_spans, pair_lengths)
    obj1_spans = _merge_span(obj1_spans, obj1_lengths)
    obj2_spans = _merge_span(obj2_spans, obj2_lengths)
    pair_spans_no_obj = _merge_span(pair_spans_no_obj, pair_lengths)
    obj1_spans_no_obj = _merge_span(obj1_spans_no_obj, obj1_lengths)
    obj2_spans_no_obj = _merge_span(obj2_spans_no_obj, obj2_lengths)
    pair_lengths = torch.LongTensor(pair_lengths)
    obj1_lengths = torch.LongTensor(obj1_lengths)
    obj2_lengths = torch.LongTensor(obj2_lengths)

    pair_spans = pair_spans ^ pair_spans_no_obj # xor
    obj1_spans = obj1_spans ^ obj1_spans_no_obj
    obj2_spans = obj2_spans ^ obj2_spans_no_obj

    # sort sequences by length (descending order) to use pack_padded_sequence later
    pair_lengths, sort_pairidx = torch.sort(pair_lengths, dim=0, descending=True)
    obj1_lengths, sort_obj1idx = torch.sort(obj1_lengths, dim=0, descending=True)
    obj2_lengths, sort_obj2idx = torch.sort(obj2_lengths, dim=0, descending=True)
    pair_texts = pair_texts[sort_pairidx.data]
    obj1_texts = obj1_texts[sort_obj1idx.data]
    obj2_texts = obj2_texts[sort_obj2idx.data]
    pair_spans = pair_spans[sort_pairidx.data]
    obj1_spans = obj1_spans[sort_obj1idx.data]
    obj2_spans = obj2_spans[sort_obj2idx.data]

    # image features
    pair_feats = torch.stack(pair_feats, dim=0) # (batch, 1, dim)
    obj1_feats = torch.stack(obj1_feats, dim=0)
    obj2_feats = torch.stack(obj2_feats, dim=0)
    pair_feats = pair_feats[sort_pairidx.data]
    obj1_feats = obj1_feats[sort_obj1idx.data]
    obj2_feats = obj2_feats[sort_obj2idx.data]

    return pair_texts, pair_lengths, pair_spans, \
        obj1_texts, obj1_lengths, obj1_spans, \
        obj2_texts, obj2_lengths, obj2_spans, \
        pair_feats, obj1_feats, obj2_feats


# NOTE: passing ground-truth captions without <unk>
# NOTE: speaksee tokenizer removes "." and lower case later
class EvalDataset(data.Dataset):
    """ Custom data.Dataset compatible with data.DataLoader. """
    def __init__(self, mode):
        # val/test is separated with `mode`
        self.val_test_dict = val_test_dict[mode]
        if mode == "val":
            self.img_feat_hdf5 = val_feat_hdf5
            self.obj_dict = val_obj_dict
        elif mode == "test":
            self.img_feat_hdf5 = test_feat_hdf5
            self.obj_dict = test_obj_dict
        else:
            raise KeyError("Invalid mode specified in eval dataloader: {}".format(mode))
        self.sin2plu = sin2plu
        self.img_idxs = {i:img for i, img in enumerate(self.val_test_dict.keys())}
        self.num_total_img = len(self.img_idxs)

    def __getitem__(self, index):
        """ Returns image features and corresponding caption. """
        img_id = self.img_idxs[index]
        caps = "\t".join(self.val_test_dict[img_id])  # "cap1[\t]..."
        img_feats = torch.FloatTensor(self.img_feat_hdf5[img_id][()])   # (1, dimi)
        _objs = self.obj_dict[img_id]
        _objs_plu = [self.sin2plu[obj] for obj in _objs if obj in self.sin2plu]
        objs = set(_objs + _objs_plu) 
        return caps, img_feats, img_id, objs

    def __len__(self):
        return self.num_total_img


def collate_eval(data):
    """ Creates eval mini-batch tensors.

    Args:
        data: List of a tuple (caps, img_feats, img_id, objs),
              made by [self.dataset[i] for i in indexes].
        caps: str
        img_feats: (1, dim)
        img_id: str
        objs: {obj1, ...}

    Returns:
        caps: tuple(batch)
        img_feats: (batch, 1, dim)
        img_ids: tuple(batch)
        objs: tuple({obj1, ...} * batch)
    """
    caps, img_feats, img_ids, objs = zip(*data)
    img_feats = torch.stack(img_feats, dim=0)
    assert len(caps) == len(img_feats) == len(img_ids) == len(objs)

    return caps, img_feats, img_ids, objs




def get_dataloader(args):
    """ Returns data loader for custom datasets and pre-trained embeddings.

    Returns:
        train_loader, val_loader, test_loader, vocab, embs
    """
    # preprocess data
    load_data(args)
    load_vocab_emb(args)
    load_train_text(args)
    load_obj_class()

    # create custom datasets
    train_dataset = TrainDataset()
    val_dataset = EvalDataset("val")
    test_dataset = EvalDataset("test")

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_train, 
        shuffle=True, collate_fn=collate_train
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_eval, 
        shuffle=False, collate_fn=collate_eval
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_eval, 
        shuffle=False, collate_fn=collate_eval
    )

    return train_loader, val_loader, test_loader, \
        vocab, embs
