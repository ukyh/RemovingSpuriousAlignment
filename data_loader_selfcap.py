# Based on: https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py

import torch
import torch.utils.data as data

import os
import numpy as np
import json
import h5py
from logging import getLogger

from tools.utils import Vocab


logger = getLogger(__name__)


######################################################################
## PREPROCESS DATA
######################################################################

def preprocess_train(args):
    PATH = os.path.join(args.data_dir, "train_dict.json")
    if os.path.exists(PATH):
        with open(PATH) as f:
            train_dict = json.load(f)
            logger.info("Loaded {}".format(PATH))
    else:
        with open(os.path.join(args.data_dir, args.train_obj_path), encoding="utf-8") as f:
            _train_dict = json.load(f)
            train_dict = {k:[] for k in _train_dict}
        
        def load_caps(coco_json, train_dict):
            with open(os.path.join(args.data_dir, coco_json), encoding="utf-8") as f:
                caption_data = json.load(f)
                id_to_name = [(x['id'], x['file_name']) for x in caption_data['images']]
                id_to_name = dict(id_to_name)
                for items in caption_data['annotations']:
                    fname = id_to_name[items['image_id']]
                    cap = items['caption']
                    if fname in train_dict:
                        train_dict[fname].append(cap)
            return train_dict

        train_dict = load_caps("captions_train2014.json", train_dict)
        train_dict = load_caps("captions_val2014.json", train_dict)
        with open(PATH, "w", encoding="utf-8") as outfile:
            json.dump(train_dict, outfile, indent=4)
        logger.info("Saved {}".format(PATH))

    return train_dict


train_dict = {}
train_feat_hdf5 = {}
train_obj_dict = {}
sin2plu = {}
def load_data(args):
    global train_dict, train_feat_hdf5, \
        train_obj_dict, sin2plu
    train_dict = preprocess_train(args)
    train_feat_hdf5 = h5py.File(os.path.join(args.img_dir, args.train_feat_path), "r")
    logger.info("Loaded {}".format(args.train_feat_path))
    with open(os.path.join(args.data_dir, args.train_obj_path)) as f:
        train_obj_dict = json.load(f)
        logger.info("Loaded {}".format(args.train_obj_path))
    with open(os.path.join(args.data_dir, args.plural_path), encoding="utf-8") as f:
        sin2plu = json.load(f)
        logger.info("Loaded plural dict")


vocab = Vocab()
embs = []
def load_vocab_emb(args):
    global vocab, embs
    with open(args.t2i_path) as f:
        vocab.t2i_dict = json.load(f)
    vocab.i2t_dict = {int(i):t for t, i in vocab.t2i_dict.items()}
    embs = np.random.uniform(
        low=-0.08, high=0.08, size=(len(vocab), args.dim_tok)
    ).astype(np.float32)
    logger.info('Loaded the saved vocab: {}'.format(len(vocab)))


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

class EvalDataset(data.Dataset):
    """ Custom data.Dataset compatible with data.DataLoader. """
    def __init__(self):
        self.train_dict = train_dict
        self.img_feat_hdf5 = train_feat_hdf5
        self.obj_dict = train_obj_dict
        self.sin2plu = sin2plu
        self.img_idxs = {i:img for i, img in enumerate(self.train_dict.keys())}
        self.num_total_img = len(self.img_idxs)

    def __getitem__(self, index):
        """ Returns image features and corresponding caption. """
        img_id = self.img_idxs[index]
        caps = "\t".join(self.train_dict[img_id])  # "cap1[\t]..."
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
        train_loader, vocab, embs
    """
    # preprocess data
    load_data(args)
    load_vocab_emb(args)
    load_obj_class()

    # create custom datasets
    train_dataset = EvalDataset()

    # create dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_eval, 
        shuffle=False, collate_fn=collate_eval
    )

    return train_loader, vocab, embs
