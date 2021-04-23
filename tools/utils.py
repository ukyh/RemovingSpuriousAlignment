import datetime
from logging import INFO, DEBUG, Formatter, StreamHandler, FileHandler


def trace(*args):
    """ Simple logging. """
    print(datetime.datetime.now().strftime("%H:%M:%S"), " ".join(map(str, args)))


def decorate_logger(args, logger):
    """ Decorate a root logger. 
        Stream for debug and File for experimental logs.
    """
    logger.setLevel(DEBUG)
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if args.log_path != "":
        f_handler = FileHandler(filename=args.log_path, mode="w", encoding="utf-8")
        f_handler.setLevel(INFO)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger


class Vocab:

    def __init__(self):
        self.t2i_dict = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3}
        self.i2t_dict = {0:'<pad>', 1:'<unk>', 2:'<bos>', 3:'<eos>'}
        self.obj_pool = set()
        self.pad = self.t2i_dict['<pad>']
        self.unk = self.t2i_dict['<unk>']
        self.bos = self.t2i_dict['<bos>']
        self.eos = self.t2i_dict['<eos>']
        self.specials = 4
    
    def __len__(self):
        return len(self.t2i_dict)

    def t2i(self, token):
        return self.t2i_dict.get(token, self.unk)
    
    def i2t(self, index):
        if index in self.i2t_dict:
            return self.i2t_dict[index]
        else:
            raise KeyError('Invalid index: {}'.format(index))


def bow_eval(gts, gen):
    """
    Input:
        gts: {i:[{BoW}_1, ...]}
        gen: {i:{BoW}}
    
    Return:
        precision, recall, f1
    """
    assert len(gts) == len(gen)
    prec, rec, f1 = [], [], []
    for i in range(len(gts)):
        intersect_skip = [len(gts[i][j] & gen[i]) for j in range(len(gts[i])) if len(gts[i][j]) > 0] # skip if there is no gts in the category
        intersect = [len(gts[i][j] & gen[i]) for j in range(len(gts[i]))]
        _prec = [c / len(gen[i]) if len(gen[i]) > 0 else 0 for c in intersect_skip]
        _rec = [c / len(gts[i][j]) for j, c in enumerate(intersect) if len(gts[i][j]) > 0]
        _f1 = [2 * p * r / (p + r) for p, r in zip(_prec, _rec) if p + r > 0]    # skip 0 prec + rec
        prec.extend(_prec)
        rec.extend(_rec)
        f1.extend(_f1)
    assert len(prec) == len(rec)
    return sum(prec) / len(prec), sum(rec) / len(rec), sum(f1) / len(prec)  # divide by len(prec) since there are skipped f1 
