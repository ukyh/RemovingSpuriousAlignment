# Based on: https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5

import torch.nn as nn
import torch.nn.init as init
from logging import getLogger


logger = getLogger(__name__)
initializer = None


def get_initializer(init_method):
    
    global initializer

    if init_method == "he_normal":
        initializer = init.kaiming_normal_
    elif init_method == "he_uniform":
        initializer = init.kaiming_uniform_
    elif init_method == "xavier_normal":
        initializer = init.xavier_normal_
    elif init_method == "xavier_uniform":
        initializer = init.xavier_uniform_
    elif init_method == "normal":
        initializer = init.normal_
    elif init_method == "uniform":
        initializer = init.uniform_
    elif init_method == "orthogonal":
        initializer = init.orthogonal_
    elif init_method == "zero":
        initializer = init.zeros_
    elif init_method == "default":
        initializer = None
    else:
        raise KeyError("Unsupported init method: {}".format(init_method))


def _init(model):
    """
    Usage:
        model = Model()
        model.apply(_init)
    """

    if initializer == None: # default initializer
        pass
    else:
        # CNN
        #   Bias is initialized with zeros
        if isinstance(model, nn.Conv1d):
            initializer(model.weight)
            if model.bias is not None:
                init.zeros_(model.bias)
        elif isinstance(model, nn.Conv2d):
            initializer(model.weight)
            if model.bias is not None:
                init.zeros_(model.bias)
        elif isinstance(model, nn.Conv3d):
            initializer(model.weight)
            if model.bias is not None:
                init.zeros_(model.bias)
        elif isinstance(model, nn.ConvTranspose1d):
            initializer(model.weight)
            if model.bias is not None:
                init.zeros_(model.bias)
        elif isinstance(model, nn.ConvTranspose2d):
            initializer(model.weight)
            if model.bias is not None:
                init.zeros_(model.bias)
        elif isinstance(model, nn.ConvTranspose3d):
            initializer(model.weight)
            if model.bias is not None:
                init.zeros_(model.bias)

        # Linear:
        #   Bias is initialized with zeros
        elif isinstance(model, nn.Linear):
            initializer(model.weight)
            if model.bias is not None:
                init.zeros_(model.bias)

        # RNN:
        #   Bias is initialized with zeros,
        #   except for the forget gate bias (between 1/4 and 2/4 in pytorch implementation),
        #   which is set to ones
        #   cf. http://proceedings.mlr.press/v37/jozefowicz15.pdf
        elif isinstance(model, nn.LSTM):
            for param in model.parameters():
                if len(param.shape) >= 2:
                    initializer(param)
                else:
                    blen = len(param)
                    init.zeros_(param)
                    init.ones_(param[blen // 4: blen // 2])
        elif isinstance(model, nn.LSTMCell):
            for param in model.parameters():
                if len(param.shape) >= 2:
                    initializer(param)
                else:
                    blen = len(param)
                    init.zeros_(param)
                    init.ones_(param[blen // 4: blen // 2])
        elif isinstance(model, nn.GRU):
            for param in model.parameters():
                if len(param.shape) >= 2:
                    initializer(param)
                else:
                    blen = len(param)
                    init.zeros_(param)
                    init.ones_(param[blen // 4: blen // 2])
        elif isinstance(model, nn.GRUCell):
            for param in model.parameters():
                if len(param.shape) >= 2:
                    initializer(param)
                else:
                    blen = len(param)
                    init.zeros_(param)
                    init.ones_(param[blen // 4: blen // 2])
        else:
            pass


def init_model_(model, init_method):
    """
    Usage:
        from initializer import init_model_
        model = Model()
        init_method = hoge

        init_model_(model, init_method)
    """
    get_initializer(init_method)
    model.apply(_init)
    logger.info("Init params with: {}".format(init_method))
