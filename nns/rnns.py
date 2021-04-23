import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class LSTM(nn.Module):

    def __init__(self, dim_in, dim_hid, nlayer, 
        drop_rate=0., out_pad=0.):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            dim_in, dim_hid, nlayer, batch_first=True,
            dropout=drop_rate, bidirectional=False
        )
        self.nlayer = nlayer
        self.dim_hid = dim_hid
        self.out_pad = out_pad
        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.nlayer, batch_size, self.dim_hid).to(self.device),
            torch.zeros(self.nlayer, batch_size, self.dim_hid).to(self.device)
        )   # ( (nlayer, n_sent, dimh) * 2 )

    def forward(self, x, slen, prev_hc, mode):
        """
        Args:
            mode: str{"train", "val", "test"}
            mode == "train":
                x: (n_sent, max_slen+1, dimw), padded and sorted tensor + image feature
                slen: (n_sent) or [n_sent], each element corresponds to a sentence length,
                      which includes "<eos>" BUT NOT the added initial input
                prev_hc: () in encoder and ( (nlayer, n_sent, dimh) * 2 ) in decoder
            mode in {"val", "test"}:
                x: (n_sent, 1, dimw), image feature in the first step and previous word embedding in the later step
                slen: []
                prev_hc: () in first step and ( (nlayer, n_sent, dimh) * 2 ) in the later steps
        """
        if mode == "train":
            # NOTE: input corresponds to "<eos>" is omitted in pack
            if type(slen) is list:
                total_length = max(slen)
            else:
                total_length, _ = slen.max(dim=0)
                total_length = int(total_length.item())
            x = pack(x, slen, batch_first=True)                                 # (sum(slen), dimw)

            if len(prev_hc) < 1:
                prev_hc = self.init_hidden(len(slen))                           # ( (nlayer, n_sent, dim) * 2 )
            hid, (hid_n, cell_n) = self.lstm(x, prev_hc)                        # (sum(slen), dimh), ( (nlayer, n_sent, dimh) * 2 )

            hid, hsize = unpack(
                hid, batch_first=True,
                padding_value=self.out_pad, total_length=total_length
            )   # (n_sent, maxlen, dimh)
            
            return hid, (hid_n, cell_n)

        else:
            if len(prev_hc) < 1:
                prev_hc = self.init_hidden(len(x))                              # ( (nlayer, n_sent, dim) * 2 )
            hid, (hid_n, cell_n) = self.lstm(x, prev_hc)                        # (n_sent, 1, dimh), ( (nlayer, n_sent, dimh) * 2 )
            
            return hid, (hid_n, cell_n)


class GRU(nn.Module):

    def __init__(self, dim_in, dim_hid, nlayer, 
        drop_rate=0., out_pad=0.):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            dim_in, dim_hid, nlayer, batch_first=True,
            dropout=drop_rate, bidirectional=False
        )
        self.nlayer = nlayer
        self.dim_hid = dim_hid
        self.out_pad = out_pad
        if torch.cuda.device_count() > 0:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def init_hidden(self, batch_size):
        return torch.zeros(self.nlayer, batch_size, self.dim_hid).to(self.device) # (nlayer, n_sent, dimh)

    def forward(self, x, slen, prev_h, mode):
        """
        Args:
            mode: str{"train", "val", "test"}
            mode == "train":
                x: (n_sent, max_slen+1, dimw), padded and sorted tensor + image feature
                slen: (n_sent) or [n_sent], each element corresponds to a sentence length,
                      which includes "<eos>" BUT NOT the added initial input
                prev_h: () in encoder and ((nlayer, n_sent, dimh), None) in decoder
                        (2nd item is to make it compatible with LSTM)
            mode in {"val", "test"}:
                x: (n_sent, 1, dimw), image feature in the first step and previous word embedding in the later step
                slen: []
                prev_h: () in first step and ((nlayer, n_sent, dimh), None) in the later steps
                        (2nd item is to make it compatible with LSTM)
        """
        if mode == "train":
            # NOTE: input corresponds to "<eos>" is omitted in pack
            if type(slen) is list:
                total_length = max(slen)
            else:
                total_length, _ = slen.max(dim=0)
                total_length = int(total_length.item())
            x = pack(x, slen, batch_first=True)                                 # (sum(slen), dimw)

            if len(prev_h) < 1:
                prev_h = (self.init_hidden(len(slen)), None)                    # ( (nlayer, n_sent, dim), None )
            hid, hid_n = self.gru(x, prev_h[0])                                 # (sum(slen+1), dimh), (nlayer, n_sent, dimh)

            hid, hsize = unpack(
                hid, batch_first=True,
                padding_value=self.out_pad, total_length=total_length
            )   # (n_sent, maxlen+1, dimh)
            
            return hid, (hid_n, None)

        else:
            if len(prev_h) < 1:
                prev_h = (self.init_hidden(len(x)), None)                       # (nlayer, n_sent, dim)
            hid, hid_n = self.gru(x, prev_h[0])                                 # (n_sent, 1, dimh), (nlayer, n_sent, dimh)
            
            return hid, (hid_n, None)

