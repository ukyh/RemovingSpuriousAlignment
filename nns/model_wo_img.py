import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, args, embs, embs_obj, vocab):
        super(Model, self).__init__()
        # embedder
        self.embedder = nn.Embedding(embs.shape[0], embs.shape[1], padding_idx=0)
        embs = torch.from_numpy(embs)[1:]
        pad = torch.zeros(1, embs.size(1))
        self.embedder.weight = nn.Parameter(torch.cat([pad, embs], dim=0))

        self.obj_embedder = nn.Embedding(embs_obj.shape[0], embs_obj.shape[1], padding_idx=0)
        embs_obj = torch.from_numpy(embs_obj)[1:]
        pad_obj = torch.zeros(1, embs_obj.size(1))
        self.obj_embedder.weight = nn.Parameter(torch.cat([pad_obj, embs_obj], dim=0))

        # sequence encoders
        if args.encoder == "lstm":
            from nns.rnns import LSTM
            self.seq_encoder = LSTM(args.dim_tok, args.dim_hid, args.nlayer_enc,
                drop_rate=0., out_pad=args.out_pad)
        elif args.encoder == "gru":
            from nns.rnns import GRU
            self.seq_encoder = GRU(args.dim_tok, args.dim_hid, args.nlayer_enc,
                drop_rate=0., out_pad=args.out_pad)
        else:
            raise KeyError("invalid encoder: {}".format(args.encoder))
        self.seq_dropout = nn.Dropout(p=args.dropout_seq)

        # image encoders
        self.img_input_encoder = nn.Linear(args.dim_tok, args.dim_tok)
        self.img_key_encoder = nn.Linear(args.dim_tok, args.dim_hid)
        self.img_val_encoder = nn.Linear(args.dim_tok, args.dim_hid)

        self.pred_linear = nn.Linear(args.dim_hid, self.embedder.weight.size(0))

        self.device = args.device
        self.norm_img = args.norm_img
        self.use_gate = args.use_gate
        self.use_unique = args.use_unique
        self.decode_size = args.decode_size
        self.beam_size = args.beam_size
        self.beam_len_norm = args.beam_len_norm
        self.beam_with_gate = args.beam_with_gate
        self.vocab_size = self.embedder.weight.size(0)
        self.eos_idx = vocab.eos
        self.obj_class = vocab.obj_class

    def get_batch(self, batch):
        pair_texts, pair_lengths, pair_spans, \
            obj1_texts, obj1_lengths, obj1_spans, \
            obj2_texts, obj2_lengths, obj2_spans, \
            pair_feats, obj1_feats, obj2_feats, \
            pair_num_objs, obj1_num_objs, obj2_num_objs = batch
        pair_texts = pair_texts.to(self.device) # (batch, maxlen)
        pair_lengths = pair_lengths.to(self.device) # (batch)
        pair_spans = pair_spans.to(self.device) # (batch, maxlen)
        obj1_texts = obj1_texts.to(self.device)
        obj1_lengths = obj1_lengths.to(self.device)
        obj1_spans = obj1_spans.to(self.device)
        obj2_texts = obj2_texts.to(self.device)
        obj2_lengths = obj2_lengths.to(self.device)
        obj2_spans = obj2_spans.to(self.device)
        pair_feats = pair_feats.to(self.device) # (batch, 1, dim)
        obj1_feats = obj1_feats.to(self.device)
        obj2_feats = obj2_feats.to(self.device)
        pair_num_objs = pair_num_objs.to(self.device)
        obj1_num_objs = obj1_num_objs.to(self.device)
        obj2_num_objs = obj2_num_objs.to(self.device)
        return pair_texts, pair_lengths, pair_spans, \
            obj1_texts, obj1_lengths, obj1_spans, \
            obj2_texts, obj2_lengths, obj2_spans, \
            pair_feats, obj1_feats, obj2_feats, \
            pair_num_objs, obj1_num_objs, obj2_num_objs

    def enc_img_train(self, img_feats, num_objs):
        feat_embs = self.obj_embedder(img_feats).sum(dim=1, keepdim=True)  # (batch, 1, dimt)
        feat_embs = feat_embs / num_objs.unsqueeze(1).unsqueeze(2).expand_as(feat_embs)
        img_init = self.img_input_encoder(feat_embs)   # (batch, 1, dimt)
        img_keys = self.img_key_encoder(img_init).tanh()  # (batch, 1, dimh)
        img_vals = self.img_val_encoder(img_init)  # (batch, 1, dimh)
        if self.norm_img:
            img_init = F.normalize(img_init, p=2, dim=-1)
            img_vals = F.normalize(img_vals, p=2, dim=-1)
        return img_init, img_keys, img_vals

    def enc_img_eval(self, img_feats, num_objs):
        feat_embs = self.obj_embedder(img_feats).sum(dim=1, keepdim=True)  # (batch, 1, dimt)
        feat_embs = feat_embs / num_objs.unsqueeze(1).unsqueeze(2).expand_as(feat_embs)
        img_init = self.img_input_encoder(feat_embs)   # (batch, 1, dimt)
        img_keys = self.img_key_encoder(img_init).tanh()  # (batch, 1, dimh)
        img_vals = self.img_val_encoder(img_init)  # (batch, 1, dimh)
        if self.norm_img:
            img_init = F.normalize(img_init, p=2, dim=-1)
            img_vals = F.normalize(img_vals, p=2, dim=-1)
        if self.beam_size > 1:
            dimt = img_init.size(2)
            dimh = img_keys.size(2)
            img_init = img_init.expand(-1, self.beam_size, -1).contiguous().view(-1, 1, dimt) # (batch * beam, 1, dimt)
            img_keys = img_keys.expand(-1, self.beam_size, -1).contiguous().view(-1, 1, dimh) # (batch * beam, 1, dimh)
            img_vals = img_vals.expand(-1, self.beam_size, -1).contiguous().view(-1, 1, dimh) # (batch * beam, 1, dimh)
        return img_init, img_keys, img_vals    # (batch (* beam), 1, dimh or dimt)

    def enc_seq(self, seq, seq_lengths, img_init):
        token_embs = self.embedder(seq)  # (batch, maxlen, dimt)
        full_token_embs = torch.cat([img_init, token_embs], dim=1)   # (batch, maxlen+1, dimt)
        # NOTE: since maxlen includes "<eos>",
        #       we can safely omit the maxlen+1-th hidden, which corresponds to the hidden given "<eos>"
        seq_embs, _ = self.seq_encoder(full_token_embs, seq_lengths, (), "train")  # (batch, maxlen, dimh)
        return seq_embs

    def calc_gate(self, seq_embs, img_keys):
        """
        Input:
            seq_embs: (batch (* beam), x, dimh)
            img_keys: (batch (* beam), 1, dimh)
        
        Output:
            gate_logit: (batch (* beam), x, 1)
        
        x = maxlen in train and x = 1 in eval
        """
        gate_logit = torch.bmm(seq_embs, img_keys.transpose(1, 2))
        return gate_logit

    def predict(self, seq_embs, img_vals, gate_logit):
        """
        Input:
            seq_embs: (batch (* beam), x, dimh)
            img_vals: (batch (* beam), 1, dimh)
            gate_logit: (batch (* beam), x, 1)
        
        Output:
            (batch (* beam), x, vocab)
        
        x = maxlen in train and x = 1 in eval
        """
        if self.use_gate:
            _img_vals = img_vals.expand_as(seq_embs)    # (batch (* beam), x, dimh)
            gate = gate_logit.sigmoid().expand_as(seq_embs)    # (batch (* beam), x, dimh)
            out_feats = gate * _img_vals + (1. - gate) * seq_embs
        else:
            out_feats = seq_embs
        pred_map = self.pred_linear(out_feats)    # (batch (* beam), x, vocab)
        return pred_map

    def reshape_for_loss(self, pred_map, seqs, lengths, gate_logit, spans):
        """Remove paddings from the loss calculation.

        Input:
            pred_map: (batch, maxlen, vocab)
            seqs: (batch, maxlen)            
            lengths: (batch)
            gate_logit: (batch, maxlen, 1)
            spans: (batch, maxlen)
        
        Output:
            preds: (sum(slen), vocab)
            gts: (sum(slen))
            gate_preds: (sum(slen))
            gate_label: (sum(slen))
        """
        preds = torch.cat([pred[:slen] for pred, slen in zip(pred_map, lengths)], dim=0) # (sum(slen), vocab)
        gts = torch.cat([seq[:slen] for seq, slen in zip(seqs, lengths)], dim=0) # (sum(slen))
        
        _gate_logit = gate_logit.squeeze(2) # (batch, maxlen)
        gate_preds = []
        gate_label = []
        for logit, span, slen in zip(_gate_logit, spans, lengths):
            gate_preds.append(logit[:slen])
            gate_label.append(span[:slen])
        gate_preds = torch.cat(gate_preds, dim=0)    # (sum(slen))
        gate_label = torch.cat(gate_label, dim=0).float()    # (sum(slen))
        return preds, gts, gate_preds, gate_label

    def greedy_search(self, batch, mode):
        """
        Input:
            batch (= img_feats): (batch, 1, dimi)
            mode: str
        
        Output:
            pred_seqs: [[decode_size] * batch]
        """
        img_feats, num_objs = batch
        img_init, img_keys, img_vals = self.enc_img_eval(
            img_feats.to(self.device), num_objs.to(self.device)
        ) # (batch, 1, dimt or dimh)
        prev_outs = img_init   # (batch, 1, dimt)
        prev_hc = ()
        prev_objs = [[] for _ in range(prev_outs.size(0))]   # [[] * batch]
        pred_seqs = []
        for t in range(self.decode_size):
            next_outs, next_hc = self.seq_encoder(prev_outs, [], prev_hc, mode) # (batch, 1, dimh), ((nlayer, batch, dimh) * 2)
            gate_logit = self.calc_gate(next_outs, img_keys) # (batch, 1, 1)
            next_pred = self.predict(next_outs, img_vals, gate_logit)  # (batch, 1, vocab)
            if self.use_unique and t > 0:
                for i, obj_idxs in enumerate(prev_objs):
                    next_pred[i][0][obj_idxs] = -1e+20  # exclude previously generated objects (safely ignore empty obj_idxs)
            next_pred = tuple(torch.max(pred, dim=-1)[1] for pred in next_pred) # (1 * batch)
            if self.use_unique:
                for i, obj_idx in enumerate(next_pred):   # one-element tensor: tensor([single_val])
                    if obj_idx.item() in self.obj_class:
                        prev_objs[i].append(obj_idx.item())
            pred_seqs.append(next_pred)
            prev_outs = self.embedder(torch.stack(next_pred))   # (batch, 1, dimt)
            prev_hc = next_hc
        pred_seqs = list(zip(*pred_seqs))   # ((1 * decode_size) * batch)
        pred_seqs = [[tok.item() for tok in seq] for seq in pred_seqs]  # [[decode_size] * batch]
        return pred_seqs

    def beam_search(self, batch, mode):
        """
        Input:
            batch (= img_feats): (batch, 1, dimi)
            mode: str
        
        Output:
            pred_seqs: [[slen] * batch]
        """
        img_feats, num_objs = batch
        batch_size = img_feats.size(0)
        img_init, img_keys, img_vals = self.enc_img_eval(
            img_feats.to(self.device), num_objs.to(self.device)
        ) # (batch * beam, 1, dimt or dimh)
        prev_outs = img_init   # (batch * beam, 1, dimh)
        prev_hc = ()
        prev_seqs = [[[] for __ in range(self.beam_size)] for _ in range(batch_size)]   # [batch, beam]
        prev_probs = [[0. for __ in range(self.beam_size)] for _ in range(batch_size)]   # [batch, [0.] * beam]
        prev_objs = [[[] for __ in range(self.beam_size)] for _ in range(batch_size)]   # [batch, beam]
        comp_seqs = [[] for _ in range(batch_size)]   # [[] * batch]
        for t in range(self.decode_size):
            next_outs, next_hc = self.seq_encoder(prev_outs, [], prev_hc, mode) # (batch * beam, 1, dimh), ((nlayer, batch * beam, dimh) * 2)
            gate_logit = self.calc_gate(next_outs, img_keys) # (batch * beam, 1, 1)
            next_pred = self.predict(next_outs, img_vals, gate_logit)  # (batch * beam, 1, vocab)
            # NOTE: Current implementation is `1`
            #   1. exclude prev_obj BEFORE log_softmax, not to penalize generating 2nd- objects
            #   2. exclude prev_obj AFTER log_softmax, not to generate unconfident 2nd- objects
            # next_pred = next_pred.log_softmax(dim=-1)
            if self.use_unique and t > 0:  # exclusion
                for i in range(next_pred.size(0)):
                    batch_idx = i // self.beam_size
                    beam_idx = i % self.beam_size
                    next_pred[i][0][prev_objs[batch_idx][beam_idx]] = -1e+20  # exclude previously generated objects (safely ignore empty obj_idxs)
            next_pred = next_pred.log_softmax(dim=-1)
            curr_outs = []
            curr_hc = []
            for i in range(batch_size):
                if t > 0:
                    flat_pred = next_pred[i * self.beam_size:(i + 1) * self.beam_size].view(-1)   # (beam * vocab)
                else:
                    flat_pred = next_pred[i * self.beam_size:i * self.beam_size + 1].view(-1)   # (vocab), to avoid the same word decode (all beam is the same at t == 0)
                prob, index = flat_pred.topk(k=self.beam_size, dim=0, largest=True, sorted=True)    # (beam), (beam)
                curr_seqs = []
                curr_probs = []
                curr_objs = []
                for pr, idx in zip(prob, index):
                    beam_idx = idx.item() // self.vocab_size
                    tok_idx = idx.item() % self.vocab_size
                    gate_prob = F.logsigmoid(gate_logit[i * self.beam_size + beam_idx]).squeeze().item()
                    curr_seqs.append(prev_seqs[i][beam_idx] + [tok_idx])
                    length_norm = len(curr_seqs[-1]) ** self.beam_len_norm
                    if tok_idx == self.eos_idx:
                        curr_probs.append(prev_probs[i][beam_idx] + pr.item() + gate_prob * self.beam_with_gate - 1e+20)
                        end_prob = (prev_probs[i][beam_idx] + pr.item() + gate_prob * self.beam_with_gate) / length_norm
                        comp_seqs[i].append([curr_seqs[-1], end_prob])    # [[[seq1, pr], ...], ...]
                    elif t + 1 == self.decode_size and len(comp_seqs[i]) == 0:  # when there is no complete sentence
                        curr_probs.append(prev_probs[i][beam_idx] + pr.item() + gate_prob * self.beam_with_gate - 1e+20)
                        end_prob = prev_probs[i][beam_idx] + pr.item() + gate_prob * self.beam_with_gate    # w/o length normalization to penalize partial sequences
                        comp_seqs[i].append([curr_seqs[-1], end_prob])    # [[[seq1, pr], ...], ...]
                    else:
                        curr_probs.append(prev_probs[i][beam_idx] + pr.item() + gate_prob * self.beam_with_gate)
                    if self.use_unique:
                        if tok_idx in self.obj_class:
                            curr_objs.append(prev_objs[i][beam_idx] + [tok_idx])
                        else:
                            curr_objs.append(prev_objs[i][beam_idx])
                    curr_outs.append(tok_idx)
                    curr_hc.append(self.beam_size * i + beam_idx)
                prev_seqs[i] = curr_seqs
                prev_probs[i] = curr_probs
                prev_objs[i] = curr_objs
            prev_outs = self.embedder(torch.LongTensor(curr_outs).to(self.device)).unsqueeze(1)   # (batch * beam, 1, dimt)
            if next_hc[1] != None:  # LSTM
                prev_hc = (
                    next_hc[0].transpose(0, 1)[curr_hc].transpose(0, 1), 
                    next_hc[1].transpose(0, 1)[curr_hc].transpose(0, 1)
                )   # ((nlayer, batch * beam, dimh) * 2)
            else:   # GRU
                prev_hc = (
                    next_hc[0].transpose(0, 1)[curr_hc].transpose(0, 1), 
                    None
                )   # ((nlayer, batch * beam, dimh), None)
        pred_seqs = []
        for seqs in comp_seqs:
            sort_seqs = sorted(seqs, key=lambda x:x[1], reverse=True)
            pred_seqs.append(sort_seqs[0][0])  # [[slen] * batch]
        return pred_seqs

    def forward(self, batch, mode):
        if mode == "train":
            pair_texts, pair_lengths, pair_spans, \
                obj1_texts, obj1_lengths, obj1_spans, \
                obj2_texts, obj2_lengths, obj2_spans, \
                pair_feats, obj1_feats, obj2_feats, \
                pair_num_objs, obj1_num_objs, obj2_num_objs = self.get_batch(batch)

            pair_img_init, pair_img_keys, pair_img_vals = self.enc_img_train(pair_feats, pair_num_objs)
            obj1_img_init, obj1_img_keys, obj1_img_vals = self.enc_img_train(obj1_feats, obj1_num_objs)
            obj2_img_init, obj2_img_keys, obj2_img_vals = self.enc_img_train(obj2_feats, obj2_num_objs)

            pair_seq_embs = self.enc_seq(pair_texts, pair_lengths, pair_img_init)
            obj1_seq_embs = self.enc_seq(obj1_texts, obj1_lengths, obj1_img_init)
            obj2_seq_embs = self.enc_seq(obj2_texts, obj2_lengths, obj2_img_init)

            pair_gate_logit = self.calc_gate(pair_seq_embs, pair_img_keys)
            obj1_gate_logit = self.calc_gate(obj1_seq_embs, obj1_img_keys)
            obj2_gate_logit = self.calc_gate(obj2_seq_embs, obj2_img_keys)

            pair_pred_map = self.predict(self.seq_dropout(pair_seq_embs), pair_img_vals, pair_gate_logit)
            obj1_pred_map = self.predict(self.seq_dropout(obj1_seq_embs), obj1_img_vals, obj1_gate_logit)
            obj2_pred_map = self.predict(self.seq_dropout(obj2_seq_embs), obj2_img_vals, obj2_gate_logit)

            pair_preds, pair_gts, pair_gate_preds, pair_gate_label \
                = self.reshape_for_loss(pair_pred_map, pair_texts, pair_lengths, pair_gate_logit, pair_spans)
            obj1_preds, obj1_gts, obj1_gate_preds, obj1_gate_label \
                = self.reshape_for_loss(obj1_pred_map, obj1_texts, obj1_lengths, obj1_gate_logit, obj1_spans)
            obj2_preds, obj2_gts, obj2_gate_preds, obj2_gate_label \
                = self.reshape_for_loss(obj2_pred_map, obj2_texts, obj2_lengths, obj2_gate_logit, obj2_spans)

            return pair_preds, pair_gts, pair_gate_preds, pair_gate_label, \
                obj1_preds, obj1_gts, obj1_gate_preds, obj1_gate_label, \
                obj2_preds, obj2_gts, obj2_gate_preds, obj2_gate_label
        
        else:
            if self.beam_size > 1:
                pred_seqs = self.beam_search(batch, mode)
            else:
                pred_seqs = self.greedy_search(batch, mode)

            return pred_seqs

    def analyze(self, batch, mode):
        """
        Input:
            batch (= img_feats): (batch, 1, dimi)
            mode: str
        
        Output:
            pred_seqs: [[decode_size] * batch]
        """
        img_feats, num_objs = batch
        img_init, img_keys, img_vals = self.enc_img_eval(
            img_feats.to(self.device), num_objs.to(self.device)
        ) # (batch, 1, dimt or dimh)
        prev_outs = img_init   # (batch, 1, dimt)
        prev_hc = ()
        prev_objs = [[] for _ in range(prev_outs.size(0))]   # [[] * batch]
        pred_seqs = []
        img_norm_tensor = img_vals.squeeze(1).norm(p="fro", dim=-1) # (batch)
        img_norm_list = [inorm.item() for inorm in img_norm_tensor] # [batch]
        seq_norm_tensor = []
        gate_val_tensor = []
        pred_cand_tensor = []
        cand_prob_tensor = []
        for t in range(self.decode_size):
            next_outs, next_hc = self.seq_encoder(prev_outs, [], prev_hc, mode) # (batch, 1, dimh), ((nlayer, batch, dimh) * 2)
            gate_logit = self.calc_gate(next_outs, img_keys) # (batch, 1)
            next_pred = self.predict(next_outs, img_vals, gate_logit)  # (batch, 1, vocab)
            seq_norm_tensor.append(next_outs.squeeze().norm(p="fro", dim=1))    # [(batch), ...]
            gate_val_tensor.append(gate_logit.squeeze().sigmoid())  # [(batch), ...]
            cand_prob, pred_cand = next_pred.squeeze().softmax(dim=1).topk(k=8, dim=1, largest=True, sorted=True)
            pred_cand_tensor.append(pred_cand)    # [(batch, k), ...]
            cand_prob_tensor.append(cand_prob)    # [(batch, k), ...]
            if self.use_unique and t > 0:
                for i, obj_idxs in enumerate(prev_objs):
                    next_pred[i][0][obj_idxs] = -1e+20  # exclude previously generated objects (safely ignore empty obj_idxs)
            next_pred = tuple(torch.max(pred, dim=-1)[1] for pred in next_pred) # (1 * batch)
            if self.use_unique:
                for i, obj_idx in enumerate(next_pred):   # one-element tensor: tensor([single_val])
                    if obj_idx.item() in self.obj_class:
                        prev_objs[i].append(obj_idx.item())
            pred_seqs.append(next_pred)
            prev_outs = self.embedder(torch.stack(next_pred))   # (batch, 1, dimt)
            prev_hc = next_hc
        pred_seqs = list(zip(*pred_seqs))   # ((1 * decode_size) * batch)
        pred_seqs = [[tok.item() for tok in seq] for seq in pred_seqs]  # [[decode_size] * batch]
        seq_norm_tensor = torch.stack(seq_norm_tensor, dim=0).t()   # (batch, decode_size)
        seq_norm_list = [[snorm.item() for snorm in seq] for seq in seq_norm_tensor]    # [batch, decode_size]
        gate_val_tensor = torch.stack(gate_val_tensor, dim=0).t()   # (batch, decode_size)
        gate_val_list = [[gval.item() for gval in seq] for seq in gate_val_tensor]  # [batch, decode_size]
        pred_cand_tensor = torch.stack(pred_cand_tensor, dim=0).transpose(0, 1)   # (batch, decode_size, k)
        pred_cand_list = [[[k.item() for k in tok] for tok in seq] for seq in pred_cand_tensor]  # [batch, decode_size, k]
        cand_prob_tensor = torch.stack(cand_prob_tensor, dim=0).transpose(0, 1)   # (batch, decode_size, k)
        cand_prob_list = [[[k.item() for k in tok] for tok in seq] for seq in cand_prob_tensor]  # [batch, decode_size, k]
        return pred_seqs, img_norm_list, seq_norm_list, \
            gate_val_list, pred_cand_list, cand_prob_list

