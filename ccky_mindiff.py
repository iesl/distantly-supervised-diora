import torch
import torch.nn as nn
import numpy as np
from net_utils import *




class ConstrainedCKY(object):
    def __init__(self, net, word2idx, scalars_key='inside_xs_components', pred_weight=1000, constraint_weight=10000, initial_scalar=1):
        super(ConstrainedCKY, self).__init__()
        self.net = net
        self.batch_size = net.batch_size
        self.length = net.length
        self.device = net.device
        self.index = net.index
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.initial_scalar = initial_scalar
        self.pred_weight = pred_weight
        self.constraint_weight = constraint_weight
        self.scalars_key = scalars_key

    def predict(self, batch_map, return_components=False):
        batch = batch_map['sentences']
        example_ids = batch_map['example_ids']
        batch_span = batch_map['ner_labels']

        batch_size = self.net.batch_size

        trees, components = self.parse_batch(batch,batch_span)

        out = []
        for i in range(batch_size):
            assert trees[i] is not None
            out.append(dict(example_id=example_ids[i], binary_tree=trees[i]))

        if return_components:
            return (out, components)

        return out

    def parse_batch(self, batch, batch_span, cell_loss=False, return_components=False):
        batch_size = self.batch_size
        length = self.length
        scalars = self.net.cache[self.scalars_key].copy()
        device = self.device
        dtype = torch.float32

        # Assign missing scalars
        for i in range(length):
            scalars[0][i] = torch.full((batch_size, 1), self.initial_scalar, dtype=dtype, device=device)
        scalars = self.flatten_s(scalars)

        leaves = [None for _ in range(batch_size)]
        for i in range(batch_size):
            batch_i = batch[i].tolist()
            leaves[i] = [self.idx2word[idx] for idx in batch_i]

        trees, components = self.batched_cky(scalars, leaves, batch_span)

        return trees, components

    def flatten_s(self, s):
        length = self.length

        # Flatten the levels of `s`.
        flat_s = {}
        for level in range(0, length):
            L = length - level
            s_level = []
            s_ = s[level] # {pos: [B x N]}
            for pos in range(L):
                s_level.append(s_[pos])
            flat_s[level] = torch.stack(s_level, 1) # {level: [B x L x N]}

        return flat_s

    def initial_constrained_chart(self,batch_span):
        batch_size = self.batch_size
        length = self.length
        device = self.device
        dtype = torch.float32
        # spans: [[[start,length],[]],[[],[],...],...]
        offset_cache = self.index.get_offset(length)
        ncells = int(length * (1 + length) / 2)
        # print('length',length,'ncells',ncells)
        chart = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device) # batch, levelxlength, 1
        for idx, spans in enumerate(batch_span):
            for span in spans:
                level = span[1]-1
                if  level>0:
                    pos = offset_cache[level] + span[0]
                    chart[idx, pos] = self.constraint_weight
        return chart

    def initial_chart(self):
        batch_size = self.batch_size
        length = self.length
        device = self.device
        dtype = torch.float32
        ncells = int(length * (1 + length) / 2)
        chart = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)
        return chart

    def get_pred_chart(self, scalars):
        batch_size = self.batch_size
        length = self.length
        device = self.device
        dtype = torch.float32

        # Chart.
        chart = self.initial_chart()
        components = {}

        # Backpointers.
        bp = {}
        for ib in range(batch_size):
            bp[ib] = [[None] * (length - i) for i in range(length)]
            bp[ib][0] = [i for i in range(length)]

        for level in range(1, length):
            B = batch_size
            L = length - level
            N = level

            batch_info = BatchInfo(
                batch_size=batch_size,
                length=length,
                size=self.net.size,
                level=level,
                phase='inside',
                )
            ls, rs = get_inside_states(batch_info, chart, self.index, 1)
            xs = scalars[level]
            ps = ls.view(B,L,N,1) + rs.view(B,L,N,1) + xs #B,L,N,1
            offset = self.index.get_offset(length)[level]
            chart[:,offset:offset+L] += torch.max(ps,2)[0] #B, L
            argmax = ps.argmax(2).long() #B,L

            for pos in range(L):
                components[(level, pos)] = ps
                pairs = []
                #To Do: store pairs
                for idx in range(N):
                    l_level = idx
                    l_pos = pos
                    r_level = level-idx-1
                    r_pos = pos+idx+1
                    l = (l_level, l_pos)
                    r = (r_level, r_pos)
                    pairs.append((l, r))

                for i, ix in enumerate(argmax[:,pos].squeeze(-1).tolist()):
                    # print(ix)
                    bp[i][level][pos] = pairs[ix]

        ncells = int(length * (1 + length) / 2)
        pred_chart = torch.full((batch_size, ncells, 1), 0, dtype=dtype, device=device)
        for i in range(batch_size):
            # pred_chart = pred_charts #level, length , 1
            self.fill_chart(bp[i], pred_chart[i], bp[i][-1][0])

        return pred_chart

    def fill_chart(self, bp, pred_chart, pair):
        if isinstance(pair, int):
            return
        l, r = pair

        offset_cache = self.index.get_offset(self.length)
        pred_chart[offset_cache[l[0]] + l[1]] = self.pred_weight
        pred_chart[offset_cache[r[0]] + r[1]] = self.pred_weight
        self.fill_chart(bp, pred_chart, bp[l[0]][l[1]])
        self.fill_chart(bp, pred_chart, bp[r[0]][r[1]])


    def batched_cky(self, scalars, leaves, batch_span):
        batch_size = self.batch_size
        length = self.length
        device = self.device
        dtype = torch.float32

        # Chart.
        chart = self.initial_constrained_chart(batch_span)
        pred_chart = self.get_pred_chart(scalars)
        for lvl in range(len(pred_chart)):
            chart[lvl] += pred_chart[lvl]
        # print(chart)

        components = {}

        # Backpointers.
        bp = {}
        for ib in range(batch_size):
            bp[ib] = [[None] * (length - i) for i in range(length)]
            bp[ib][0] = [i for i in range(length)]

        for level in range(1, length):
            B = batch_size
            L = length - level
            N = level

            batch_info = BatchInfo(
                batch_size=batch_size,
                length=length,
                size=self.net.size,
                level=level,
                phase='inside',
                )
            ls, rs = get_inside_states(batch_info, chart, self.net.index, 1)
            xs = scalars[level]
            ps = ls.view(B,L,N,1) + rs.view(B,L,N,1) + xs #B,L,N,1
            offset = self.index.get_offset(length)[level]
            chart[:,offset:offset+L] += torch.max(ps,2)[0] #B, L
            argmax = ps.argmax(2).long() #B,L

            for pos in range(L):
                components[(level, pos)] = ps
                pairs = []

                #To Do: store pairs
                for idx in range(N):
                    l_level = idx
                    l_pos = pos
                    r_level = level-idx-1
                    r_pos = pos+idx+1
                    l = (l_level, l_pos)
                    r = (r_level, r_pos)
                    pairs.append((l, r))

                for i, ix in enumerate(argmax[:,pos].squeeze(-1).tolist()):
                    bp[i][level][pos] = pairs[ix]

        trees = []
        for i in range(batch_size):
            tree = self.follow_backpointers(bp[i], leaves[i], bp[i][-1][0])
            trees.append(tree)

        return trees, components

    def follow_backpointers(self, bp, words, pair):
        if isinstance(pair, int):
            return words[pair]

        l, r = pair
        lout = self.follow_backpointers(bp, words, bp[l[0]][l[1]])
        rout = self.follow_backpointers(bp, words, bp[r[0]][r[1]])

        return (lout, rout)