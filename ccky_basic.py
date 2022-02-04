import json

import torch

class ConstrainedCKY(object):
    def __init__(self, net, word2idx, initial_scalar=1, mode='force_include', scalars_key='inside_xs_components'):
        super(ConstrainedCKY, self).__init__()
        self.net = net
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.initial_scalar = initial_scalar
        self.mode = mode
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
        batch_size = self.net.batch_size
        length = self.net.length
        scalars = self.net.cache[self.scalars_key].copy()
        device = self.net.device
        dtype = torch.float32

        # Assign missing scalars
        for i in range(length):
            scalars[0][i] = torch.full((batch_size, 1), self.initial_scalar, dtype=dtype, device=device)

        leaves = [None for _ in range(batch_size)]
        for i in range(batch_size):
            batch_i = batch[i].tolist()
            leaves[i] = [self.idx2word[idx] for idx in batch_i]

        trees, components = self.batched_cky(scalars, leaves, batch_span)

        return trees, components

    def initial_chart(self):
        batch_size = self.net.batch_size
        length = self.net.length
        device = self.net.device
        dtype = torch.float32
        # spans: [[[start,length],[]],[[],[],...],...]
        chart = [torch.full((length-i, batch_size), 0, dtype=dtype, device=device) for i in range(length)] #level, length, batch
        return chart

    def batched_cky(self, scalars, leaves, batch_span):
        batch_size = self.net.batch_size
        length = self.net.length
        device = self.net.device
        dtype = torch.float32

        # Chart.
        chart = self.initial_chart()
        components = {}

        # Backpointers.
        bp = {}
        for ib in range(batch_size):
            bp[ib] = [[None] * (length - i) for i in range(length)]
            bp[ib][0] = [i for i in range(length)]

        # Constraint.
        chart_constraint = {(0, pos): torch.LongTensor(batch_size).fill_(0) for pos in range(length)}
        #
        lookup = []
        for i_b in range(batch_size):
            lookup.append(set([(size - 1, pos) for pos, size, _ in batch_span[i_b] if size > 1 and size < length]))

        for level in range(1, length):
            L = length - level
            N = level

            for pos in range(L):

                pairs, lps, rps, sps = [], [], [], []

                # Assumes that the bottom-left most leaf is in the first constituent.
                spbatch = scalars[level][pos]

                # Book-keeping.
                zs = []

                for idx in range(N):
                    # (level, pos)
                    l_level = idx
                    l_pos = pos
                    r_level = level-idx-1
                    r_pos = pos+idx+1

                    l_size, r_size = l_level + 1, r_level + 1

                    # assert l_level >= 0
                    # assert l_pos >= 0
                    # assert r_level >= 0
                    # assert r_pos >= 0

                    l = (l_level, l_pos)
                    r = (r_level, r_pos)

                    x_constraint = chart_constraint[(l_level, l_pos)] + chart_constraint[(r_level, r_pos)]
                    zs.append(x_constraint.view(-1, 1))

                    lp = chart[l_level][l_pos].view(-1, 1)
                    rp = chart[r_level][r_pos].view(-1, 1)
                    sp = spbatch[:, idx].view(-1, 1)

                    lps.append(lp)
                    rps.append(rp)
                    sps.append(sp)

                    pairs.append((l, r))

                lps, rps, sps = torch.cat(lps, 1), torch.cat(rps, 1), torch.cat(sps, 1)
                ps = lps + rps + sps
                components[(level, pos)] = ps

                ps_for_argmax = ps.clone()

                # Force constraint.
                zs = torch.cat(zs, 1)
                if self.mode == 'force_include':
                    zmax = zs.max(1, keepdim=True).values
                    ps_for_argmax[zs < zmax] = ps.min()

                elif self.mode == 'force_exclude':
                    zmin = zs.min(1, keepdim=True).values
                    ps_for_argmax[zs > zmin] = ps.min()

                elif self.mode == 'count_include':
                    ps_for_argmax = ps_for_argmax + zs

                elif self.mode == 'discount_include':
                    ps_for_argmax = ps_for_argmax - zs

                # Compute argmax.
                argmax = ps_for_argmax.argmax(1).long()

                # Update constraint count.
                local_z = zs[range(batch_size), argmax]
                for i_b in range(batch_size):
                    if (level, pos) in lookup[i_b]:
                        local_z[i_b] += 1
                chart_constraint[(level, pos)] = local_z

                # Update chart with value.
                valmax = ps[range(batch_size), argmax]
                chart[level][pos, :] += valmax

                for i, idx in enumerate(argmax.tolist()):
                    bp[i][level][pos] = pairs[idx]

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




