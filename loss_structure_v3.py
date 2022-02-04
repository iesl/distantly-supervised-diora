import torch
import torch.nn as nn
import numpy as np

from net_utils import *

from cky import ParsePredictor as CKY
from ccky_basic import ConstrainedCKY as CCKY_Basic
from constrained_cky import ConstrainedCKY as CCKY_MinDiff


def get_spans(tr, lookup):
    spans = []
    def helper(tr, pos):
        if isinstance(tr, str):
            return 1
        assert len(tr) == 2
        l_pos = pos
        l_size = helper(tr[0], l_pos)
        r_pos = l_pos + l_size
        r_size = helper(tr[1], r_pos)
        size = l_size + r_size
        idx = lookup[(l_pos, l_size, r_pos, r_size)]
        spans.append((pos, size, idx))
        return size
    helper(tr, 0)
    return spans


def build_inside_lookup(length):
    lookup = {}

    for level in range(1, length):
        L = length - level
        N = level
        for pos in range(L):
            for idx in range(N):
                l_level = idx
                l_pos = pos
                l_size = l_level + 1

                r_level = level-idx-1
                r_pos = pos+idx+1
                r_size = r_level + 1

                lookup[(l_pos, l_size, r_pos, r_size)] = idx

    return lookup


def is_crossing(query, spans):
    def is_crossing_(pos, size, pos_2, size_2):
        assert pos < pos_2
        if pos + size > pos_2:
            if (pos + size) < (pos_2 + size_2):
                return True
        return False

    pos, size = query
    for pos_2, size_2 in spans:
        if pos < pos_2 and is_crossing_(pos, size, pos_2, size_2):
            return True
        elif pos > pos_2 and is_crossing_(pos_2, size_2, pos, size):
            return True
    return False


class TreeStructureV3(nn.Module):
    name = 'tree_structure_v3'

    def __init__(self, embeddings, input_size, size, word2idx=None, cuda=False, print=False, skip=False,
                 **kwargs):
        super().__init__()
        self.embeddings = embeddings
        self.size = size
        self._cuda = cuda
        self.skip = skip
        self.word2idx = word2idx

        self.init_defaults()

        for k, v in kwargs.items():
            setattr(self, k, v)

    def init_defaults(self):
        self.margin = 1.0
        self.weight = 1.0
        self.ccky_mode = 'ccky_mindiff'
        self.original_weight = 1000
        self.constraint_weight = 10000
        self.force_exclude = False
        self.rescale = False
        self.scalars_key = 'inside_xs_components'

    @classmethod
    def from_kwargs_dict(cls, context, kwargs_dict):
        kwargs_dict['embeddings'] = context['embedding_layer']
        kwargs_dict['cuda'] = context['cuda']
        kwargs_dict['word2idx'] = context['word2idx']
        return cls(**kwargs_dict)

    def get_tree_scores(self, s, batch_span_lst):
        scores = []

        for i_b, span_lst in enumerate(batch_span_lst):

            score_ = torch.FloatTensor(1).fill_(0).to(self.device)
            for pos, size, idx in span_lst:
                assert size > 1

                level = size - 1
                score_ = score_ + s[level][pos][i_b, idx]
            scores.append(score_.view(1))

        return torch.cat(scores)

    def forward(self, sentences, diora, info, embed=None):
        # print(self.force_exclude, self.rescale, self.ccky_mode)

        if self.ccky_mode == 'ccky_mindiff' and self.force_exclude:
            raise Exception("Sorry, invalid cky mode")
        self.device = device = torch.cuda.current_device() if self._cuda else None
        self.batch_size, self.length = batch_size, length = sentences.shape
        size = self.size

        # TODO: Should we cache this?
        self.lookup = build_inside_lookup(length)

        # CKY
        cky_parser = CKY(net=diora, word2idx=self.word2idx, scalars_key=self.scalars_key)

        # Constrained CKY
        if self.ccky_mode == 'ccky_basic':
            ccky_parser = CCKY_Basic(
                net=diora, word2idx=self.word2idx, scalars_key=self.scalars_key)
        elif self.ccky_mode == 'ccky_mindiff':
            ccky_parser = CCKY_MinDiff(
                net=diora, word2idx=self.word2idx, scalars_key=self.scalars_key, pred_weight=self.original_weight, constraint_weight=self.constraint_weight)

        neg_parser = cky_parser
        if self.force_exclude:
            neg_parser = ccky_parser
        pos_parser = ccky_parser

        constraints = []
        for lst in info['constraints']:
            constraints.append([sp for sp in lst if sp[1] > 1])

        batch_map = {}
        batch_map['sentences'] = sentences
        batch_map['example_ids'] = info['example_ids']
        batch_map['ner_labels'] = constraints

        # Score components.
        score_components = diora.cache[self.scalars_key]

        # First, get the bad trees.
        neg_parser.mode = 'force_exclude'
        with torch.no_grad():
            pred_trees, pred_spans = [], []
            for j, x in enumerate(neg_parser.predict(batch_map)):
                spans = get_spans(x['binary_tree'], self.lookup)
                assert spans[-1][1] == length
                pred_trees.append(x['binary_tree'])
                pred_spans.append(set(spans))

        # Then, get the constrained trees.
        pos_parser.mode = 'force_include'
        with torch.no_grad():
            constrained_trees, constrained_spans = [], []
            for j, x in enumerate(pos_parser.predict(batch_map)):
                spans = get_spans(x['binary_tree'], self.lookup)
                assert spans[-1][1] == length
                constrained_trees.append(x['binary_tree'])
                constrained_spans.append(set(spans))

        pred_scores = self.get_tree_scores(score_components, pred_spans)
        ccky_scores = self.get_tree_scores(score_components, constrained_spans)

        # Mask sentences based on relevance.
        mask_is_different = torch.BoolTensor([a != b for a, b in zip(pred_spans, constrained_spans)]).to(device)
        mask_has_constraint = torch.BoolTensor([len(lst) > 0 for lst in constraints]).to(device)
        mask = torch.logical_and(mask_is_different, mask_has_constraint)

        # Compute margin-based loss.
        hinge_b = torch.clamp(self.margin + pred_scores - ccky_scores, min=0)

        # Optionally, rescale loss according to distance between trees.
        if self.rescale:
            g = torch.FloatTensor(batch_size).fill_(1).to(device)

            for i_b, (constr, pred) in enumerate(zip(constrained_spans, pred_spans)):
                total = len(constr)
                correct = len(set.intersection(constr, pred))
                g[i_b] = correct / total

            hinge_b = hinge_b * g

        # Apply mask.
        if mask.any().item():
            tr_loss = hinge_b[mask].mean()
        else:
            tr_loss = torch.FloatTensor(1).fill_(0).to(device)

        loss = tr_loss * self.weight
        ret = {}
        ret[self.name + '_loss'] = loss

        return loss, ret


structure_v3_class = TreeStructureV3
