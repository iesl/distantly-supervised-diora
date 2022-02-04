import collections
import json
import os
import subprocess

import nltk
from nltk.tree import Tree
from nltk.treeprettyprinter import TreePrettyPrinter
import numpy as np
import torch
from tqdm import tqdm

from cky import ParsePredictor as CKY
from experiment_logger import get_logger
from evaluation_utils import BaseEvalFunc


word_tags = set(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
               'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
               'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
               'WDT', 'WP', 'WP$', 'WRB'])


class ConstraintCKY(object):
    def __init__(self, net, word2idx, scalars_key='inside_s_components', initial_scalar=1):
        super(ConstraintCKY, self).__init__()
        self.net = net
        self.idx2word = {v: k for k, v in word2idx.items()}
        self.initial_scalar = initial_scalar
        self.scalars_key = scalars_key

    def predict(self, batch_map, return_components=False):
        def filter_span(span_lst):
            def filter_(lst):
                return [sp for sp in lst if sp[1] > 1]
            return [filter_(lst) for lst in span_lst]
        batch = batch_map['sentences']
        example_ids = batch_map['example_ids']
        #batch_span = [gt['ner_span'] for gt in batch_map['ground_truth']]
        batch_span = filter_span(batch_map['ner_labels'])

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

    def initial_chart(self,batch_span):
        batch_size = self.net.batch_size
        length = self.net.length
        device = self.net.device
        dtype = torch.float32
        # spans: [[[start,length],[]],[[],[],...],...]
        chart = [torch.full((length-i, batch_size), 1, dtype=dtype, device=device) for i in range(length)]
        for idx, spans in enumerate(batch_span):
            for span in spans:
                level = span[1]-1
                assert level>0
                pos = span[0]
                # cross = range(min(0,pos-level),max(length-level,pos+level))
                # for i in cross:
                # chart[level][min(0,pos-level):max(length-level,pos+level),idx] = float('-inf')
                chart[level][pos,idx] = 10000.0
        return chart



    def batched_cky(self, scalars, leaves, batch_span):
        batch_size = self.net.batch_size
        length = self.net.length
        device = self.net.device
        dtype = torch.float32

        # Chart.
        chart = self.initial_chart(batch_span)
        components = {}

        # Backpointers.
        bp = {}
        for ib in range(batch_size):
            bp[ib] = [[None] * (length - i) for i in range(length)]
            bp[ib][0] = [i for i in range(length)]

        for level in range(1, length):
            L = length - level
            N = level

            for pos in range(L):

                pairs, lps, rps, sps = [], [], [], []

                # Assumes that the bottom-left most leaf is in the first constituent.
                spbatch = scalars[level][pos]

                for idx in range(N):
                    # (level, pos)
                    l_level = idx
                    l_pos = pos
                    r_level = level-idx-1
                    r_pos = pos+idx+1

                    # assert l_level >= 0
                    # assert l_pos >= 0
                    # assert r_level >= 0
                    # assert r_pos >= 0

                    l = (l_level, l_pos)
                    r = (r_level, r_pos)

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
                argmax = ps.argmax(1).long()

                valmax = ps[range(batch_size), argmax]
                chart[level][pos, :] += valmax

                for i, ix in enumerate(argmax.tolist()):
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

def to_raw_parse(tr, tokens):
    def helper(tr, pos=0):
        if isinstance(tr, (str, int)):
            size = 1
            return '(DT {})'.format(tokens[pos]), size
        nodes = []
        size = 0
        for x in tr:
            xnode, xsize = helper(x, pos + size)
            nodes.append(xnode)
            size += xsize
        return '(S {})'.format(' '.join(nodes)), size
    node, _ = helper(tr)
    return '(ROOT {})'.format(node)


def to_raw_parse_nopunct(tr, tokens, part_of_speech):
    mask = [x in word_tags for x in part_of_speech]
    new_tokens = [x for x, m in zip(tokens, mask) if m]
    new_tr, kept, removed = remove_using_flat_mask_nary_tree(tr, mask)
    return to_raw_parse(new_tr, new_tokens)


def gt_to_raw_parse_nopunct(tr, tokens, part_of_speech):
    mask = [x in word_tags for x in part_of_speech]
    new_tokens = [x for x, m in zip(tokens, mask) if m]
    new_tr, kept, removed = gt_remove_using_flat_mask_nary_tree(tr, mask)
    return new_tr.pformat(margin=10000)


def gt_remove_using_flat_mask_nary_tree(tr, mask):
    """
    Input:
        tr: A tree such as (ROOT (S (X a) (G (Y b) (Z c) (Z d)))).
        mask: Boolean mask with length same as tree leaves
              such as [True, False, True, True].

    Returns:
        A new tree with tokens removed according to mask
        such as (ROOT (S (X a) (G (Z c) (Z d)))).
    """
    kept, removed = [], []
    def func(tr, pos=0):
        if len(tr) == 1 and isinstance(tr[0], (int, str)):
            if mask[pos] == False:
                removed.append(tr)
                return None, 1
            kept.append(tr)
            return tr, 1

        size = 0
        children = []

        for x in tr:
            xnode, xsize = func(x, pos=pos + size)
            if xnode is not None:
                children.append(xnode)
            size += xsize
        if len(children) == 1:
            return children[0], size
        if len(children) == 0:
            return None, size
        new_tr = nltk.Tree(tr.label(), children=children)
        return new_tr, size
    new_tree, _ = func(tr)
    return new_tree, kept, removed


def remove_using_flat_mask_nary_tree(tr, mask):
    """
    Input:
        tr: A tree such as (ROOT (S (X a) (G (Y b) (Z c) (Z d)))).
        mask: Boolean mask with length same as tree leaves
              such as [True, False, True, True].

    Returns:
        A new tree with tokens removed according to mask
        such as (ROOT (S (X a) (G (Z c) (Z d)))).
    """
    kept, removed = [], []
    def func(tr, pos=0):
        if not isinstance(tr, (list, tuple)):
            if mask[pos] == False:
                removed.append(tr)
                return None, 1
            kept.append(tr)
            return tr, 1

        size = 0
        node = []

        for subtree in tr:
            x, xsize = func(subtree, pos=pos + size)
            if x is not None:
                node.append(x)
            size += xsize

        for x in node:
            if isinstance(x, (list, tuple)):
                assert len(x) > 1

        if len(node) == 1:
            node = node[0]
        elif len(node) == 0:
            return None, size
        if isinstance(node, list):
            node = tuple(node)
        return node, size
    new_tree, _ = func(tr)
    return new_tree, kept, removed


def convert_to_nltk(tr, label='|'):
    def helper(tr):
        if not isinstance(tr, (list, tuple)):
            return '({} {})'.format(label, tr)
        nodes = []
        for x in tr:
            nodes.append(helper(x))
        return '({} {})'.format(label, ' '.join(nodes))
    return helper(tr)


def example_f1(gt, pred):
    correct = len(gt.intersection(pred))
    if correct == 0:
        return 0., 0., 0.
    gt_total = len(gt)
    pred_total = len(pred)
    prec = float(correct) / pred_total
    recall = float(correct) / gt_total
    f1 = 2 * (prec * recall) / (prec + recall)
    return f1, prec, recall


def tree_to_spans(tree):
    spans = []

    def helper(tr, pos):
        if not isinstance(tr, (list, tuple)):
            size = 1
            return size
        elif isinstance(tr, Tree) and len(tr.leaves()) == 1:
            size = 1
            return size
        size = 0
        for x in tr:
            xpos = pos + size
            xsize = helper(x, xpos)
            size += xsize
        spans.append((pos, size))
        return size

    helper(tree, 0)

    return spans

# def tree_to_spans(tree):
#     spans = []

#     def helper(tr, pos):
#         if not isinstance(tr, (list, tuple)):
#             size = 1
#             return size
#         size = 0
#         for x in tr:
#             xpos = pos + size
#             xsize = helper(x, xpos)
#             size += xsize
#         spans.append((pos, size))
#         return size

#     helper(tree, 0)

#     return spans


def spans_to_tree(spans, tokens):
    length = len(tokens)

    # Add missing spans.
    span_set = set(spans)
    for pos in range(length):
        if pos not in span_set:
            spans.append((pos, 1))

    spans = sorted(spans, key=lambda x: (x[1], x[0]))

    pos_to_node = {}
    root_node = None

    for i, span in enumerate(spans):

        pos, size = span

        if i < length:
            assert i == pos
            node = (pos, size, tokens[i])
            pos_to_node[pos] = node
            continue

        node = (pos, size, [])

        for i_pos in range(pos, pos+size):
            child = pos_to_node[i_pos]
            c_pos, c_size = child[0], child[1]

            if i_pos == c_pos:
                node[2].append(child)
            pos_to_node[i_pos] = node

    def helper(node):
        pos, size, x = node
        if isinstance(x, str):
            return x
        return tuple([helper(xx) for xx in x])

    root_node = pos_to_node[0]
    tree = helper(root_node)

    return tree


class TreesFromDiora(object):
    def __init__(self, diora, word2idx, outside, oracle):
        self.diora = diora
        self.word2idx = word2idx
        self.idx2word = {idx: w for w, idx in word2idx.items()}
        self.outside = outside
        self.oracle = oracle

    def to_spans(self, lst):
        return [(pos, level + 1) for level, pos in lst]

    def predict(self, batch_map):
        batch_size, length = batch_map['sentences'].shape
        example_ids = batch_map['example_ids']
        tscores = [0.0] * batch_size
        K = self.diora.K

        for i_b in range(batch_size):
            tokens = batch_map['ground_truth'][i_b]['tokens']
            root_level, root_pos = length - 1, 0
            spans = self.to_spans(self.diora.cache['inside_tree'][(i_b, 0)][(root_level, root_pos)])
            binary_tree = spans_to_tree(spans, tokens)
            other_trees = []

            yield dict(example_id=example_ids[i_b], binary_tree=binary_tree, binary_tree_score=tscores[i_b], other_trees=other_trees)


class NERComponent(BaseEvalFunc):

    def init_defaults(self):
        self.agg_mode = 'sum'
        self.cky_mode = 'sum'
        self.ground_truth = None
        self.inside_pool = 'sum'
        self.oracle = {'use': False}
        self.outside = True
        self.seed = 121
        self.semi_supervised = False
        self.K = None
        self.choose_tree = 'local'

    def compare(self, prev_best, results):
        out = []
        # F1
        key = 'f1'
        best_dict_key = 'best__{}__{}'.format(self.name, key)
        val = results['meta'][key]
        is_best = True
        if best_dict_key in prev_best:
            prev_val = prev_best[best_dict_key]['value']
            is_best = prev_val < val
        out.append((key, val, is_best))
        #
        return out

    def parse(self, trainer, info):
        logger = self.logger

        multilayer = False
        diora = trainer.get_single_net(trainer.net).diora
        if hasattr(diora, 'layers'):
            multilayer = True
            pred_lst = []
            for i, layer in enumerate(diora.layers):
                logger.info(f'Diora Layer {i}:')
                pred = self.single_layer_parser(trainer, layer, info)
                pred_lst.append(pred)
        else:
            pred_lst = self.single_layer_parser(trainer, diora, info)
        return pred_lst, multilayer

    def single_layer_parser(self, trainer, diora, info):
        logger = self.logger
        epoch = info.get('epoch', 0)

        original_K = diora.K
        if self.K is not None:
            diora.safe_set_K(self.K)

        # set choose_tree
        if hasattr(diora, 'choose_tree'):
            original_choose_tree = diora.choose_tree
            diora.choose_tree = self.choose_tree

        word2idx = self.dataset['word2idx']
        if self.cky_mode == 'cky':
            parse_predictor = CKY(net=diora, word2idx=word2idx)
        elif self.cky_mode == 'constrained_cky':
            parse_predictor = ConstraintCKY(net=diora, word2idx=word2idx)
        elif self.cky_mode == 'diora':
            parse_predictor = TreesFromDiora(diora=diora, word2idx=word2idx, outside=self.outside, oracle=self.oracle)

        batches = self.batch_iterator.get_iterator(random_seed=self.seed, epoch=epoch)

        logger.info('Parsing.')

        pred_lst = []
        counter = 0
        eval_cache = {}

        if self.ground_truth is not None:
            self.ground_truth = os.path.expanduser(self.ground_truth)
            ground_truth_data = {}
            with open(self.ground_truth) as f:
                for line in f:
                    ex = json.loads(line)
                    ground_truth_data[ex['example_id']] = ex

        # Eval loop.
        with torch.no_grad():
            for i, batch_map in enumerate(batches):
                batch_size, length = batch_map['sentences'].shape

                if length <= 2:
                    continue

                example_ids = batch_map['example_ids']
                if self.ground_truth is not None:
                    batch_ground_truth = [ground_truth_data[x] for x in example_ids]
                    batch_map['ground_truth'] = batch_ground_truth

                _ = trainer.step(batch_map, train=False, compute_loss=False, info={ 'inside_pool': self.inside_pool, 'outside': self.outside })

                for j, x in enumerate(parse_predictor.predict(batch_map)):

                    pred_lst.append(x)

                self.eval_loop_hook(trainer, diora, info, eval_cache, batch_map)

        self.post_eval_hook(trainer, diora, info, eval_cache)

        diora.safe_set_K(original_K)

        # set choose_tree
        if hasattr(diora, 'choose_tree'):
            diora.choose_tree = original_choose_tree

        return pred_lst

    def eval_loop_hook(self, trainer, diora, info, eval_cache, batch_map):
        pass

    def post_eval_hook(self, trainer, diora, info, eval_cache):
        pass

    def run(self, trainer, info):
        logger = self.logger
        outfile = info.get('outfile', None)
        pred_lst, multilayer = self.parse(trainer, info)

        corpus = collections.OrderedDict()

        # Read the ground truth.
        with open(self.ground_truth) as f:
            for line in f:
                ex = json.loads(line)
                corpus[ex['example_id']] = ex

        total_instance = 0
        num_ner_span = 0
        covered_ner_spans = 0
        num_addition_span = 0
        covered_addition_span = 0
        path = outfile + '.constraint' +'.coverage'
        for x in pred_lst:
            total_instance+=1
            example_id = x['example_id']
            pred_span = set(tree_to_spans(x['binary_tree']))
            if 'ner_span' in corpus[example_id]:
                ner_span = set([tuple(tmp) for tmp in corpus[example_id]['ner_span']])
                ner_overlap = len(ner_span.intersection(pred_span))
                num_ner_span+=len(ner_span)
                covered_ner_spans+=ner_overlap
            if 'additional_constraint' in corpus[example_id]:
                addition_span = set([tuple(tmp) for tmp in corpus[example_id]['additional_constraint']])
                addition_overlap = len(addition_span.intersection(pred_span))
                num_addition_span += len(addition_span)
                covered_addition_span+=addition_overlap
        with open(path,'w') as f:
            f.write('total instance '+str(total_instance) + '\n')
            f.write('num ner span '+str(num_ner_span)+ '\n')
            f.write('num covered ner spans '+ str(covered_ner_spans) + '\n')
            if num_ner_span >0:
                f.write('accuracy '+ str(float(covered_ner_spans)/num_ner_span)+ '\n')
                print('ner accuracy: '+str(float(covered_ner_spans)/num_ner_span))
            else:
                f.write('accuracy None''\n')
            f.write('num additional span '+str(num_addition_span)+ '\n')
            f.write('num covered additional spans '+ str(covered_addition_span) + '\n')
            if num_addition_span >0:
                f.write('accuracy '+ str(float(covered_addition_span)/num_addition_span)+ '\n')
                print('additional accuracy: '+str(float(covered_addition_span)/num_addition_span))
            else:
                f.write('accuracy None''\n')


        # Write more general format.
        path = outfile + '.pred'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                pred_binary_tree = x['binary_tree']
                example_id = x['example_id']
                gt = corpus[example_id]
                tokens = corpus[example_id]['tokens']
                f.write(to_raw_parse(pred_binary_tree, tokens) + '\n')

        path = outfile + '.pred.nopunct'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                pred_binary_tree = x['binary_tree']
                example_id = x['example_id']
                gt = corpus[example_id]
                part_of_speech = [x[1] for x in nltk.Tree.fromstring(gt['raw_parse']).pos()]
                tokens = corpus[example_id]['tokens']
                f.write(to_raw_parse_nopunct(pred_binary_tree, tokens, part_of_speech) + '\n')
        pred_path = path

        path = outfile + '.gold'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                example_id = x['example_id']
                gt = corpus[example_id]
                f.write(gt['raw_parse'] + '\n')

        path = outfile + '.gold.nopunct'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                example_id = x['example_id']
                gt = corpus[example_id]
                tokens = gt['tokens']
                part_of_speech = [x[1] for x in nltk.Tree.fromstring(gt['raw_parse']).pos()]
                gt_nltk_tree = nltk.Tree.fromstring(gt['raw_parse'])
                f.write(gt_to_raw_parse_nopunct(gt_nltk_tree, tokens, part_of_speech) + '\n')
        gold_path = path

        path = outfile + '.diora'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                example_id = x['example_id']
                gt = corpus[example_id]
                tokens = gt['tokens']
                o = collections.OrderedDict()
                o['example_id'] = example_id
                o['binary_tree'] = x['binary_tree']
                o['raw_parse'] = to_raw_parse(x['binary_tree'], tokens)
                o['tokens'] = tokens
                f.write(json.dumps(o) + '\n')

        evalb_path = './EVALB/evalb'
        if not os.path.exists(evalb_path):
            build_command = 'cd {} && make'.format(os.path.dirname(evalb_path))
            logger.info('Building EVALB. $ {}'.format(build_command))
            os.system(build_command)

        config_path = './EVALB/diora.prm'
        out_path = outfile + '.evalb'
        evalb_command = '{evalb} -p {evalb_config} {gold} {pred} > {out}'.format(
            evalb=evalb_path,
            evalb_config=config_path,
            gold=gold_path,
            pred=pred_path,
            out=out_path)

        logger.info('Running eval. $ {}'.format(evalb_command))
        subprocess.run(evalb_command, shell=True)

        # Parse EVALB Results
        with open(out_path) as f:
            evalb_results = collections.defaultdict(dict)
            section = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('--') and line.endswith('--'):
                    section = line[3:-3]
                    continue
                if section is None:
                    continue
                key, val = line.split('=')
                key = key.strip()
                val = float(val.strip())
                evalb_results[section][key] = val

        eval_result = dict()
        eval_result['name'] = self.name
        eval_result['meta'] = dict()
        eval_result['meta']['f1'] =        evalb_results['All']['Bracketing FMeasure']
        eval_result['meta']['recall'] =    evalb_results['All']['Bracketing Recall']
        eval_result['meta']['precision'] = evalb_results['All']['Bracketing Precision']
        eval_result['meta']['exact_match'] = evalb_results['All']['Complete match']

        return eval_result


constraint_class = NERComponent

# import collections
# import json
# import os

# import nltk
# from nltk.tree import Tree
# from nltk.treeprettyprinter import TreePrettyPrinter
# import numpy as np
# import torch
# from tqdm import tqdm

# from cky import ParsePredictor as CKY
# from experiment_logger import get_logger
# from evaluation_utils import BaseEvalFunc


# class ConstraintCKY(object):
#     def __init__(self, net, word2idx, initial_scalar=1):
#         super(ConstraintCKY, self).__init__()
#         self.net = net
#         self.idx2word = {v: k for k, v in word2idx.items()}
#         self.initial_scalar = initial_scalar

#     def predict(self, batch_map, return_components=False):
#         batch = batch_map['sentences']
#         example_ids = batch_map['example_ids']
#         batch_span = [gt['ner_span'] for gt in batch_map['ground_truth']]

#         batch_size = self.net.batch_size

#         trees, components = self.parse_batch(batch,batch_span)

#         out = []
#         for i in range(batch_size):
#             assert trees[i] is not None
#             out.append(dict(example_id=example_ids[i], binary_tree=trees[i]))

#         if return_components:
#             return (out, components)

#         return out

#     def parse_batch(self, batch, batch_span, cell_loss=False, return_components=False):
#         batch_size = self.net.batch_size
#         length = self.net.length
#         scalars = self.net.cache['inside_s_components'].copy()
#         device = self.net.device
#         dtype = torch.float32

#         # Assign missing scalars
#         for i in range(length):
#             scalars[0][i] = torch.full((batch_size, 1), self.initial_scalar, dtype=dtype, device=device)

#         leaves = [None for _ in range(batch_size)]
#         for i in range(batch_size):
#             batch_i = batch[i].tolist()
#             leaves[i] = [self.idx2word[idx] for idx in batch_i]

#         trees, components = self.batched_cky(scalars, leaves, batch_span)

#         return trees, components

#     def initial_chart(self,batch_span):
#         batch_size = self.net.batch_size
#         length = self.net.length
#         device = self.net.device
#         dtype = torch.float32
#         # spans: [[[start,length],[]],[[],[],...],...]
#         chart = [torch.full((length-i, batch_size), 1, dtype=dtype, device=device) for i in range(length)]
#         for idx, spans in enumerate(batch_span):
#             for span in spans:
#                 level = span[1]-1
#                 assert level>0
#                 pos = span[0]
#                 # cross = range(min(0,pos-level),max(length-level,pos+level))
#                 # for i in cross:
#                 # chart[level][min(0,pos-level):max(length-level,pos+level),idx] = float('-inf')
#                 chart[level][pos,idx] = 10000.0
#         return chart



#     def batched_cky(self, scalars, leaves, batch_span):
#         batch_size = self.net.batch_size
#         length = self.net.length
#         device = self.net.device
#         dtype = torch.float32

#         # Chart.
#         chart = self.initial_chart(batch_span)
#         components = {}

#         # Backpointers.
#         bp = {}
#         for ib in range(batch_size):
#             bp[ib] = [[None] * (length - i) for i in range(length)]
#             bp[ib][0] = [i for i in range(length)]

#         for level in range(1, length):
#             L = length - level
#             N = level

#             for pos in range(L):

#                 pairs, lps, rps, sps = [], [], [], []

#                 # Assumes that the bottom-left most leaf is in the first constituent.
#                 spbatch = scalars[level][pos]

#                 for idx in range(N):
#                     # (level, pos)
#                     l_level = idx
#                     l_pos = pos
#                     r_level = level-idx-1
#                     r_pos = pos+idx+1

#                     # assert l_level >= 0
#                     # assert l_pos >= 0
#                     # assert r_level >= 0
#                     # assert r_pos >= 0

#                     l = (l_level, l_pos)
#                     r = (r_level, r_pos)

#                     lp = chart[l_level][l_pos].view(-1, 1)
#                     rp = chart[r_level][r_pos].view(-1, 1)
#                     sp = spbatch[:, idx].view(-1, 1)

#                     lps.append(lp)
#                     rps.append(rp)
#                     sps.append(sp)

#                     pairs.append((l, r))

#                 lps, rps, sps = torch.cat(lps, 1), torch.cat(rps, 1), torch.cat(sps, 1)

#                 ps = lps + rps + sps
#                 components[(level, pos)] = ps
#                 argmax = ps.argmax(1).long()

#                 valmax = ps[range(batch_size), argmax]
#                 chart[level][pos, :] += valmax

#                 for i, ix in enumerate(argmax.tolist()):
#                     bp[i][level][pos] = pairs[ix]

#         trees = []
#         for i in range(batch_size):
#             tree = self.follow_backpointers(bp[i], leaves[i], bp[i][-1][0])
#             trees.append(tree)

#         return trees, components

#     def follow_backpointers(self, bp, words, pair):
#         if isinstance(pair, int):
#             return words[pair]

#         l, r = pair
#         lout = self.follow_backpointers(bp, words, bp[l[0]][l[1]])
#         rout = self.follow_backpointers(bp, words, bp[r[0]][r[1]])

#         return (lout, rout)




# def convert_to_nltk(tr, label='|'):
#     def helper(tr):
#         if not isinstance(tr, (list, tuple)):
#             return '({} {})'.format(label, tr)
#         nodes = []
#         for x in tr:
#             nodes.append(helper(x))
#         return '({} {})'.format(label, ' '.join(nodes))
#     return helper(tr)


# def example_f1(gt, pred):
#     correct = len(gt.intersection(pred))
#     if correct == 0:
#         return 0., 0., 0.
#     gt_total = len(gt)
#     pred_total = len(pred)
#     prec = float(correct) / pred_total
#     recall = float(correct) / gt_total
#     f1 = 2 * (prec * recall) / (prec + recall)
#     return f1, prec, recall

# def per_sentence_f1(gold_tree, pred_tree):
#     gold_tree = Tree.fromstring(gold_tree)
#     gt_spans = set(tree_to_spans(gold_tree))
#     pred_spans = set(tree_to_spans(pred_tree))
#     return example_f1(gt_spans,pred_spans)



# def tree_to_spans(tree):
#     spans = []

#     def helper(tr, pos):
#         if not isinstance(tr, (list, tuple)):
#             size = 1
#             return size
#         elif isinstance(tr, Tree) and len(tr) == 1:
#             size = 1
#             return size
#         size = 0
#         for x in tr:
#             xpos = pos + size
#             xsize = helper(x, xpos)
#             size += xsize
#         spans.append((pos, size))
#         return size

#     helper(tree, 0)

#     return spans


# def spans_to_tree(spans, tokens):
#     length = len(tokens)

#     # Add missing spans.
#     span_set = set(spans)
#     for pos in range(length):
#         if pos not in span_set:
#             spans.append((pos, 1))

#     spans = sorted(spans, key=lambda x: (x[1], x[0]))

#     pos_to_node = {}
#     root_node = None

#     for i, span in enumerate(spans):

#         pos, size = span

#         if i < length:
#             assert i == pos
#             node = (pos, size, tokens[i])
#             pos_to_node[pos] = node
#             continue

#         node = (pos, size, [])

#         for i_pos in range(pos, pos+size):
#             child = pos_to_node[i_pos]
#             c_pos, c_size = child[0], child[1]

#             if i_pos == c_pos:
#                 node[2].append(child)
#             pos_to_node[i_pos] = node

#     def helper(node):
#         pos, size, x = node
#         if isinstance(x, str):
#             return x
#         return tuple([helper(xx) for xx in x])

#     root_node = pos_to_node[0]
#     tree = helper(root_node)

#     return tree


# class TreesFromDiora(object):
#     def __init__(self, diora, word2idx, outside, oracle):
#         self.diora = diora
#         self.word2idx = word2idx
#         self.idx2word = {idx: w for w, idx in word2idx.items()}
#         self.outside = outside
#         self.oracle = oracle

#     def to_spans(self, lst):
#         return [(pos, level + 1) for level, pos in lst]

#     def predict(self, batch_map):
#         batch_size, length = batch_map['sentences'].shape
#         example_ids = batch_map['example_ids']
#         tscores = [0.0] * batch_size
#         K = self.diora.K

#         for i_b in range(batch_size):
#             tokens = batch_map['ground_truth'][i_b]['tokens']
#             root_level, root_pos = length - 1, 0
#             spans = self.to_spans(self.diora.cache['inside_tree'][(i_b, 0)][(root_level, root_pos)])
#             binary_tree = spans_to_tree(spans, tokens)
#             other_trees = []

#             yield dict(example_id=example_ids[i_b], binary_tree=binary_tree, binary_tree_score=tscores[i_b], other_trees=other_trees)



# class NERComponent(BaseEvalFunc):

#     def init_defaults(self):
#         self.agg_mode = 'sum'
#         self.cky_mode = 'sum'
#         self.ground_truth = None
#         self.inside_pool = 'sum'
#         self.oracle = {'use': False}
#         self.outside = True
#         self.seed = 121
#         self.semi_supervised = False
#         self.K = None
#         self.choose_tree = 'local'

#     def compare(self, prev_best, results):
#         out = []
#         key, val, is_best = 'placeholder', None, True
#         out.append((key, val, is_best))
#         return out

#     def parse(self, trainer, info):
#         logger = self.logger

#         multilayer = False
#         diora = trainer.get_single_net(trainer.net).diora
#         if hasattr(diora, 'layers'):
#             multilayer = True
#             pred_lst = []
#             for i, layer in enumerate(diora.layers):
#                 logger.info(f'Diora Layer {i}:')
#                 pred = self.single_layer_parser(trainer, layer, info)
#                 pred_lst.append(pred)
#         else:
#             pred_lst = self.single_layer_parser(trainer, diora, info)
#         return pred_lst, multilayer

#     def single_layer_parser(self, trainer, diora, info):
#         logger = self.logger
#         epoch = info.get('epoch', 0)

#         original_K = diora.K
#         if self.K is not None:
#             diora.safe_set_K(self.K)

#         # set choose_tree
#         if hasattr(diora, 'choose_tree'):
#             original_choose_tree = diora.choose_tree
#             diora.choose_tree = self.choose_tree

#         word2idx = self.dataset['word2idx']
#         if self.cky_mode == 'cky':
#             parse_predictor = CKY(net=diora, word2idx=word2idx)
#         elif self.cky_mode == 'constraint_cky':
#             parse_predictor = ConstraintCKY(net=diora, word2idx=word2idx)
#         elif self.cky_mode == 'diora':
#             parse_predictor = TreesFromDiora(diora=diora, word2idx=word2idx, outside=self.outside, oracle=self.oracle)

#         batches = self.batch_iterator.get_iterator(random_seed=self.seed, epoch=epoch)

#         logger.info('Parsing.')

#         pred_lst = []
#         counter = 0
#         eval_cache = {}

#         if self.ground_truth is not None:
#             self.ground_truth = os.path.expanduser(self.ground_truth)
#             ground_truth_data = {}
#             with open(self.ground_truth) as f:
#                 for line in f:
#                     ex = json.loads(line)
#                     ground_truth_data[ex['example_id']] = ex

#         # Eval loop.
#         with torch.no_grad():
#             for i, batch_map in enumerate(batches):
#                 logger.info('Current batch {}.'.format(i))
#                 batch_size, length = batch_map['sentences'].shape

#                 if length <= 2:
#                     continue

#                 example_ids = batch_map['example_ids']
#                 if self.ground_truth is not None:
#                     batch_ground_truth = [ground_truth_data[x] for x in example_ids]
#                     batch_map['ground_truth'] = batch_ground_truth

#                 _ = trainer.step(batch_map, train=False, compute_loss=False, info={ 'inside_pool': self.inside_pool, 'outside': self.outside })

#                 for j, x in enumerate(parse_predictor.predict(batch_map)):

#                     pred_lst.append(x)

#                 self.eval_loop_hook(trainer, diora, info, eval_cache, batch_map)

#         self.post_eval_hook(trainer, diora, info, eval_cache)

#         diora.safe_set_K(original_K)

#         # set choose_tree
#         if hasattr(diora, 'choose_tree'):
#             diora.choose_tree = original_choose_tree

#         return pred_lst

#     def eval_loop_hook(self, trainer, diora, info, eval_cache, batch_map):
#         pass

#     def post_eval_hook(self, trainer, diora, info, eval_cache):
#         pass

#     def run(self, trainer, info):
#         logger = self.logger
#         outfile = info.get('outfile', None)
#         pred_lst, multilayer = self.parse(trainer, info)
#         if self.write:
#             corpus = collections.OrderedDict()

#             # Read the ground truth.
#             with open(self.ground_truth) as f:
#                 for line in f:
#                     ex = json.loads(line)
#                     corpus[ex['example_id']] = ex

#             def to_raw_parse(tr):
#                 def helper(tr):
#                     if isinstance(tr, (str, int)):
#                         return '(DT {})'.format(tr)
#                     nodes = []
#                     for x in tr:
#                         nodes.append(helper(x))
#                     return '(S {})'.format(' '.join(nodes))
#                 return '(ROOT {})'.format(helper(tr))

#             #evaluate the span coverage
#             total_instance = len(pred_lst)
#             #print(total_instance)
#             num_ner_span = 0
#             covered_spans = 0
#             F1 = []
#             Prec = []
#             Recall = []
#             path = outfile + '.' +self.cky_mode
#             with open(path, 'w') as f:
#                 for x in pred_lst:
#                     example_id = x['example_id']
#                     ner_span = set([tuple(tmp) for tmp in corpus[example_id]['ner_span']])
#                     pred_span = set(tree_to_spans(x['binary_tree']))
#                     overlap = len(ner_span.intersection(pred_span))
#                     o = collections.OrderedDict()
#                     o['example_id'] = example_id
#                     o['tokens'] = corpus[example_id]['tokens']
#                     o['binary_tree'] = x['binary_tree']
#                     o['raw_parse'] = corpus[example_id]['raw_parse']
#                     o['entity'] = [ corpus[example_id]['tokens'][tmp[0]:tmp[0]+tmp[1]] for tmp in corpus[example_id]['ner_span']]
#                     o['ner_span'] = tuple(ner_span)
#                     o['pred_span'] = tuple(pred_span)
#                     o['overlap'] = (overlap, float(overlap)/len(ner_span) if len(ner_span)>0 else 0.0)
#                     f.write(json.dumps(o) + '\n')
#                     num_ner_span+=len(ner_span)
#                     covered_spans+=overlap
#                     f1 , prec, recall = per_sentence_f1(o['raw_parse'],o['binary_tree'])
#                     F1.append(f1)
#                     Prec.append(prec)
#                     Recall.append(recall)
#             path = outfile + '.'+self.cky_mode +'.accu'
#             with open(path, 'w') as f:
#                 f.write('total instance '+str(total_instance) + '\n')
#                 f.write('num ner span '+str(num_ner_span)+ '\n')
#                 f.write('num covered spans '+ str(covered_spans) + '\n')
#                 f.write('accuracy '+ str(float(covered_spans)/num_ner_span)+ '\n')
#                 f.write('parsing F1 '+ str(np.mean(np.asarray(F1)))+ '\n')
#                 f.write('parsing Precision '+ str(np.mean(np.asarray(Prec)))+ '\n')
#                 f.write('parsing Recall '+ str(np.mean(np.asarray(Recall)))+ '\n')
#             #print('total_instance ',total_instance)
#             #print('num_ner_span ',num_ner_span)
#             #print('covered_spans ',covered_spans)
#                 #if len(ner_span) <1:
#                 #    continue
#                 #print(x)
#                 #print('ner span',ner_span)
#                 #print('pred span',pred_span)
#                 #print('coverd: ',len(set(ner_span).intersection(set(pred_span))))
#             """
#             # Write more general format.
#             path = outfile + '.pred'
#             logger.info('writing parse tree output -> {}'.format(path))
#             with open(path, 'w') as f:
#                 for x in pred_lst:
#                     pred_binary_tree = x['binary_tree']
#                     f.write(to_raw_parse(pred_binary_tree) + '\n')

#             path = outfile + '.gold'
#             logger.info('writing parse tree output -> {}'.format(path))
#             with open(path, 'w') as f:
#                 for x in pred_lst:
#                     example_id = x['example_id']
#                     gt = corpus[example_id]
#                     gt_binary_tree = gt['binary_tree']
#                     f.write(to_raw_parse(gt_binary_tree) + '\n')

#             path = outfile + '.diora'
#             logger.info('writing parse tree output -> {}'.format(path))
#             with open(path, 'w') as f:
#                 for x in pred_lst:
#                     example_id = x['example_id']
#                     gt = corpus[example_id]
#                     o = collections.OrderedDict()
#                     o['example_id'] = example_id
#                     o['binary_tree'] = x['binary_tree']
#                     o['raw_parse'] = to_raw_parse(x['binary_tree'])
#                     o['tokens'] = gt['tokens']
#                     f.write(json.dumps(o) + '\n')
#         """
#         eval_result = dict()
#         eval_result['name'] = self.name
#         eval_result['meta'] = dict()

#         return eval_result
