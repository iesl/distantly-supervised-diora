import collections
import json
import os
import subprocess
import sys

import nltk
from nltk.treeprettyprinter import TreePrettyPrinter
import numpy as np
import torch
from tqdm import tqdm

from cky import ParsePredictor as CKY
from ccky_basic import ConstrainedCKY as CCKY_Basic
from constrained_cky import ConstrainedCKY as CCKY_MinDiff
from eval_constraint import ConstraintCKY as CCKY
from experiment_logger import get_logger
from evaluation_utils import BaseEvalFunc

word_tags = set(['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
               'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS',
               'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
               'WDT', 'WP', 'WP$', 'WRB'])


def to_raw_parse(tr, tokens, part_of_speech):
    assert len(tokens) == len(part_of_speech)
    def helper(tr, pos=0):
        if isinstance(tr, (str, int)):
            size = 1
            return '({} {})'.format(part_of_speech[pos], tokens[pos]), size
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
    new_pos = [x for x, m in zip(part_of_speech, mask) if m]
    return to_raw_parse(new_tr, new_tokens, new_pos)


def make_rb_tree(tokens):
    def helper(tokens):
        if len(tokens) == 1:
            return tokens[0]
        return (tokens[0], helper(tokens[1:]))
    return helper(tokens)


def to_raw_parse_nopunct_rb(tr, tokens, part_of_speech):
    mask = [x in word_tags for x in part_of_speech]
    new_tokens = [x for x, m in zip(tokens, mask) if m]
    new_tr, kept, removed = remove_using_flat_mask_nary_tree(tr, mask)
    new_tr = make_rb_tree(new_tokens)
    new_pos = [x for x, m in zip(part_of_speech, mask) if m]
    return to_raw_parse(new_tr, new_tokens, new_pos)

def to_raw_parse_rb(tr, tokens, part_of_speech):
    mask = [True for x in part_of_speech]
    new_tokens = [x for x, m in zip(tokens, mask) if m]
    new_tr, kept, removed = remove_using_flat_mask_nary_tree(tr, mask)
    new_tr = make_rb_tree(new_tokens)
    new_pos = [x for x, m in zip(part_of_speech, mask) if m]
    return to_raw_parse(new_tr, new_tokens, new_pos)


def gt_to_raw_parse_nopunct(tr, tokens, part_of_speech):
    mask = [x in word_tags for x in part_of_speech]
    new_tokens = [x for x, m in zip(tokens, mask) if m]
    new_tr, kept, removed = gt_remove_using_flat_mask_nary_tree(tr, mask)
    return new_tr.pformat(margin=10000)


def validate_binary(tr):
    def helper(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            return
        for x in tr:
            helper(x)
        assert len(tr) == 2
    helper(tr)


def replace_labels(tr):
    def helper(tr):
        if len(tr) == 1 and isinstance(tr[0], str):
            return '({} {})'.format(tr.label(), tr[0])
        nodes = [helper(x) for x in tr]
        return '({} {})'.format('XX', ' '.join(nodes))
    return helper(tr)

def gt_to_raw_parse_nopunct_binary(tr, tokens, part_of_speech):
    mask = [x in word_tags for x in part_of_speech]
    new_tokens = [x for x, m in zip(tokens, mask) if m]
    new_tr, kept, removed = gt_remove_using_flat_mask_nary_tree(tr, mask)
    new_tr.chomsky_normal_form()
    validate_binary(new_tr)
    new_tr_s = replace_labels(new_tr)
    return new_tr_s

def gt_to_raw_parse_binary(tr, tokens, part_of_speech):
    mask = [True for x in part_of_speech]
    new_tokens = [x for x, m in zip(tokens, mask) if m]
    new_tr, kept, removed = gt_remove_using_flat_mask_nary_tree(tr, mask)
    new_tr.chomsky_normal_form()
    validate_binary(new_tr)
    new_tr_s = replace_labels(new_tr)
    return new_tr_s


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


def get_spans_binary_tree(tr):
    spans = []
    def helper(tr, pos):
        if isinstance(tr, str):
            return 1

        assert len(tr) == 2

        size = 0
        for x in tr:
            xsize = helper(x, pos + size)
            size += xsize
        spans.append((pos, size))
        return size
    helper(tr, 0)
    return spans


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
        size = 0
        for x in tr:
            xpos = pos + size
            xsize = helper(x, xpos)
            size += xsize
        spans.append((pos, size))
        return size

    helper(tree, 0)

    return spans


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


class ParsingComponent(BaseEvalFunc):

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
        self.scalars_key = 'inside_s_components'
        self.verbose = False

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
        diora = trainer.get_single_net(trainer.net).diora
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
            parse_predictor = CKY(net=diora, word2idx=word2idx, scalars_key=self.scalars_key)
        elif self.cky_mode == 'ccky' or self.cky_mode == 'constrained_cky':
            parse_predictor = CCKY(net=diora, word2idx=word2idx, scalars_key=self.scalars_key)
        elif self.cky_mode == 'ccky_basic':
            parse_predictor = CCKY_Basic(net=diora, word2idx=word2idx, scalars_key=self.scalars_key)
        elif self.cky_mode == 'ccky_mindiff' or self.cky_mode == 'constrained_cky':
            parse_predictor = CCKY_MinDiff(net=diora, word2idx=word2idx, scalars_key=self.scalars_key, pred_weight=1000, constraint_weight=10000)
        elif self.cky_mode == 'diora':
            parse_predictor = TreesFromDiora(diora=diora, word2idx=word2idx, outside=self.outside, oracle=self.oracle)

        batches = self.batch_iterator.get_iterator(random_seed=self.seed, epoch=epoch)

        logger.info('Parsing. cky_mode={} scalars_key={} '.format(self.cky_mode, self.scalars_key))

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
            for i, batch_map in tqdm(enumerate(batches), disable=not self.verbose):
                batch_size, length = batch_map['sentences'].shape

                if length <= 2:
                    continue

                example_ids = batch_map['example_ids']
                if self.ground_truth is not None:
                    batch_ground_truth = [ground_truth_data[x] for x in example_ids]
                    batch_map['ground_truth'] = batch_ground_truth

                _ = trainer.step(batch_map, train=False, compute_loss=False, info={ 'inside_pool': self.inside_pool, 'outside': self.outside })

                for j, x in enumerate(parse_predictor.predict(batch_map)):

                    x['ner_label'] = []
                    if 'ner_labels' in batch_map:
                        gold_spans = set([(pos, size) for pos, size, label in batch_map['ner_labels'][j] if size > 1])
                        binary_tree = x['binary_tree']
                        pred_spans = set([(pos, size) for pos, size in get_spans_binary_tree(binary_tree) if size > 1])

                        found_ner = len(set.intersection(gold_spans, pred_spans))
                        total_ner = len(gold_spans)

                        x['found_ner'] = found_ner
                        x['total_ner'] = total_ner
                        x['ner_label'] = batch_map['ner_labels'][j]
                        x['example_ids'] = batch_map['example_ids'][j]
                        x['found_ner_list'] = list(set.intersection(gold_spans, pred_spans))

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
        pred_lst = self.parse(trainer, info)
        skip_eval = False

        corpus = collections.OrderedDict()

        # Read the ground truth.
        with open(self.ground_truth) as f:
            for line in f:
                ex = json.loads(line)
                corpus[ex['example_id']] = ex

        # Add part-of-speech and tree.
        for x in pred_lst:
            example_id = x['example_id']
            gt = corpus[example_id]
            tokens = gt['tokens']
            try:
                gt_nltk_tree = nltk.Tree.fromstring(gt['raw_parse'])
                part_of_speech = [x[1] for x in gt_nltk_tree.pos()]
                assert len(tokens) == len(part_of_speech)
            except:
                skip_eval = True # If this exception hits, then we can not run eval.
                gt_nltk_tree = None
                part_of_speech = ['DT'] * len(tokens)

            gt['part_of_speech'] = part_of_speech
            gt['nltk_tree'] = gt_nltk_tree

        # Count found constraints
        if 'total_ner' in pred_lst[0]:
            path = outfile + '.ner_result.jsonl'
            with open(path,'w') as f:
                total_ner, found_ner = 0, 0
                for x in pred_lst:
                    found_ner += x['found_ner']
                    total_ner += x['total_ner']
                    f.write(json.dumps([x['example_ids'],x['found_ner_list']])+'\n')
                logger.info('SPAN-RECALL: {}/{} {:.3f}'.format(
                    found_ner, total_ner, found_ner / total_ner))

        path = outfile + '.pred.diora'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                pred_binary_tree = x['binary_tree']
                example_id = x['example_id']
                gt = corpus[example_id]
                part_of_speech = gt['part_of_speech']
                tokens = gt['tokens']
                o = collections.OrderedDict()
                o['example_id'] = example_id
                o['binary_tree'] = pred_binary_tree
                o['raw_parse'] = to_raw_parse(pred_binary_tree, tokens, part_of_speech)
                o['raw_parse_nopunct'] = to_raw_parse_nopunct(pred_binary_tree, tokens, part_of_speech)
                o['tokens'] = tokens
                o['ner_label'] = x['ner_label']
                f.write(json.dumps(o) + '\n')

        if skip_eval:
            print('Done! Ending early because not able to run eval.')
            sys.exit()

        path = outfile + '.gold.diora'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                example_id = x['example_id']
                gt = corpus[example_id]
                gt_nltk_tree = gt['nltk_tree']
                part_of_speech = gt['part_of_speech']
                tokens = gt['tokens']
                o = collections.OrderedDict()
                o['example_id'] = example_id
                o['raw_parse'] = gt['raw_parse']
                o['raw_parse_nopunct'] = gt_to_raw_parse_nopunct(gt_nltk_tree, tokens, part_of_speech)
                o['tokens'] = tokens
                o['ner_label'] = x['ner_label']
                f.write(json.dumps(o) + '\n')

        # UPPER BOUND
        path = outfile + '.gold.nopunct.binary'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                example_id = x['example_id']
                gt = corpus[example_id]
                tokens = gt['tokens']
                part_of_speech = gt['part_of_speech']
                gt_nltk_tree = gt['nltk_tree']
                f.write(gt_to_raw_parse_nopunct_binary(gt_nltk_tree, tokens, part_of_speech) + '\n')
        upperbound_path = path

        #save upperbound
        path = outfile + '.upperbound.diora'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                example_id = x['example_id']
                gt = corpus[example_id]
                gt_nltk_tree = gt['nltk_tree']
                part_of_speech = gt['part_of_speech']
                tokens = gt['tokens']
                o = collections.OrderedDict()
                o['example_id'] = example_id
                o['raw_parse'] = gt_to_raw_parse_binary(gt_nltk_tree, tokens, part_of_speech)
                o['raw_parse_nopunct'] = gt_to_raw_parse_nopunct_binary(gt_nltk_tree, tokens, part_of_speech)
                o['tokens'] = tokens
                o['ner_label'] = x['ner_label']
                f.write(json.dumps(o) + '\n')


        # right BRANCHING
        path = outfile + '.rightbranching.nopunct'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                pred_binary_tree = x['binary_tree']
                example_id = x['example_id']
                gt = corpus[example_id]
                part_of_speech = gt['part_of_speech']
                tokens = corpus[example_id]['tokens']
                f.write(to_raw_parse_nopunct_rb(pred_binary_tree, tokens, part_of_speech) + '\n')
        rightbranching_path = path

        #save right
        path = outfile + '.rightbranching.diora'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                pred_binary_tree = x['binary_tree']
                example_id = x['example_id']
                gt = corpus[example_id]
                part_of_speech = gt['part_of_speech']
                tokens = gt['tokens']
                o = collections.OrderedDict()
                o['example_id'] = example_id
                o['binary_tree'] = pred_binary_tree
                o['raw_parse'] = to_raw_parse_rb(pred_binary_tree, tokens, part_of_speech)
                o['raw_parse_nopunct'] = to_raw_parse_nopunct_rb(pred_binary_tree, tokens, part_of_speech)
                o['tokens'] = tokens
                o['ner_label'] = x['ner_label']
                f.write(json.dumps(o) + '\n')

        # EVALB

        # Write more general format.
        path = outfile + '.pred'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                pred_binary_tree = x['binary_tree']
                example_id = x['example_id']
                gt = corpus[example_id]
                part_of_speech = gt['part_of_speech']
                tokens = corpus[example_id]['tokens']
                f.write(to_raw_parse(pred_binary_tree, tokens, part_of_speech) + '\n')

        path = outfile + '.pred.nopunct'
        logger.info('writing parse tree output -> {}'.format(path))
        with open(path, 'w') as f:
            for x in pred_lst:
                pred_binary_tree = x['binary_tree']
                example_id = x['example_id']
                gt = corpus[example_id]
                part_of_speech = gt['part_of_speech']
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
                part_of_speech = gt['part_of_speech']
                tokens = gt['tokens']
                gt_nltk_tree = gt['nltk_tree']
                f.write(gt_to_raw_parse_nopunct(gt_nltk_tree, tokens, part_of_speech) + '\n')
        gold_path = path

        def run_evalb(gold_path, pred_path, out_path):
            evalb_path = './EVALB/evalb'
            if not os.path.exists(evalb_path):
                build_command = 'cd {} && make'.format(os.path.dirname(evalb_path))
                logger.info('Building EVALB. $ {}'.format(build_command))
                os.system(build_command)

            config_path = './EVALB/diora.prm'
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

            return evalb_results

        # UPPER BOUND
        out_path = outfile + '.evalb.upperbound'
        evalb_results = run_evalb(gold_path, upperbound_path, out_path)
        logger.info('UPPERBOUND: F1={:.3f} R={:.3f} P={:.3f} EM={:.3f}'.format(
            evalb_results['All']['Bracketing FMeasure'],
            evalb_results['All']['Bracketing Recall'],
            evalb_results['All']['Bracketing Precision'],
            evalb_results['All']['Complete match']
            ))

        # RIGHT BRANCHING
        out_path = outfile + '.evalb.rightbranching'
        evalb_results = run_evalb(gold_path, rightbranching_path, out_path)
        logger.info('RB: F1={:.3f} R={:.3f} P={:.3f} EM={:.3f}'.format(
            evalb_results['All']['Bracketing FMeasure'],
            evalb_results['All']['Bracketing Recall'],
            evalb_results['All']['Bracketing Precision'],
            evalb_results['All']['Complete match']
            ))

        # ACTUAL
        out_path = outfile + '.evalb'
        evalb_results = run_evalb(gold_path, pred_path, out_path)

        eval_result = dict()
        eval_result['name'] = self.name
        eval_result['meta'] = dict()
        eval_result['meta']['f1'] =        evalb_results['All']['Bracketing FMeasure']
        eval_result['meta']['recall'] =    evalb_results['All']['Bracketing Recall']
        eval_result['meta']['precision'] = evalb_results['All']['Bracketing Precision']
        eval_result['meta']['exact_match'] = evalb_results['All']['Complete match']

        return eval_result


parsing_class = ParsingComponent
