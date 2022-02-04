from eval_parsing import *


def run_evalb(gold_path, pred_path, out_path):
    evalb_path = './EVALB/evalb'
    if not os.path.exists(evalb_path):
        build_command = 'cd {} && make'.format(os.path.dirname(evalb_path))
        print('Building EVALB. $ {}'.format(build_command))
        os.system(build_command)

    config_path = './EVALB/diora.prm'
    evalb_command = '{evalb} -p {evalb_config} {gold} {pred} > {out}'.format(
        evalb=evalb_path,
        evalb_config=config_path,
        gold=gold_path,
        pred=pred_path,
        out=out_path)

    print('Running eval. $ {}'.format(evalb_command))
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


def get_spans(tr):
    spans = []
    def helper(tr, pos):
        if len(tr) == 1 and isinstance(tr[0], str):
            return 1

        size = 0
        for x in tr:
            xsize = helper(x, pos + size)
            size += xsize

        label = tr.label()
        spans.append((pos, size, label))

        return size
    helper(tr, 0)
    return spans


def check_labels(nltk_tree):
    labels = "NP    VP  PP  S   SBAR    ADJP    ADVP    QP  NML".strip().split()

    spans = get_spans(nltk_tree)

    length = len(nltk_tree.leaves())

    for pos, size, label in spans:
        if size == length:
            continue
        if label in labels:
            return True
    return False


def readfile(path):
    data = {}
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            if len(ex['tokens']) <= 2:
                continue
            ex['nltk_tree'] = nltk.Tree.fromstring(ex['raw_parse'])
            if not check_labels(ex['nltk_tree']):
                continue
            example_id = ex['example_id']
            assert example_id not in data
            data[example_id] = ex
    return data


def readfile_pred(path):
    data = {}
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            ex['nltk_tree'] = nltk.Tree.fromstring(ex['raw_parse'])
            example_id = ex['example_id']
            assert example_id not in data
            data[example_id] = ex
    return data


def main():
    print('read gold')
    gold_data = readfile(args.gold)
    print('read pred')
    pred_data = readfile_pred(args.pred)

    keys = []
    for k in gold_data.keys():
        assert k in pred_data
        keys.append(k)

    print('total', len(keys))

    path = 'run_evalb.gold.nopunct'
    with open(path, 'w') as f:
        for k in keys:
            ex = gold_data[k]
            tokens = ex['tokens']
            nltk_tree = ex['nltk_tree']
            part_of_speech = [x[1] for x in nltk_tree.pos()]
            f.write(gt_to_raw_parse_nopunct(nltk_tree, tokens, part_of_speech) + '\n')
    gold_path = path

    path = 'run_evalb.pred.nopunct'
    with open(path, 'w') as f:
        for k in keys:
            pred = pred_data[k]
            gold = gold_data[k]
            tokens = gold['tokens']
            part_of_speech = [x[1] for x in gold['nltk_tree'].pos()]
            f.write(to_raw_parse_nopunct(pred['nltk_tree'], tokens, part_of_speech) + '\n')
    pred_path = path


    out_path = 'run_evalb.evalb.upperbound'
    evalb_results = run_evalb(gold_path, pred_path, out_path)

    print('F1={:.3f} R={:.3f} P={:.3f} EM={:.3f}'.format(
            evalb_results['All']['Bracketing FMeasure'],
            evalb_results['All']['Bracketing Recall'],
            evalb_results['All']['Bracketing Precision'],
            evalb_results['All']['Complete match']
            ))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', type=str, default='ontonotes_ground_truth.va_d.jsonl')
    parser.add_argument('--pred', type=str, default='ontonotes_ground_truth.va_d.jsonl-diora_parsed')
    args = parser.parse_args()
    main()