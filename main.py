import argparse
import copy
import datetime
import math
import json
import os
import random
import sys
import uuid

import numpy as np

import torch
import torch.nn as nn

from help_get_eval_components import get_eval_components, ModelEvaluation
from help_get_net_components import get_net_components

from dataset_utils import get_train_dataset
from dataset_utils import get_train_iterator
from dataset_utils import get_validation_dataset
from dataset_utils import get_validation_iterator
from dataset_utils import get_train_and_validation
from trainer import build_net as trainer_build_net

from flags import stringify_flags, init_with_flags_file, save_flags
from experiment_logger import ExperimentLogger, configure_experiment, get_logger


def count_params(net):
    return sum([x.numel() for x in net.parameters() if x.requires_grad])


def save_experiment(experiment_file, metadata):
    with open(experiment_file, 'w') as f:
        f.write(json.dumps(metadata, indent=4, sort_keys=True))


def build_net(options, embeddings, batch_iterator=None, word2idx=None):

    cuda = options.cuda

    context = {}
    context['cuda'] = cuda
    context['embeddings'] = embeddings
    context['batch_iterator'] = batch_iterator
    context['word2idx'] = word2idx

    net_components = get_net_components(options, context)

    trainer = trainer_build_net(options, context=context, net_components=net_components)

    logger = get_logger()
    logger.info('# of params = {}'.format(count_params(trainer.net)))

    return trainer


def generate_seeds(n, seed=11):
    random.seed(seed)
    seeds = [random.randint(0, 2**16) for _ in range(n)]
    return seeds


def run_evaluation(options, trainer, model_evaluation, info, metadata):
    logger = get_logger()

    lst = []

    for eval_result_dict in model_evaluation.run(trainer, info, metadata):
        eval_result = eval_result_dict['result']
        eval_name = eval_result['name']
        for k, v in eval_result['meta'].items():
            logger.info('eval[{}] {}={}'.format(eval_name, k, v))
        lst.append(eval_result_dict)

    return lst


class MyTrainIterator(object):
    def __init__(self, options, batch_iterator, trainer):
        self.options = options
        self.batch_iterator = batch_iterator
        self.trainer = trainer

    def get_iterator(self):
        options = self.options
        logger = get_logger()
        logger.info('Generating seeds for {} epochs using initial seed {}.'.format(options.max_epoch, options.seed))
        seeds = generate_seeds(options.max_epoch, options.seed)

        for epoch, seed in zip(range(options.max_epoch), seeds):
            assert epoch == self.trainer.optimizer_epoch

            logger.info('epoch={} seed={}'.format(epoch, seed))
            it = self.batch_iterator.get_iterator(random_seed=seed, epoch=epoch)

            def iter_func_filter_length(it, min_length=3):
                for batch_map in it:
                    batch_size, length = batch_map['sentences'].shape
                    if length < min_length:
                        continue
                    yield batch_map
            it = iter_func_filter_length(it)

            def iter_func_add_end_of_epoch(it):
                batch_map_ = None
                end_of_epoch = False
                for batch_map in it:
                    # yield the previous batch
                    if batch_map_ is not None:
                        yield batch_map_, end_of_epoch
                    # the last batch is not yielded yet
                    batch_map_ = batch_map
                # yield the last batch here
                end_of_epoch = True
                yield batch_map_, end_of_epoch
            it = iter_func_add_end_of_epoch(it)

            for batch_idx, (batch_map, end_of_epoch) in enumerate(it):
                step = self.trainer.optimizer_step

                first_in_subbatch = (batch_idx % options.accum_steps) == 0
                last_in_subbatch = (batch_idx % options.accum_steps) == (options.accum_steps - 1)
                last_in_subbatch = last_in_subbatch or end_of_epoch

                should = {}
                should['zero_grad'] = first_in_subbatch is True
                should['update'] = (last_in_subbatch is True) or end_of_epoch
                should['end_of_epoch'] = end_of_epoch

                should['log'] = step % options.log_every_batch == 0 and options.log_every_batch > 0 and last_in_subbatch
                should['periodic'] = step % options.save_latest == 0 and step >= options.save_after and last_in_subbatch
                should['distinct'] = step % options.save_distinct == 0 and step >= options.save_after and last_in_subbatch

                should_eval_for_batch = (
                    options.eval_every_batch > 0 and
                    step % options.eval_every_batch == 0 and
                    step >= options.eval_after and last_in_subbatch
                    )
                should_eval_for_epoch = (
                    end_of_epoch and
                    options.eval_every_epoch > 0 and
                    epoch % options.eval_every_epoch == 0 and
                    step >= options.eval_after
                    )
                should['eval'] = should_eval_for_batch or should_eval_for_epoch

                yield batch_map, should

                if options.max_step >= 0 and step >= options.max_step:
                    logger.info('Max-Step={} Quitting.'.format(options.max_step))
                    sys.exit()


def run_train(options, train_iterator, trainer, model_evaluation):
    logger = get_logger()
    experiment_logger = ExperimentLogger()

    logger.info('Running train.')

    best_step = 0
    best_metric = math.inf
    best_dict = {}

    my_iterator = MyTrainIterator(options, train_iterator, trainer)

    for batch_map, should in my_iterator.get_iterator():
        step = trainer.optimizer_step
        epoch = trainer.optimizer_epoch
        epoch_step = trainer.optimizer_epoch_step

        # Forward pass.
        trainer_output = trainer.step(batch_map)

        # Backward pass.
        result, model_output = trainer_output['result'], trainer_output['model_output']
        total_loss = model_output['total_loss'].mean(dim=0).sum()
        total_loss = total_loss / options.accum_steps

        trainer.backward_and_maybe_update(total_loss, update=should['update'], zero_grad=should['zero_grad'])

        # Record result.
        experiment_logger.record(result)
        del result

        if should['log']:
            metrics = experiment_logger.log_batch(epoch, step, epoch_step)

        # -- Periodic Checkpoints -- #

        if should['periodic']:
            logger.info('Saving model (periodic).')
            trainer.save_model(os.path.join(options.experiment_path, 'model_periodic.pt'))
            save_experiment(os.path.join(options.experiment_path, 'experiment_periodic.json'),
                            dict(step=step, epoch=epoch, best_step=best_step, best_metric=best_metric))

        if should['distinct']:
            logger.info('Saving model (distinct).')
            trainer.save_model(os.path.join(options.experiment_path, 'model.step_{}.pt'.format(step)))
            save_experiment(os.path.join(options.experiment_path, 'experiment.step_{}.json'.format(step)),
                            dict(step=step, epoch=epoch, best_step=best_step, best_metric=best_metric))

        # -- Validation -- #

        if should['eval']:
            logger.info('Evaluation.')

            info = dict()
            info['experiment_path'] = options.experiment_path
            # info['step'] = step
            # info['epoch'] = epoch

            for eval_result_dict in run_evaluation(options, trainer, model_evaluation, info, metadata=dict(step=step)):
                result = eval_result_dict['result']
                func = eval_result_dict['component']
                name = func.name

                for key, val, is_best in func.compare(best_dict, result):
                    best_dict_key = 'best__{}__{}'.format(name, key)

                    # Early stopping.
                    if is_best:
                        if best_dict_key in best_dict:
                            prev_val = best_dict[best_dict_key]['value']
                        else:
                            prev_val = None
                        # Update running result.
                        best_dict[best_dict_key] = {}
                        best_dict[best_dict_key]['eval'] = name
                        best_dict[best_dict_key]['metric'] = key
                        best_dict[best_dict_key]['value'] = val
                        best_dict[best_dict_key]['step'] = step
                        best_dict[best_dict_key]['epoch'] = epoch

                        logger.info('Recording, best eval, key = {}, val = {} -> {}, json = {}'.format(
                            best_dict_key, prev_val, val, json.dumps(best_dict[best_dict_key])))

                        if step >= options.save_after:
                            logger.info('Saving model, best eval, key = {}, json = {}'.format(
                                best_dict_key, json.dumps(best_dict[best_dict_key])))
                            logger.info('checkpoint_dir = {}'.format(options.experiment_path))

                            # Save result and model.
                            trainer.save_model(
                                os.path.join(options.experiment_path, 'model.{}.pt'.format(best_dict_key)))
                            save_experiment(os.path.join(options.experiment_path, 'experiment.{}.json'.format(best_dict_key)),
                                best_dict[best_dict_key])

        # END OF EPOCH

        if should['end_of_epoch']:
            experiment_logger.log_epoch(epoch, step)
            trainer.end_of_epoch(best_dict)


def run(options):
    logger = get_logger()
    experiment_logger = ExperimentLogger()

    train_dataset, validation_dataset = get_train_and_validation(options)
    train_iterator = get_train_iterator(options, train_dataset)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    embeddings = train_dataset['embeddings']
    word2idx = train_dataset['word2idx']

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings, train_iterator, word2idx=word2idx)
    logger.info('Model:')
    for name, p in trainer.net.named_parameters():
        logger.info('{} {} {}'.format(name, p.shape, p.requires_grad))

    # Evaluation.
    context = {}
    context['dataset'] = validation_dataset
    context['batch_iterator'] = validation_iterator
    model_evaluation = ModelEvaluation(get_eval_components(options, context, config_lst=options.eval_config))


    if options.eval_only_mode:
        info = dict()
        info['experiment_path'] = options.experiment_path
        info['step'] = 0
        for eval_result_dict in run_evaluation(options, trainer, model_evaluation, info, metadata=dict(step=0)):
            result = eval_result_dict['result']
            func = eval_result_dict['component']
            name = func.name

            for key, val, _ in func.compare({}, result):
                best_dict_key = 'best__{}__{}'.format(name, key)

                save_dict = {}
                save_dict['eval'] = name
                save_dict['metric'] = key
                save_dict['value'] = val
                save_dict['step'] = -1
                save_dict['epoch'] = -1

                save_path = os.path.join(options.experiment_path, 'eval.{}.json'.format(best_dict_key))
                save_experiment(save_path, save_dict)
                logger.info('saved eval output to {}'.format(save_path))

        sys.exit()

    if options.save_init:
        logger.info('Saving model (init).')
        trainer.save_model(os.path.join(options.experiment_path, 'model_init.pt'))

    run_train(options, train_iterator, trainer, model_evaluation)


def argument_parser():
    parser = argparse.ArgumentParser()

    # Debug.
    parser.add_argument('--seed', default=None, type=int, help='Random seed.')
    parser.add_argument('--torch_version', default=None, type=str, help='[debug] Torch version.')
    parser.add_argument('--git_sha', default=None, type=str, help='[debug] Git SHA.')
    parser.add_argument('--git_branch_name', default=None, type=str, help='[debug] Git branch name.')
    parser.add_argument('--git_dirty', default=None, type=str, help='[debug] Indicates whether there are uncommited changes.')
    parser.add_argument('--uuid', default=None, type=str, help='[debug] Unique identifier for experiment.')
    parser.add_argument('--hostname', default=None, type=str)

    # Pytorch
    parser.add_argument('--cuda', action='store_true', help='If true, use GPU.')

    # Logging.
    parser.add_argument('--default_experiment_directory', default='./log', type=str)
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--experiment_path', default=None, type=str)
    parser.add_argument('--log_every_batch', default=10, type=int, help='Log every N updates.')
    parser.add_argument('--save_latest', default=1000, type=int, help='Save a checkpoint every N updates. Overwrites previous checkpoints.')
    parser.add_argument('--save_distinct', default=50000, type=int, help='Save a checkpoint every N updates. Does not overwrite.')
    parser.add_argument('--save_after', default=1000, type=int, help='Only save after N updates.')
    parser.add_argument('--save_init', action='store_true', help='If true, save initialization.')

    # Data.
    parser.add_argument('--init_vocab_path', default=None, type=str,
                        help='If provided, add vocab from this file. Useful for certain cases such as adversarial word substitution.')
    parser.add_argument('--data_type', default='nli', type=str)
    parser.add_argument('--train_data_type', default=None, type=str)
    parser.add_argument('--validation_data_type', default=None, type=str)
    parser.add_argument('--train_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_train.jsonl'), type=str)
    parser.add_argument('--validation_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_dev.jsonl'), type=str)
    parser.add_argument('--embeddings_path', default=os.path.expanduser('~/data/glove/glove.6B.300d.txt'), type=str,
                        help='Necessary if using word embeddings (w2v). Not used if character-based embeddings set (elmo).')

    # Data (preprocessing).
    parser.add_argument('--nolowercase', action='store_true')
    parser.add_argument('--require_ner', action='store_true')
    parser.add_argument('--require_output_vocab', action='store_true')
    parser.add_argument('--output_vocab', default=None, type=str)
    parser.add_argument('--train_filter_length', default=50, type=int)
    parser.add_argument('--validation_filter_length', default=0, type=int)

    # Loading.
    parser.add_argument('--load_model_path', default=None, type=str)

    # Evaluation.
    parser.add_argument('--eval_every_batch', default=1000, type=int)
    parser.add_argument('--eval_every_epoch', default=-1, type=int)
    parser.add_argument('--eval_after', default=0, type=int)
    parser.add_argument('--eval_config', default=None, action='append')

    # Model.
    parser.add_argument('--input_config', default=None, type=str)
    parser.add_argument('--model_config', default=None, type=str)
    parser.add_argument('--loss_config', default=None, action='append')

    # Model (Embeddings).
    parser.add_argument('--emb', default='w2v', choices=('w2v', 'elmo'))
    parser.add_argument('--emb_lang', default='en', type=str)
    parser.add_argument('--projection', default='word2vec', choices=('word2vec',))

    # ELMo
    parser.add_argument('--elmo_options_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', type=str)
    parser.add_argument('--elmo_weights_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', type=str)
    parser.add_argument('--elmo_cache_dir', default=None, type=str)
    parser.add_argument('--emb_cache_dir', default=None, type=str)
    parser.add_argument('--version', default='medium', type=str)

    # Training.
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size.')
    parser.add_argument('--accum_steps', default=1, type=int, help='Hyperparam for gradient accumulation.')
    parser.add_argument('--train_dataset_size', default=None, type=int)
    parser.add_argument('--validation_dataset_size', default=None, type=int)
    parser.add_argument('--validation_batch_size', default=None, type=int)
    parser.add_argument('--max_epoch', default=5000, type=int, help='Max number of training epochs (if > 0).')
    parser.add_argument('--max_step', default=-1, type=int, help='Max number of gradient steps (if > 0).')
    parser.add_argument('--eval_only_mode', action='store_true', help='Run evaluation only.')

    # Optimization.
    opt_types = ('sgd', 'adam')
    parser.add_argument('--opt', default='sgd', choices=opt_types)
    parser.add_argument('--lr', default=4e-3, type=float, help='Learning rate.')
    parser.add_argument('--num_warmup_steps', default=1, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--mlp_reg', default=0, type=float)
    parser.add_argument('--clip_threshold', default=5.0, type=float)
    parser.add_argument('--warmup', default=1000, type=int)
    parser.add_argument('--use_lr_scheduler', action='store_true')
    parser.add_argument('--lr_metric', default=None, type=str)

    return parser


def parse_args(parser):
    options, other_args = parser.parse_known_args()

    # Set default flag values (data).
    options.train_data_type = options.data_type if options.train_data_type is None else options.train_data_type
    options.validation_data_type = options.data_type if options.validation_data_type is None else options.validation_data_type
    options.validation_batch_size = options.batch_size if options.validation_batch_size is None else options.validation_batch_size

    # Set default flag values (config).
    if not options.git_branch_name:
        options.git_branch_name = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not options.torch_version:
        options.torch_version = torch.__version__

    if not options.git_sha:
        options.git_sha = os.popen('git rev-parse HEAD').read().strip()

    if not options.git_dirty:
        options.git_dirty = os.popen("git diff --quiet && echo 'clean' || echo 'dirty'").read().strip()

    if not options.uuid:
        options.uuid = str(uuid.uuid4())

    if not options.hostname:
        options.hostname = os.popen('hostname').read().strip()

    if not options.experiment_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%s")
        options.experiment_name = '{}-{}'.format(options.train_data_type, timestamp)

    if not options.experiment_path:
        options.experiment_path = os.path.join(options.default_experiment_directory, options.experiment_name)

    for k, v in options.__dict__.items():
        if type(v) == str and v.startswith('~'):
            options.__dict__[k] = os.path.expanduser(v)

    if options.elmo_cache_dir is not None:
        print('WARNING: Use `emb_cache_dir` instead. Rewriting `emb_cache_dir` with `elmo_cache_dir`, but `elmo_cache_dir` will be removed in future versions.')
        options.emb_cache_dir = options.elmo_cache_dir

    # Create a `no`-prefix equivalent for all boolean flags.
    options.lowercase = not options.nolowercase

    # Random seed.
    if options.seed is None:
        options.seed = np.random.randint(2147483648)

    options.hidden_dim = list(json.loads(options.model_config).values())[0].get('size', 400)

    return options


def configure(options):
    # Configure output paths for this experiment.
    configure_experiment(options.experiment_path)

    # Get logger.
    logger = get_logger()

    # Print flags.
    logger.info(stringify_flags(options))

    save_flags(options, options.experiment_path)


if __name__ == '__main__':
    from preprocessing import set_random_seed

    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    set_random_seed(options.seed)

    run(options)

