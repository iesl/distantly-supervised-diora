import importlib
import os
import json

from evaluation_utils import BaseEvalFunc
from experiment_logger import get_logger


def build_eval_class(key):
    module_name = 'eval_{}'.format(key)
    class_name = '{}_class'.format(key)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


class BuildComponent(object):
    def __init__(self, name, kwargs_dict):
        self.name = name
        self.kwargs_dict = kwargs_dict
        self.logger = get_logger()

    def build(self, context):
        clz = build_eval_class(self.name)
        self.logger.info('building eval component name = {}, class = {}'.format(self.name, clz))
        kwargs_dict = self.kwargs_dict.copy()
        if 'name' not in kwargs_dict:
            kwargs_dict['name'] = self.name
        return clz.from_kwargs_dict(context, kwargs_dict)


class ModelEvaluation(object):
    def __init__(self, components):
        super(ModelEvaluation, self).__init__()
        self.validate(components)
        self.components = components

    def validate(self, components):
        check = set([func.name for func in components])
        assert len(check) == len(components), "Each name must be unique."

    def run(self, trainer, info, metadata):
        for func in self.components:
            assert isinstance(func, BaseEvalFunc), "All eval funcs should be subclass of BaseEvalFunc."
            assert hasattr(func, 'is_initialized') and func.is_initialized == True, \
                "Do not override __init__ for BaseEvalFunc, instead use `init_defaults` or `from_kwargs_dict`."
            if not func.enabled:
                continue
            step = trainer.optimizer_step - 1 # TODO: This is off by one, but matches original functionality.
            outfile = os.path.join(info['experiment_path'], 'eval_{}.step_{}.txt'.format(func.name, step))
            info['outfile'] = outfile
            result = func.run(trainer, info)
            yield {'result': result, 'component': func}


def get_default_kwargs(options):
    return {}


def get_override_kwargs(kwargs_dict, user_defined_kwargs):
    kwargs_dict = kwargs_dict.copy()
    for k, v in user_defined_kwargs.items():
        kwargs_dict[k] = v
    return kwargs_dict


def get_eval_components(options, context=dict(), config_lst=None):
    assert isinstance(config_lst, (list, tuple)), "There should be a `list` of configs."

    result = []

    for i, config_str in enumerate(config_lst):
        config = json.loads(config_str)

        assert isinstance(config, dict), "Config[{}] with value {} is not type dict.".format(i, config)

        assert len(config.keys()) == 1, "Each config should have 1 key only."

        name = list(config.keys())[0]

        if not config[name].get('enabled', True):
            continue

        # Use default config if it exists.
        kwargs_dict = get_default_kwargs(options)
        # Override defaults with user-defined values.
        kwargs_dict = get_override_kwargs(kwargs_dict, config[name])
        # Build and append component.
        result.append(BuildComponent(name, kwargs_dict).build(context))

    return result
