import importlib
import json

from input_layer import *


def build_loss_class(key):
    module_name = 'loss_{}'.format(key)
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
        clz = build_loss_class(self.name)
        self.logger.info('building loss component name = {}, class = {}'.format(self.name, clz))
        self.logger.info('with kwargs = {}'.format(self.kwargs_dict))
        return clz.from_kwargs_dict(context, self.kwargs_dict)


def get_default_kwargs(options, context):
    kwargs_dict = {}
    kwargs_dict['input_size'] = options.input_dim
    kwargs_dict['size'] = options.hidden_dim
    return kwargs_dict


def get_override_kwargs(kwargs_dict, user_defined_kwargs):
    kwargs_dict = kwargs_dict.copy()
    for k, v in user_defined_kwargs.items():
        kwargs_dict[k] = v
    return kwargs_dict


def get_loss_funcs(options, context, config_lst=None):
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
        kwargs_dict = get_default_kwargs(options, context)
        # Override defaults with user-defined values.
        kwargs_dict = get_override_kwargs(kwargs_dict, config[name])
        # Build and append component.
        result.append(BuildComponent(name, kwargs_dict).build(context))

    return result
