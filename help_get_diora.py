import importlib
import json

from experiment_logger import get_logger


def build_model_class(key):
    module_name = 'model_{}'.format(key)
    class_name = '{}_class'.format(key)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_diora(options, context, config=None):
    config = json.loads(config)

    assert isinstance(config, dict), "Config with value {} is not type dict.".format(i, config)

    assert len(config.keys()) == 1, "Config should have 1 key only."

    name = list(config.keys())[0]

    # Use default config if it exists.
    kwargs_dict = get_default_kwargs(options, context)
    # Override defaults with user-defined values.
    kwargs_dict = get_override_kwargs(kwargs_dict, config[name])
    # Build and return.
    return BuildComponent(name, kwargs_dict).build(context)


class BuildComponent(object):
    def __init__(self, name, kwargs_dict):
        self.name = name
        self.kwargs_dict = kwargs_dict
        self.logger = get_logger()

    def build(self, context):
        kwargs_dict = self.kwargs_dict.copy()
        kwargs_dict['projection_layer'] = None
        clz = build_model_class(self.name)
        self.logger.info('building diora name = {}, class = {}'.format(self.name, clz))
        self.logger.info('and kwargs = {}'.format(json.dumps(kwargs_dict)))
        return clz.from_kwargs_dict(context, self.kwargs_dict)


def get_override_kwargs(kwargs_dict, user_defined_kwargs):
    kwargs_dict = kwargs_dict.copy()
    for k, v in user_defined_kwargs.items():
        kwargs_dict[k] = v
    return kwargs_dict


def get_default_kwargs(options, context):
    size = 400
    normalize = 'unit'
    n_layers = 2

    return dict(size=size, outside=True, normalize=normalize,  n_layers=n_layers)
