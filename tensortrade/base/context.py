import threading
import json
import yaml

from collections import UserDict

from .registry import registered_names, get_major_component_names


class TradingContext(UserDict):
    """A class for objects that put themselves in a `Context` using
    the `with` statement.

    The implementation for this class is heavily borrowed from the pymc3
    library and adapted with the design goals of TensorTrade in mind.

    Arguments:
        shared: A context that is shared between all components that are made under the overarching `TradingContext`.

    Warnings:
        If there is a conflict in the contexts of different components because
        they were initialized under different contexts, can have undesirable effects.
        Therefore, a warning should be made to the user indicating that using
        components together that have conflicting contexts can lead to unwanted
        behavior.

    Reference:
        - https://github.com/pymc-devs/pymc3/blob/master/pymc3/model.py

    """
    contexts = threading.local()

    def __init__(self, config: dict):
        super().__init__(**config)

        for name in registered_names():
            if name not in get_major_component_names():
                setattr(self, name, config.get(name, {}))

        config_items = {k: config[k] for k in config.keys()
                        if k not in registered_names()}

        self._config = config
        self._shared = config.get('shared', {})

        self._shared = {
            **self._shared,
            **config_items
        }

    @property
    def shared(self) -> dict:
        return self._shared

    def __enter__(self):
        """Adds a new context to the context stack.

        This method is used for a `with` statement and adds a `TradingContext`
        to the context stack. The new context on the stack is then used by every
        class that subclasses `Component` the initialization of its instances.
        """
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = [TradingContext({})]

        return cls.contexts.stack

    @classmethod
    def get_context(cls):
        """Gets the deepest context on the stack."""
        return cls.get_contexts()[-1]

    @classmethod
    def from_json(cls, path: str):
        with open(path, "rb") as fp:
            config = json.load(fp)

        return TradingContext(config)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "rb") as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)

        return TradingContext(config)


class Context(UserDict):
    """A context that is injected into every instance of a class that is
    a subclass of component.
    """

    def __init__(self, **kwargs):
        super(Context, self).__init__(**kwargs)

        self.__dict__ = {**self.__dict__, **self.data}

    def __str__(self):
        data = ['{}={}'.format(k, getattr(self, k)) for k in self.__slots__]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))
