import threading
import json
import yaml

from collections import UserDict
from typing import List

import numpy as np

from . import registry


class TradingContext(UserDict):
    """A class for objects that put themselves in a `Context` using
    the `with` statement.

    The implementation for this class is heavily borrowed from the pymc3
    library and adapted with the design goals of TensorTrade in mind.

    Parameters
    ----------
    config : dict
        The configuration holding the information for each `Component`.

    Methods
    -------
    from_json(path)
        Creates a `TradingContext` from a json file.
    from_yaml(path)
        Creates a `TradingContext` from a yaml file.

    Warnings
    --------
    If there is a conflict in the contexts of different components because
    they were initialized under different contexts, can have undesirable effects.
    Therefore, a warning should be made to the user indicating that using
    components together that have conflicting contexts can lead to unwanted
    behavior.

    References
    ----------
    [1] https://github.com/pymc-devs/pymc3/blob/master/pymc3/model.py
    """

    contexts = threading.local()

    def __init__(self, config: dict):
        super().__init__(**config)

        r = registry.registry()
        registered_names = list(np.unique([r[i] for i in r.keys()]))

        for name in registered_names:
            if name not in registry.MAJOR_COMPONENTS:
                setattr(self, name, config.get(name, {}))

        config_items = {k: config[k]
                        for k in config.keys()
                        if k not in registered_names}

        self._config = config
        self._shared = config.get('shared', {})

        self._shared = {
            **self._shared,
            **config_items
        }

    @property
    def shared(self) -> dict:
        """The shared values in common for all components involved with the
        `TradingContext`.

        Returns
        -------
        dict
            Shared values for components under the `TradingContext`.
        """
        return self._shared

    def __enter__(self) -> 'TradingContext':
        """Adds a new `TradingContext` to the context stack.

        This method is used for a `with` statement and adds a `TradingContext`
        to the context stack. The new context on the stack is then used by every
        class that subclasses `Component` the initialization of its instances.

        Returns
        -------
        `TradingContext`
            The context associated with the given with statement.
        """
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback) -> None:
        """Pops the first `TradingContext` of the stack.

        Parameters
        ----------
        typ : type
            The type of `Exception`
        value : `Exception`
            An instance of `typ`.
        traceback : python traceback object
            The traceback object associated with the exception.
        """
        type(self).get_contexts().pop()

    @classmethod
    def get_contexts(cls) -> List['TradingContext']:
        """Gets the stack of trading contexts.

        Returns
        -------
        List['TradingContext']
            The stack of trading contexts.
        """
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = [TradingContext({})]
        return cls.contexts.stack

    @classmethod
    def get_context(cls) -> 'TradingContext':
        """Gets the first context on the stack.

        Returns
        -------
        `TradingContext`
            The first context on the stack.
        """
        return cls.get_contexts()[-1]

    @classmethod
    def from_json(cls, path: str) -> 'TradingContext':
        """Creates a `TradingContext` from a json file.

        Parameters
        ----------
        path : str
            The path to locate the json file.

        Returns
        -------
        `TradingContext`
            A trading context with all the variables provided in the json file.
        """
        with open(path, "rb") as fp:
            config = json.load(fp)
        return TradingContext(config)

    @classmethod
    def from_yaml(cls, path: str) -> 'TradingContext':
        """Creates a `TradingContext` from a yaml file.

        Parameters
        ----------
        path : str
            The path to locate the yaml file.

        Returns
        -------
        `TradingContext`
            A trading context with all the variables provided in the yaml file.
        """
        with open(path, "rb") as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
        return TradingContext(config)


class Context(UserDict):
    """A context that is injected into every instance of a class that is
    a subclass of `Component`.
    """

    def __init__(self, **kwargs):
        super(Context, self).__init__(**kwargs)
        self.__dict__ = {**self.__dict__, **self.data}

    def __str__(self):
        data = ['{}={}'.format(k, getattr(self, k)) for k in self.__slots__]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))
