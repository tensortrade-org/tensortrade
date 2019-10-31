import threading
import json
import yaml

from typing import Union, List
from collections import UserDict
from .registry import registered_names, get_major_component_names, registry


def _diff(c1, c2, path=None, changes=None):
    path = [] if not path else path
    changes = [] if not changes else changes
    old_path = path
    for k in c1.keys():
        path = old_path + [k]
        if k not in c2:
            changes += [('key', path, True, False)]
        else:
            if isinstance(c1[k], dict) and isinstance(c2[k], dict):
                changes = _diff(c1[k], c2[k], path, changes)
            else:
                if c1[k] != c2[k]:
                    changes += [('value', path, c1[k], c2[k])]

    for k in c2.keys():
        path = old_path + [k]
        if k not in c1:
            changes += [('key', path, False, True)]
    return changes


def diff(c1, c2):
    return _diff(c1, c2)


def is_conflicting(d1, d2):
    for d in diff(d1, d2):
        if d[0] == 'value':
            return True
    return False


class TradingContext(UserDict):
    """A class that objects that put themselves in a `Context` using
    the `with` statement.

    This implementation for this class is heavily borrowed from the pymc3
    library and adapted with the design goals of TensorTrade in mind.

    Parameters:
    ----------
    shared : Context
        A context that is shared between all components that are made under
        the overarching `TradingContext`.
    exchanges : Context
        A context that is specific to components with a registered name of
        `exchanges`.
    actions : Context
        A context that is specific to components with a registered name of
        `actions`.
    rewards : Context
        A context that is specific to components with a registered name of
        `rewards`.
    features : Context
        A context that is specific to components with a registered name of
        `features`.

    Warnings:
    --------
        If there is a conflict in the contexts of different components because
    they were initialized under different contexts, can have undesirable effects.
    Therefore, a warning should be made to the user indicating that using
    components together that have conflicting contexts can lead to unwanted
    behavior.

    Reference:
    ---------
        - https://github.com/pymc-devs/pymc3/blob/master/pymc3/model.py

    """
    contexts = threading.local()

    def __init__(self,
                 base_instrument: str = 'USD',
                 products: Union[str, List[str]] = 'BTC',
                 **config):
        super(TradingContext, self).__init__(
            base_instrument=base_instrument,
            products=products,
            **config
        )
        if type(products) == str:
            products = [products]

        for name in registered_names():
            if name not in get_major_component_names():
                setattr(self, name, config.get(name, {}))

        self._shared = config.get('shared', {})
        self._exchanges = config.get('exchanges', {})
        self._actions = config.get('actions', {})
        self._rewards = config.get('rewards', {})
        self._features = config.get('features', {})
        self._slippage = config.get('slippage', {})

        self._shared = {
            'base_instrument': base_instrument,
            'products': products,
            **self._shared,
            **{k: config[k] for k in config.keys()
                if k not in registered_names()}
        }

    @property
    def shared(self) -> dict:
        return self._shared

    @property
    def exchanges(self) -> dict:
        return self._exchanges

    @property
    def actions(self) -> dict:
        return self._actions

    @property
    def rewards(self) -> dict:
        return self._rewards

    @property
    def features(self) -> dict:
        return self._features

    @property
    def slippage(self) -> dict:
        return self._slippage

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
            cls.contexts.stack = [TradingContext()]
        return cls.contexts.stack

    @classmethod
    def get_context(cls):
        """Gets the deepest context on the stack."""
        return cls.get_contexts()[-1]

    @classmethod
    def from_json(cls, path: str):
        with open(path, "rb") as fp:
            config = json.load(fp)
        return TradingContext(**config)

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "rb") as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
        return TradingContext(**config)


class Context(UserDict):
    """A context that is injected into every instance of a class that is
    a subclass of component.

    Parameters:
    ----------
    base_instrument : str
        The exchange symbol of the instrument to store/measure value in.
    products : List[str]
        The exchange symbols of the instruments being traded on.
    """

    def __init__(self,
                 base_instrument: str = 'USD',
                 products: Union[str, List[str]] = 'BTC',
                 **kwargs):
        super(Context, self).__init__(
            base_instrument=base_instrument,
            products=products,
            **kwargs
        )
        self._base_instrument = base_instrument
        self._products = products
        self.__dict__ = {**self.__dict__, **self.data}

    @property
    def base_instrument(self):
        return self._base_instrument

    @property
    def products(self):
        return self._products

    def __str__(self):
        data = ['{}={}'.format(k, getattr(self, k)) for k in self.__slots__]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))