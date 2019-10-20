
import abc
import threading


from typing import Union, List
from collections import UserDict


class TradingContext(UserDict):
    """Functionality for objects that put themselves in a base using
    the `with` statement.

    This implementation for this class is heavily borrowed from the pymc3
    library and adapted with the design goals of TensorTrade in mind.

    Parameters
    ----------
    base_instrument : str

    products : List[str]
        The exchange symbols of the instruments being traded.

    credentials : dict


    Reference:
        - https://github.com/pymc-devs/pymc3/blob/master/pymc3/model.py
    """
    contexts = threading.local()

    def __init__(self,
                 base_instrument: str = 'USD',
                 products: Union[str, List[str]] = 'BTC',
                 credentials: dict = None,
                 **config):
        super(TradingContext, self).__init__(
            base_instrument=base_instrument,
            products=products,
            credentials=credentials,
            **config
        )
        self._base_instrument = base_instrument
        if type(products) == str:
            products = [products]
        self._products = products
        self._credentials = credentials

        self.__dict__ = {**self.__dict__, **self.data}

    @property
    def base_instrument(self) -> str:
        return self._base_instrument

    @base_instrument.setter
    def base_instrument(self, base_instrument: str):
        self._base_instrument = base_instrument

    @property
    def products(self) -> List[str]:
        return self._products

    @products.setter
    def product(self, products: Union[str, List[str]]):
        self._products = products

    @property
    def credentials(self) -> dict:
        return self._credentials

    @credentials.setter
    def credentials(self, credentials: dict):
        self._credentials = credentials

    def __enter__(self):
        type(self).get_contexts().append(self)
        return self

    def __exit__(self, typ, value, traceback):
        type(self).get_contexts().pop()

    def __str__(self):
        data = ['{}={}'.format(k, getattr(self, k)) for k in self.__slots__]
        return '<{}: {}>'.format(self.__class__.__name__, ', '.join(data))

    @classmethod
    def get_contexts(cls):
        if not hasattr(cls.contexts, 'stack'):
            cls.contexts.stack = []
        return cls.contexts.stack

    @classmethod
    def get_context(cls):
        """Gets the deepest context on the stack."""
        try:
            return cls.get_contexts()[-1]
        except IndexError:
            raise TypeError("No context on context stack")


class InitContextMeta(abc.ABCMeta):
    """Metaclass that executes `__init__` of instance in it's base"""

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        try:
            setattr(instance, 'context', TradingContext.get_context())
        except TypeError:
            setattr(instance, 'context', None)
        instance.__init__(*args, **kwargs)
        return instance


class ContextualizedMixin(object):
    """This class is to be mixed in with any class that must function in a
    contextual setting.
    """
    @property
    def context(self) -> TradingContext:
        return self._context

    @context.setter
    def context(self, context: TradingContext):
        self._context = context
