
import abc

from .context import TradingContext, Context
from .registry import registry, register, get_registered_name


class InitContextMeta(abc.ABCMeta):
    """Metaclass that executes `__init__` of instance in it's base"""

    def __call__(cls, *args, **kwargs):
        instance = cls.__new__(cls, *args, **kwargs)
        registered_name = registry()[cls]
        tc = TradingContext.get_context()
        shared = tc.shared
        data = tc.data.get(registered_name, {})
        config = {**shared, **data}
        setattr(instance, 'context', Context(**config))
        instance.__init__(*args, **kwargs)
        return instance


class ContextualizedMixin(object):
    """This class is to be mixed in with any class that must function in a
    contextual setting.
    """
    @property
    def context(self) -> Context:
        return self._context

    @context.setter
    def context(self, context: Context):
        self._context = context


class Component(abc.ABC, ContextualizedMixin, metaclass=InitContextMeta):

    registered_name = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls not in registry():
            register(cls, cls.registered_name)
