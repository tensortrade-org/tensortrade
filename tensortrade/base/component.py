
from abc import ABC, ABCMeta


from .context import TradingContext, Context
from .registry import get_registry, register
from .core import Identifiable


class InitContextMeta(ABCMeta):
    """Metaclass that executes `__init__` of instance in it's base."""

    def __call__(cls, *args, **kwargs):
        registered_name = get_registry()[cls]
        tc = TradingContext.get_context()
        data = tc.data.get(registered_name, {})
        config = {**tc.shared, **data}

        instance = cls.__new__(cls, *args, **kwargs)
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


class Component(ABC, ContextualizedMixin, Identifiable, metaclass=InitContextMeta):

    registered_name = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls not in get_registry():
            register(cls, cls.registered_name)

    def default(self, key: str, value: any, kwargs: dict = None):
        if not kwargs:
            return self.context.get(key, None) or value

        return self.context.get(key, None) or kwargs.get(key, value)
