
import abc

from .context import ContextualizedMixin, InitContextMeta


class Component(abc.ABC, ContextualizedMixin, metaclass=InitContextMeta):
    pass
