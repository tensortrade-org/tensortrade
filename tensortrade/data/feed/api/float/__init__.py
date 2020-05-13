

from tensortrade.data.feed.core.base import Stream
from tensortrade.data.feed.core.methods import Methods
from tensortrade.data.feed.core.mixins import DataTypeMixin


@Stream.register_accessor(name="float")
class FloatMethods(Methods):
    ...


@Stream.register_mixin(dtype="float")
class FloatMixin(DataTypeMixin):
    ...


class Float:

    @classmethod
    def register(cls, names):
        def wrapper(func):
            FloatMethods.register_method(func, names)
            FloatMixin.register_method(func, names)
            return func
        return wrapper
