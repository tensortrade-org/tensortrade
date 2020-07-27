
from tensortrade.feed.core import Stream
from tensortrade.feed.core.methods import Methods
from tensortrade.feed.core.mixins import DataTypeMixin


@Stream.register_accessor(name="str")
class StringMethods(Methods):
    ...


@Stream.register_mixin(dtype="string")
class StringMixin(DataTypeMixin):
    ...


class String:

    @classmethod
    def register(cls, names):
        def wrapper(func):
            StringMethods.register_method(func, names)
            StringMixin.register_method(func, names)
            return func
        return wrapper
