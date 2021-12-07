
from typing import List

from tensortrade.feed.core import Stream
from tensortrade.feed.core.methods import Methods
from tensortrade.feed.core.mixins import DataTypeMixin


@Stream.register_accessor(name="float")
class FloatMethods(Methods):
    ...


@Stream.register_mixin(dtype="float")
class FloatMixin(DataTypeMixin):
    ...


class Float:
    """A class to register accessor and instance methods."""

    @classmethod
    def register(cls, names: List[str]):
        """A function decorator that adds accessor and instance methods for
        specified data type.

        Parameters
        ----------
        names : `List[str]`
            A list of names used to register the function as a method.

        Returns
        -------
        Callable
            A decorated function.
        """
        def wrapper(func):
            FloatMethods.register_method(func, names)
            FloatMixin.register_method(func, names)
            return func
        return wrapper


from .window import *
from .accumulators import *
from .imputation import *
from .operations import *
from .ordering import *
from .utils import *
