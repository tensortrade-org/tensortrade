
from typing import List, Callable


class Methods:
    """A class used to hold the accessor methods for a particular data type.

    Parameters
    ----------
    stream : "Stream"
        The stream to injected with the method accessor.
    """

    def __init__(self, stream: "Stream"):
        self.stream = stream

    @classmethod
    def _make_accessor(cls, stream: "Stream"):
        return cls(stream)

    @classmethod
    def register_method(cls, func: Callable, names: "List[str]"):
        """Injects an accessor into a specific stream instance.

        Parameters
        ----------
        func : `Callable`
            The function to be injected as an accessor method.
        names : `List[str]`
            The names to be given to the function.
        """
        def method(self, *args, **kwargs):
            args = (self.stream,) + args
            return func(*args, **kwargs)
        for name in names:
            setattr(cls, name, method)
        return method
