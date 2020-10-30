
from typing import List, Callable


class DataTypeMixin:

    @classmethod
    def register_method(cls, func: "Callable", names: "List[str]"):
        """Injects methods into a specific stream instance.

        Parameters
        ----------
        func : `Callable`
            The function to be injected as a method.
        names : `List[str]`
            The names to be given to the function.
        """
        def method(self, *args, **kwargs):
            args = (self,) + args
            return func(*args, **kwargs)

        for name in names:
            setattr(cls, name, method)
