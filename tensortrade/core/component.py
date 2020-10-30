
from abc import ABC, ABCMeta
from typing import Any

from . import registry
from tensortrade.core.context import TradingContext, Context
from tensortrade.core.base import Identifiable


class InitContextMeta(ABCMeta):
    """Metaclass that executes `__init__` of instance in it's core.

    This class works with the `TradingContext` class to ensure the correct
    data is being given to the instance created by a concrete class that has
    subclassed `Component`.
    """

    def __call__(cls, *args, **kwargs) -> 'InitContextMeta':
        """

        Parameters
        ----------
        args :
            positional arguments to give constructor of subclass of `Component`
        kwargs :
            keyword arguments to give constructor of subclass of `Component`

        Returns
        -------
        `Component`
            An instance of a concrete class the subclasses `Component`
        """
        context = TradingContext.get_context()
        registered_name = registry.registry()[cls]

        data = context.data.get(registered_name, {})
        config = {**context.shared, **data}

        instance = cls.__new__(cls, *args, **kwargs)
        setattr(instance, 'context', Context(**config))
        instance.__init__(*args, **kwargs)

        return instance


class ContextualizedMixin(object):
    """A mixin that is to be mixed with any class that must function in a
    contextual setting.
    """

    @property
    def context(self) -> Context:
        """Gets the `Context` the object is under.

        Returns
        -------
        `Context`
            The context the object is under.
        """
        return self._context

    @context.setter
    def context(self, context: Context) -> None:
        """Sets the context for the object.

        Parameters
        ----------
        context : `Context`
            The context to set for the object.
        """
        self._context = context


class Component(ABC, ContextualizedMixin, Identifiable, metaclass=InitContextMeta):
    """The main class for setting up components to be used in the `TradingEnv`.

    This class if responsible for providing a common way in which different
    components of the library can be created. Specifically, it enables the
    creation of components from a `TradingContext`. Therefore making the creation
    of complex environments simpler where there are only a few things that
    need to be changed from case to case.

    Attributes
    ----------
    registered_name : str
        The name under which constructor arguments are to be given in a dictionary
        and passed to a `TradingContext`.
    """

    registered_name = None

    def __init_subclass__(cls, **kwargs) -> None:
        """Constructs the concrete subclass of `Component`.

        In constructing the subclass, the concrete subclass is also registered
        into the project level registry.

        Parameters
        ----------
        kwargs : keyword arguments
            The keyword arguments to be provided to the concrete subclass of `Component`
            to create an instance.
        """
        super().__init_subclass__(**kwargs)

        if cls not in registry.registry():
            registry.register(cls, cls.registered_name)

    def default(self, key: str, value: Any, kwargs: dict = None) -> Any:
        """Resolves which defaults value to use for construction.

        A concrete subclass will use this method to resolve which default value
        it should use when creating an instance. The default value should go to
        the value specified for the variable within the `TradingContext`. If that
        one is not provided it will resolve to `value`.

        Parameters
        ----------
        key : str
            The name of the attribute to be resolved for the class.
        value : any
            The `value` the attribute should be set to if not provided in the
            `TradingContext`.
        kwargs : dict, optional
            The dictionary to search through for the value associated with `key`.
        """
        if not kwargs:
            return self.context.get(key, None) or value
        return self.context.get(key, None) or kwargs.get(key, value)
