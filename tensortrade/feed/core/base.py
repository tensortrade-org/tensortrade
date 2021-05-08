
import inspect

from abc import abstractmethod
from typing import (
    Generic,
    Iterable,
    TypeVar,
    Dict,
    Any,
    Callable,
    List,
    Tuple
)

from tensortrade.core import Observable
from tensortrade.feed.core.accessors import CachedAccessor
from tensortrade.feed.core.mixins import DataTypeMixin


T = TypeVar("T")


class Named:
    """A class for controlling the naming of objects.

    The purpose of this class is to control the naming of objects with respect
    to the `NameSpace` to which they belong to. This prevents conflicts that
    arise in the naming of similar objects under different contexts.

    Parameters
    ----------
    name : str, optional
        The name of the object.

    Attributes
    ----------
    name : str, optional
        The name of the object.
    """

    generic_name: str = "generic"
    namespaces: List[str] = []
    names: Dict[str, int] = {}

    def __init__(self, name: str = None):
        if not name:
            name = self.generic_name

            if name in Stream.names.keys():
                Stream.names[name] += 1
                name += ":/" + str(Stream.names[name] - 1)
            else:
                Stream.names[name] = 0
        self.name = name

    def rename(self, name: str, sep: str = ":/") -> "Named":
        """Renames the instance with respect to the current `NameSpace`.

        Parameters
        ----------
        name : str
            The new name to give to the instance.
        sep : str
            The separator to put between the name of the `NameSpace` and the
            new name of the instance (e.g. ns:/example).

        Returns
        -------
        `Named`
            The instance that was renamed.

        """
        if len(Named.namespaces) > 0:
            name = Named.namespaces[-1] + sep + name
        self.name = name
        return self


class NameSpace(Named):
    """A class providing a context in which to create names.

    This becomes useful in cases where `Named` object would like to use the
    same name in a different context. In order to resolve naming conflicts in
    a `DataFeed`, this class provides a way to solve it.

    Parameters
    ----------
    name : str
        The name for the `NameSpace`.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __enter__(self) -> None:
        Named.namespaces += [self.name]

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        Named.namespaces.pop()


class Stream(Generic[T], Named, Observable):
    """A class responsible for creating the inputs necessary to work in a
    `DataFeed`.

    Parameters
    ----------
    name : str, optional
        The name fo the stream.
    dtype : str, optional
        The data type of the stream.

    Methods
    -------
    source(iterable, dtype=None)
        Creates a stream from an iterable.
    group(streams)
        Creates a group of streams.
    sensor(obj,func,dtype=None)
        Creates a stream from observing a value from an object.
    select(streams,func)
        Selects a stream satisfying particular criteria from a list of
        streams.
    constant(value,dtype)
        Creates a stream to generate a constant value.
    asdtype(dtype)
        Converts the data type to `dtype`.
    """

    _mixins: "Dict[str, DataTypeMixin]" = {}
    _accessors: "List[CachedAccessor]" = []
    generic_name: str = "stream"

    def __new__(cls, *args, **kwargs):
        dtype = kwargs.get("dtype")
        instance = super().__new__(cls)
        if dtype in Stream._mixins.keys():
            mixin = Stream._mixins[dtype]
            instance = Stream.extend_instance(instance, mixin)
        return instance

    def __init__(self, name: str = None, dtype: str = None):
        Named.__init__(self, name)
        Observable.__init__(self)
        self.dtype = dtype
        self.inputs = []
        self.value = None

    def __call__(self, *inputs) -> "Stream[T]":
        """Connects the inputs to this stream.

        Parameters
        ----------
        *inputs : positional arguments
            The positional arguments, each a stream to be connected as an input
            to this stream.

        Returns
        -------
        `Stream[T]`
            The current stream inputs are being connected to.
        """
        self.inputs = inputs
        return self

    def run(self) -> None:
        """Runs the underlying streams once and iterates forward."""
        self.value = self.forward()
        for listener in self.listeners:
            listener.on_next(self.value)

    @abstractmethod
    def forward(self) -> T:
        """Generates the next value from the underlying data streams.

        Returns
        -------
        `T`
            The next value in the stream.
        """
        raise NotImplementedError()

    @abstractmethod
    def has_next(self) -> bool:
        """Checks if there is another value.

        Returns
        -------
        bool
            If there is another value or not.
        """
        raise NotImplementedError()

    def astype(self, dtype: str) -> "Stream[T]":
        """Converts the data type to `dtype`.

        Parameters
        ----------
        dtype : str
            The data type to be converted to.

        Returns
        -------
        `Stream[T]`
            The same stream with the new underlying data type `dtype`.
        """
        self.dtype = dtype
        mixin = Stream._mixins[dtype]
        return Stream.extend_instance(self, mixin)

    def reset(self) -> None:
        """Resets all inputs to and listeners of the stream and sets stream value to None."""
        for listener in self.listeners:
            if hasattr(listener, "reset"):
                listener.reset()
        
        for stream in self.inputs:
            stream.reset()

        self.value = None

    def gather(self) -> "List[Tuple[Stream, Stream]]":
        """Gathers all the edges of the DAG connected in ancestry with this
        stream.

        Returns
        -------
        `List[Tuple[Stream, Stream]]`
            The list of edges connected through ancestry to this stream.
        """
        return self._gather(self, [], [])

    @staticmethod
    def source(iterable: "Iterable[T]", dtype: str = None) -> "Stream[T]":
        """Creates a stream from an iterable.

        Parameters
        ----------
        iterable : `Iterable[T]`
            The iterable to create the stream from.
        dtype : str, optional
            The data type of the stream.

        Returns
        -------
        `Stream[T]`
            The stream with the data type `dtype` created from `iterable`.
        """
        return IterableStream(iterable, dtype=dtype)

    @staticmethod
    def group(streams: "List[Stream[T]]") -> "Stream[dict]":
        """Creates a group of streams.

        Parameters
        ----------
        streams : `List[Stream[T]]`
            Streams to be grouped together.

        Returns
        -------
        `Stream[dict]`
            A stream of dictionaries with each stream as a key/value in the
            dictionary being generated.
        """
        return Group()(*streams)

    @staticmethod
    def sensor(obj: "Any",
               func: "Callable[[Any], T]",
               dtype: str = None) -> "Stream[T]":
        """Creates a stream from observing a value from an object.

        Parameters
        ----------
        obj : `Any`
            An object to observe values from.
        func : `Callable[[Any], T]`
            A function to extract the data to be observed from the object being
            watched.
        dtype : str, optional
            The data type of the stream.

        Returns
        -------
        `Stream[T]`
            The stream of values being observed from the object.
        """
        return Sensor(obj, func, dtype=dtype)

    @staticmethod
    def select(streams: "List[Stream[T]]",
               func: "Callable[[Stream[T]], bool]") -> "Stream[T]":
        """Selects a stream satisfying particular criteria from a list of
        streams.

        Parameters
        ----------
        streams : `List[Stream[T]]`
            A list of streams to select from.
        func : `Callable[[Stream[T]], bool]`
            The criteria to be used for finding the particular stream.

        Returns
        -------
        `Stream[T]`
            The particular stream being selected.

        Raises
        ------
        Exception
            Raised if no stream is found to satisfy the given criteria.
        """
        for s in streams:
            if func(s):
                return s
        raise Exception("No stream satisfies selector condition.")

    @staticmethod
    def constant(value: "T", dtype: str = None) -> "Stream[T]":
        """Creates a stream to generate a constant value.

        Parameters
        ----------
        value : `T`
            The constant value to be streamed.
        dtype : str, optional
            The data type of the value.

        Returns
        -------
        `Stream[T]`
            A stream of the constant value.
        """
        return Constant(value, dtype=dtype)

    @staticmethod
    def placeholder(dtype: str = None) -> "Stream[T]":
        """Creates a placholder stream for data to provided to at a later date.

        Parameters
        ----------
        dtype : str
            The data type that will be provided.

        Returns
        -------
        `Stream[T]`
            A stream representing a placeholder.
        """
        return Placeholder(dtype=dtype)

    @staticmethod
    def _gather(stream: "Stream",
                vertices: "List[Stream]",
                edges: "List[Tuple[Stream, Stream]]") -> "List[Tuple[Stream, Stream]]":
        """Gathers all the edges relating back to this particular node.

        Parameters
        ----------
        stream : `Stream`
            The stream to inspect the connections of.
        vertices : `List[Stream]`
            The list of streams that have already been inspected.
        edges : `List[Tuple[Stream, Stream]]`
            The connections that have been found to be in the graph at the moment
            not including `stream`.

        Returns
        -------
        `List[Tuple[Stream, Stream]]`
            The updated list of edges after inspecting `stream`.
        """
        if stream not in vertices:
            vertices += [stream]

            for s in stream.inputs:
                edges += [(s, stream)]

            for s in stream.inputs:
                Stream._gather(s, vertices, edges)

        return edges

    @staticmethod
    def toposort(edges: "List[Tuple[Stream, Stream]]") -> "List[Stream]":
        """Sorts the order in which streams should be run.

        Parameters
        ----------
        edges : `List[Tuple[Stream, Stream]]`
            The connections that have been found in the DAG.

        Returns
        -------
        `List[Stream]`
            The list of streams sorted with respect to the order in which they
            should be run.
        """
        src = set([s for s, t in edges])
        tgt = set([t for s, t in edges])

        starting = list(src.difference(tgt))
        process = starting.copy()

        while len(starting) > 0:
            start = starting.pop()

            edges = list(filter(lambda e: e[0] != start, edges))

            src = set([s for s, t in edges])
            tgt = set([t for s, t in edges])

            starting += [v for v in src.difference(tgt) if v not in starting]

            if start not in process:
                process += [start]

        return process

    @classmethod
    def register_accessor(cls, name: str):
        """A class decorator that registers an accessor providing useful
        methods for a particular data type..

        Sets the data type accessor to be an attribute of this class.

        Parameters
        ----------
        name : str
            The name of the data type.
        """
        def wrapper(accessor):
            setattr(cls, name, CachedAccessor(name, accessor))
            cls._accessors += [name]
            return accessor
        return wrapper

    @classmethod
    def register_mixin(cls, dtype: str):
        """A class decorator the registers a data type mixin providing useful
        methods directly to the instance of the class.

        Parameters
        ----------
        dtype : str
            The name of the data type the mixin is being registered for.
        """
        def wrapper(mixin):
            cls._mixins[dtype] = mixin
            return mixin
        return wrapper

    @classmethod
    def register_generic_method(cls, names: "List[str]"):
        """A function decorator that registers the decorated function with the
        names provided as a method to the `Stream` class.

        These methods can be used for any instance of `Stream`.

        Parameters
        ----------
        names : `List[str]`
            The list of names to be used as aliases for the same method.
        """
        def wrapper(func):
            def method(self, *args, **kwargs):
                args = (self,) + args
                return func(*args, **kwargs)
            for name in names:
                setattr(Stream, name, method)
            return method
        return wrapper

    @staticmethod
    def extend_instance(instance: "Stream[T]", mixin: "DataTypeMixin") -> "Stream[T]":
        """Apply mix-ins to a class instance after creation.

        Parameters
        ----------
        instance : `Stream[T]`
            An instantiation of `Stream` to be injected with mixin methods.
        mixin : `DataTypeMixin`
            The mixin holding the methods to be injected into the `instance`.

        Returns
        -------
        `Stream[T]`
            The `instance` with the injected methods provided by the `mixin`.
        """
        base_cls = instance.__class__
        base_cls_name = instance.__class__.__name__
        instance.__class__ = type(base_cls_name, (base_cls, mixin), {})
        return instance


class IterableStream(Stream[T]):
    """A private class used the `Stream` class for creating data sources.

    Parameters
    ----------
    source : `Iterable[T]`
        The iterable to be used for providing the data.
    dtype : str, optional
        The data type of the source.
    """

    generic_name = "stream"

    def __init__(self, source: "Iterable[T]", dtype: str = None):
        super().__init__(dtype=dtype)
        self.is_gen = False
        self.iterable = None

        if inspect.isgeneratorfunction(source):
            self.gen_fn = source
            self.is_gen = True
            self.generator = self.gen_fn()
        else:
            self.iterable = source
            self.generator = iter(source)

        self.stop = False
        
        try:
            self.current = next(self.generator)
        except StopIteration:
            self.stop = True

    def forward(self) -> T:
        v = self.current
        try:
            self.current = next(self.generator)
        except StopIteration:
            self.stop = True
        return v

    def has_next(self):
        return not self.stop

    def reset(self):
        if self.is_gen:
            self.generator = self.gen_fn()
        else:
            self.generator = iter(self.iterable)
        self.stop = False

        try:
            self.current = next(self.generator)
        except StopIteration:
            self.stop = True
        super().reset()


class Group(Stream[T]):
    """A stream that groups together other streams into a dictionary."""

    def __init__(self):
        super().__init__()

    def __call__(self, *inputs):
        self.inputs = inputs
        self.streams = {s.name: s for s in inputs}
        return self

    def forward(self) -> "Dict[T]":
        return {s.name: s.value for s in self.inputs}

    def __getitem__(self, name) -> "Stream[T]":
        return self.streams[name]

    def has_next(self) -> bool:
        return True


class Sensor(Stream[T]):
    """A stream that watches and generates from a particular object."""

    generic_name = "sensor"

    def __init__(self, obj, func, dtype=None):
        super().__init__(dtype=dtype)
        self.obj = obj
        self.func = func

    def forward(self) -> T:
        return self.func(self.obj)

    def has_next(self):
        return True


class Constant(Stream[T]):
    """A stream that generates a constant value."""

    generic_name = "constant"

    def __init__(self, value, dtype: str = None):
        super().__init__(dtype=dtype)
        self.constant = value

    def forward(self):
        return self.constant

    def has_next(self):
        return True


class Placeholder(Stream[T]):
    """A stream that acts as a placeholder for data to be provided at later date.
    """

    generic_name = "placeholder"

    def __init__(self, dtype: str = None) -> None:
        super().__init__(dtype=dtype)

    def push(self, value: 'T') -> None:
        self.value = value

    def forward(self) -> 'T':
        return self.value

    def has_next(self) -> bool:
        return True

    def reset(self) -> None:
        self.value = None
