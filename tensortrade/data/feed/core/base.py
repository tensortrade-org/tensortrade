
from abc import abstractmethod

from typing import (
    Generic,
    Iterable,
    TypeVar,
    Dict,
    Any,
    Callable,
    List
)

from tensortrade.base import Observable
from tensortrade.data.feed.core.accessors import CachedAccessor


T = TypeVar("T")


class Named:

    generic_name = "generic"
    namespaces = []
    names = {}

    def __init__(self, name=None):
        if not name:
            name = self.generic_name

            if name in Stream.names.keys():
                Stream.names[name] += 1
                name += ":/" + str(Stream.names[name] - 1)
            else:
                Stream.names[name] = 0
        self.name = name

    def rename(self, name: str, sep: str = ":/") -> "Named":
        if len(Named.namespaces) > 0:
            name = Named.namespaces[-1] + sep + name
        self.name = name
        return self


class NameSpace(Named):

    def __init__(self, name):
        super().__init__(name)

    def __enter__(self):
        Named.namespaces += [self.name]

    def __exit__(self, exc_type, exc_val, exc_tb):
        Named.namespaces.pop()


class Stream(Generic[T], Named, Observable):

    _methods = {}
    _generic_methods = []
    _accessors = []
    generic_name = "stream"

    def __new__(cls, *args, **kwargs):
        dtype = kwargs.get("dtype")
        instance = super().__new__(cls, *args, **kwargs)
        if dtype in Stream._methods.keys():
            mixin = Stream._methods[dtype]
            instance = Stream.extend_instance(instance, mixin)
        return instance

    def __init__(self, name=None, dtype=None):
        Named.__init__(self, name)
        Observable.__init__(self)
        self.dtype = dtype
        self.inputs = []
        self.value = None

    def __call__(self, *inputs):
        self.inputs = inputs
        return self

    def run(self) -> None:
        self.value = self.forward()
        for listener in self.listeners:
            listener.on_next(self.value)

    @abstractmethod
    def forward(self) -> T:
        raise NotImplementedError()

    @abstractmethod
    def has_next(self) -> bool:
        raise NotImplementedError()

    def astype(self, dtype) -> "Stream[T]":
        mixin = Stream._methods[dtype]
        return Stream.extend_instance(self, mixin)

    def reset(self) -> None:
        for listener in self.listeners:
            if hasattr(listener, "reset"):
                listener.reset()

    def gather(self):
        return self._gather(self, [], [])

    @staticmethod
    def source(iterable: "Iterable[T]", dtype=None) -> "Stream[T]":
        return _Stream(iterable, dtype=dtype)

    @staticmethod
    def group(streams: "List[Stream[T]]") -> "Stream[T]":
        return Group()(*streams)

    @staticmethod
    def sensor(obj: "Any", func: "Callable[[Any], T]", dtype=None) -> "Stream[T]":
        return Sensor(obj, func, dtype=dtype)

    @staticmethod
    def select(streams: "List[Stream[T]]", func: "Callable[[Stream[T]], bool]") -> "Stream[T]":
        for s in streams:
            if func(s):
                return s
        raise Exception("No stream satisfies selector condition.")

    @staticmethod
    def constant(value: "Any", dtype=None) -> "Stream[T]":
        return Constant(value, dtype=dtype)

    @staticmethod
    def _gather(node, vertices, edges):
        if node not in vertices:
            vertices += [node]

            for input_node in node.inputs:
                edges += [(input_node, node)]

            for input_node in node.inputs:
                Stream._gather(input_node, vertices, edges)

        return edges

    @staticmethod
    def toposort(edges):
        source = set([s for s, t in edges])
        target = set([t for s, t in edges])

        starting = list(source.difference(target))
        process = starting.copy()

        while len(starting) > 0:
            start = starting.pop()

            edges = list(filter(lambda e: e[0] != start, edges))

            source = set([s for s, t in edges])
            target = set([t for s, t in edges])

            starting += [v for v in source.difference(target) if v not in starting]

            if start not in process:
                process += [start]

        return process

    @classmethod
    def register_accessor(cls, name):
        def wrapper(accessor):
            setattr(cls, name, CachedAccessor(name, accessor))
            cls._accessors += [name]
            return accessor
        return wrapper

    @classmethod
    def register_mixin(cls, dtype):
        def wrapper(mixin):
            cls._methods[dtype] = mixin
            return mixin
        return wrapper

    @classmethod
    def register_generic_method(cls, names):
        def wrapper(func):
            def method(self, *args, **kwargs):
                args = (self,) + args
                return func(*args, **kwargs)
            for name in names:
                setattr(Stream, name, method)
            return method
        return wrapper

    @staticmethod
    def extend_instance(instance, mixin):
        """Apply mix-ins to a class instance after creation"""
        base_cls = instance.__class__
        base_cls_name = instance.__class__.__name__
        instance.__class__ = type(base_cls_name, (base_cls, mixin), {})
        return instance


class _Stream(Stream[T]):

    generic_name = "stream"

    def __init__(self, iterable: "Iterable[T]", dtype=None):
        super().__init__(dtype=dtype)
        self.iterable = iterable
        self.generator = iter(iterable)

        self.stop = False

        try:
            self.current = next(self.generator)
        except StopIteration:
            self.stop = True

    def forward(self):
        v = self.current
        try:
            self.current = next(self.generator)
        except StopIteration:
            self.stop = True
        return v

    def has_next(self):
        return not self.stop

    def reset(self):
        self.generator = iter(self.iterable)
        self.stop = False

        try:
            self.current = next(self.generator)
        except StopIteration:
            self.stop = True


class Group(Stream[T]):

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

    generic_name = "constant"

    def __init__(self, value, dtype=None):
        super().__init__(dtype=dtype)
        self.constant = value

    def forward(self):
        return self.constant

    def has_next(self):
        return True
