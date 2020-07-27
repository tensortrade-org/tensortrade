import uuid

from abc import ABCMeta, abstractmethod
from typing import Dict

from .clock import Clock


global_clock = Clock()
objects = {}


class Identifiable(object, metaclass=ABCMeta):
    """Identifiable mixin for adding a unique `id` property to instances of a class."""

    @property
    def id(self) -> str:
        if not hasattr(self, '_id'):
            self._id = str(uuid.uuid4())
            objects[self._id] = self
        return self._id

    @id.setter
    def id(self, identifier: str):
        objects[identifier] = self
        self._id = identifier


class TimeIndexed:

    _clock = global_clock

    @property
    def clock(self) -> Clock:
        return self._clock

    @clock.setter
    def clock(self, clock: Clock):
        self._clock = clock


class TimedIdentifiable(Identifiable, TimeIndexed, metaclass=ABCMeta):

    def __init__(self):
        self.__created_at = self._clock.now()

    @property
    def clock(self) -> Clock:
        return self._clock

    @clock.setter
    def clock(self, clock: Clock):
        self._clock = clock
        self.__created_at = self._clock.now()

    @property
    def created_at(self):
        return self.__created_at


class Listener:
    pass


class Observable:

    def __init__(self):
        self._listeners = []

    @property
    def listeners(self):
        return self._listeners

    def attach(self, listener: Listener):
        self._listeners += [listener]
        return self

    def detach(self, listener: Listener):
        self._listeners.remove(listener)
        return self
