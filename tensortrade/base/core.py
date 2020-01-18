import uuid

from abc import ABCMeta, abstractmethod
from typing import Dict

from .clock import Clock


objects = {}
global_clock = Clock()


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

    @property
    def clock(self):
        return self._clock

    @clock.setter
    def clock(self, clock: 'Clock'):
        self._clock = clock


class TimedIdentifiable(Identifiable, TimeIndexed, metaclass=ABCMeta):

    @property
    def created_at(self):
        if not hasattr(self, '__created_at'):
            self.__created_at = self.clock.step

        return self.__created_at


class Listener:
    pass


class Observable:

    def __init__(self):
        self._listeners = []

    def attach(self, listener: Listener):
        self._listeners += [listener]

    def detach(self, listener: Listener):
        self._listeners.remove(listener)
