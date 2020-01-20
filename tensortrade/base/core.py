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


class Observable:

    @abstractmethod
    def observe(self) -> Dict[str, float]:
        raise NotImplementedError


class TimeIndexed:

    @property
    def clock(self):
        return self._clock

    @clock.setter
    def clock(self, clock: 'Clock'):
        self._clock = clock


class TimedIdentifiable(Identifiable, TimeIndexed, metaclass=ABCMeta):

    def __init__(self):
        self.__created_at = global_clock.now()

    @property
    def created_at(self):
        return self.__created_at
