import uuid

from abc import ABCMeta
from .clock import BasicClock


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

    clock = BasicClock()


class Basic(Identifiable, TimeIndexed, metaclass=ABCMeta):

    @property
    def created_at(self):
        if not hasattr(self, '__created_at'):
            self.__created_at = self.clock.now()
        return self.__created_at