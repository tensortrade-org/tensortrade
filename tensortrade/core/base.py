"""Responsible for the basic classes in the project.

Attributes
----------
global_clock : `Clock`
    A clock that provides a global reference for all objects that share a
    timeline.
"""

import uuid

from abc import ABCMeta

from tensortrade.core.clock import Clock


global_clock = Clock()


class Identifiable(object, metaclass=ABCMeta):
    """Identifiable mixin for adding a unique `id` property to instances of a class.
    """

    @property
    def id(self) -> str:
        """Gets the identifier for the object.

        Returns
        -------
        str
           The identifier for the object.
        """
        if not hasattr(self, '_id'):
            self._id = str(uuid.uuid4())
        return self._id

    @id.setter
    def id(self, identifier: str) -> None:
        """Sets the identifier for the object

        Parameters
        ----------
        identifier : str
            The identifier to set for the object.
        """
        self._id = identifier


class TimeIndexed:
    """A class for objects that are indexed by time.
    """

    _clock = global_clock

    @property
    def clock(self) -> Clock:
        """Gets the clock associated with this object.

        Returns
        -------
        `Clock`
            The clock associated with this object.
        """
        return self._clock

    @clock.setter
    def clock(self, clock: Clock) -> None:
        """Sets the clock associated with this object.

        Parameters
        ----------
        clock : `Clock`
            The clock to be associated with this object.
        """
        self._clock = clock


class TimedIdentifiable(Identifiable, TimeIndexed, metaclass=ABCMeta):
    """A class an identifiable object embedded in a time process.

    Attributes
    ----------
    created_at : `datetime.datetime`
        The time at which this object was created according to its associated
        clock.
    """

    def __init__(self) -> None:
        self.created_at = self._clock.now()

    @property
    def clock(self) -> "Clock":
        """Gets the clock associated with the object.

        Returns
        -------
        `Clock`
            The clock associated with the object.
        """
        return self._clock

    @clock.setter
    def clock(self, clock: "Clock") -> None:
        """Sets the clock associated with this object.

        In addition, the `created_at` attribute is set according to the new clock.

        Parameters
        ----------
        clock : `Clock`
            The clock to be associated with this object.
        """
        self._clock = clock
        self.created_at = self._clock.now()


class Observable:
    """An object with some value that can be observed.

    An object to which a `listener` can be attached to and be alerted about on
    an event happening.

    Attributes
    ----------
    listeners : list of listeners
        A list of listeners that the object will alert on events occurring.

    Methods
    -------
    attach(listener)
        Adds a listener to receive alerts.
    detach(listener)
        Removes a listener from receiving alerts.
    """

    def __init__(self):
        self.listeners = []

    def attach(self, listener) -> "Observable":
        """Adds a listener to receive alerts.

        Parameters
        ----------
        listener : a listener object

        Returns
        -------
        `Observable` :
            The observable being called.
        """
        self.listeners += [listener]
        return self

    def detach(self, listener) -> "Observable":
        """Removes a listener from receiving alerts.

        Parameters
        ----------
        listener : a listener object

        Returns
        -------
        `Observable`
            The observable being called.
        """
        self.listeners.remove(listener)
        return self
