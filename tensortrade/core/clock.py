from datetime import datetime


class Clock(object):
    """A class to track the time for a process.

    Attributes
    ----------
    start : int
        The time of start for the clock.
    step : int
        The time of the process the clock is at currently.

    Methods
    -------
    now(format=None)
        Gets the current time in the provided format.
    increment()
        Increments the clock by specified time increment.
    reset()
        Resets the clock.
    """

    def __init__(self):
        self.start = 0
        self.step = self.start

    def now(self, format: str = None) -> datetime:
        """Gets the current time in the provided format.
        Parameters
        ----------
        format : str or None, optional
            The format to put the current time into.

        Returns
        -------
        datetime
            The current time.
        """
        return datetime.now().strftime(format) if format else datetime.now()

    def increment(self) -> None:
        """Increments the clock by specified time increment."""
        self.step += 1

    def reset(self) -> None:
        """Resets the clock."""
        self.step = self.start
