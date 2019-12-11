

class BasicClock(object):

    def __init__(self):
        self._start = 0
        self._now = self._start

    @property
    def start(self):
        return self._start

    def now(self):
        return self._now

    def increment(self):
        self._now += 1

    def reset(self):
        self._now = self._start




