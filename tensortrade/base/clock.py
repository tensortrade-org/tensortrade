

class Clock(object):

    def __init__(self):
        self._start = 0
        self._step = self._start

    @property
    def start(self):
        return self._start

    @property
    def step(self):
        return self._step

    def increment(self):
        self._step += 1

    def reset(self):
        self._step = self._start
