import numpy as np
import math


class Welfords(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    Includes ability to compute sample variance over a window of values.
    """

    def __init__(self, iterable=None, ddof=0, window_size=None):
        self.ddof, self.n, self.mean, self.M2, self.window_size = ddof, 0, 0.0, 0.0, window_size
        if iterable is not None:
            for datum in iterable:
                self.include(datum)
        self.window = []

    def include(self, datum):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += (datum - self.mean) * self.delta

        if self.window_size:
            self.window.append(datum)
            if len(self.window) > self.window_size:
                self.exclude(self.window[0])
                del self.window[0]

    def exclude(self, datum):
        self.n -= 1
        self.delta = datum - self.mean
        self.mean -= self.delta / self.n
        self.M2 -= (datum - self.mean) * self.delta

    @property
    def variance(self):
        return self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)


class PctChange(object):
    def __init__(self):
        self.previous = np.nan

    def pct_change(self, datum):
        result = np.nan
        if not math.isnan(self.previous):
            result = (datum - self.previous)/self.previous

        self.previous = datum
        return result
