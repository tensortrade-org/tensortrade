
import numpy as np

from tensortrade.feed.core.base import Stream, T


class ForwardFill(Stream[T]):

    generic_name = "ffill"

    def __init__(self):
        super().__init__()
        self.previous = None

    def forward(self):
        node = self.inputs[0]
        if not self.previous or np.isfinite(node.value):
            self.previous = node.value
        return self.previous

    def has_next(self):
        return True


class FillNa(Stream[T]):

    generic_name = "fillna"

    def __init__(self, fill_value: T):
        super().__init__()
        self.fill_value = fill_value

    def forward(self):
        node = self.inputs[0]
        if np.isnan(node.value):
            return self.fill_value
        return node.value

    def has_next(self):
        return True

