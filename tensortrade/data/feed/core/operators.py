
import numpy as np

from tensortrade.data.feed.core.base import Stream, T

from typing import Callable


class Apply(Stream[T]):

    def __init__(self, func, dtype=None):
        super().__init__(dtype=dtype)
        self.func = func

    def forward(self):
        node = self.inputs[0]
        return self.func(node.value)

    def has_next(self):
        return True


class Lag(Stream[T]):

    generic_name = "lag"

    def __init__(self, lag: int = 1, dtype: str = None):
        super().__init__(dtype=dtype)
        self.lag = lag
        self.runs = 0
        self.history = []

    def forward(self):
        node = self.inputs[0]
        if self.runs < self.lag:
            self.runs += 1
            self.history.insert(0, node.value)
            return np.nan

        self.history.insert(0, node.value)
        return self.history.pop()

    def has_next(self):
        return True

    def reset(self):
        self.runs = 0
        self.history = []


class Accumulator(Stream[T]):

    def __init__(self, func, dtype=None):
        super().__init__(dtype)
        self.func = func
        self.past = None

    def forward(self):
        node = self.inputs[0]
        if self.past is None:
            self.past = node.value
            return self.past
        v = self.func(self.past, node.value)
        self.past = v
        return v

    def has_next(self):
        return True

    def reset(self):
        self.past = None


class Select(Stream[T]):

    generic_name = "select"

    def __init__(self, selector):
        if isinstance(selector, str):
            self.key = selector
            self.selector = lambda x: x.name == selector
        else:
            self.key = None
            self.selector = selector

        super().__init__(self.key)
        self._node = None

    def forward(self):
        if not self._node:
            self._node = list(filter(self.selector, self.inputs))[0]
            self.name = self._node.name
        return self._node.value

    def has_next(self):
        return True


class Copy(Stream[T]):

    generic_name = "copy"

    def __init__(self):
        super().__init__()

    def forward(self):
        return self.inputs[0].value

    def has_next(self):
        return True


class Freeze(Stream[T]):

    generic_name = "freeze"

    def __init__(self):
        super().__init__()
        self.freeze_value = None

    def forward(self):
        node = self.inputs[0]
        if not self.freeze_value:
            self.freeze_value = node.value
        return self.freeze_value

    def has_next(self):
        return True

    def reset(self):
        self.freeze_value = None


class BinOp(Stream[T]):

    generic_name = "bin_op"

    def __init__(self, op: Callable[[T, T], T], dtype=None):
        super().__init__(dtype=dtype)
        self.op = op

    def forward(self):
        return self.op(self.inputs[0].value, self.inputs[1].value)

    def has_next(self):
        return True
