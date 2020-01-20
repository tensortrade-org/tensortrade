
import operator
import functools

from abc import abstractmethod
from typing import Union, Callable, List

from .node import Node


class BinOp(Node):

    def __init__(self, name: str, op):
        super().__init__(name)
        self.op = op

    def call(self, inbound_data: dict):
        left_name = self.inbound[0].name
        right_name = self.inbound[1].name
        left_value = inbound_data.get(left_name, 0)
        right_value = inbound_data.get(right_name, 0)
        return self.op(left_value, right_value)

    def has_next(self):
        return True

    def reset(self):
        pass


class Reduce(Node):

    def __init__(self,
                 name: str,
                 selector: Callable[[str], bool],
                 func: Callable[[float, float], float]):
        super().__init__(name)
        self.selector = selector
        self.func = func

    def call(self, inbound_data):
        keys = list(filter(self.selector, inbound_data.keys()))
        return functools.reduce(self.func, [inbound_data[k] for k in keys])

    def has_next(self):
        return True

    def reset(self):
        pass


class Select(Node):

    def __init__(self, selector: Union[Callable[[str], bool], str]):
        if isinstance(selector, str):
            name = selector
            self.key = name
        else:
            self.key = None
            self.selector = selector
        super().__init__(self.key or "select")

        self.flatten = True

    def call(self, inbound_data):
        if not self.key:
            self.key = list(filter(self.selector, inbound_data.keys()))[0]
        return inbound_data[self.key]

    def has_next(self):
        return True

    def reset(self):
        pass


class Namespace(Node):

    def __init__(self, name: str):
        super().__init__(name)
        self.flatten = True

    def call(self, inbound_data: dict):
        return inbound_data

    def has_next(self):
        return True

    def reset(self):
        pass
