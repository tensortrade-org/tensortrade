
import operator

from abc import abstractmethod

from .node import Node


class Transform(Node):

    def __init__(self, name: str):
        super().__init__(name)

    def call(self, inbound_data):
        return self.transform(inbound_data)

    @abstractmethod
    def transform(self, inbound_data):
        raise NotImplementedError()

    def reset(self):
        pass


class BinOp(Transform):

    def __init__(self, name: str, op):
        super().__init__(name)
        self.op = op

    def transform(self, inbound_data: dict):
        left_name = self.inbound[0].name
        right_name = self.inbound[1].name
        left_value = inbound_data.get(left_name, 0)
        right_value = inbound_data.get(right_name, 0)
        return self.op(left_value, right_value)


class Select(Transform):

    def __init__(self, name: str, key: str):
        super().__init__(name)
        self.key = key

    def transform(self, inbound_data):
        return inbound_data[key]