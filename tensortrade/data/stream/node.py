
import collections

from abc import abstractmethod
from typing import List


def _flatten(data, parent_key='', sep=':/'):
    items = []
    for k, v in data.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Node:

    def __init__(self, name: str, sep: str = ":/"):
        self._name = name
        self._inputs = []
        self._inbound = []
        self._outbound = []
        self._inbound_data = {}
        self._call_count = 0
        self._data = {}
        self.flatten = False
        self._sep = sep

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def inbound(self):
        return self._inbound

    @inbound.setter
    def inbound(self, inbound: List['Node']):
        self._inbound = inbound

    @property
    def outbound(self):
        return self._outbound

    @outbound.setter
    def outbound(self, outbound: List['Node']):
        self._outbound = outbound

    @property
    def flatten(self) -> bool:
        return self._flatten

    @flatten.setter
    def flatten(self, flatten: bool):
        self._flatten = flatten

    def gather(self) -> List['Node']:
        """
        for node in self.inbound:
            if len(node.inbound) == 0:
                starting += [node]
            else:
                starting += node.gather()
        """
        if len(self.inbound) == 0:
            return [self]

        starting = []
        for node in self.inbound:
            starting += node.gather()
        return starting

    def subscribe(self, node: 'Node'):
        self.outbound += [node]

    def push(self, inbound_data: dict):
        self._inbound_data.update(inbound_data)

    def propagate(self, data: dict):
        for node in self.outbound:
            node.push(data)

    def next(self):
        self._call_count += 1
        if self._call_count < len(self.inbound):
            return
        self._call_count = 0
        data = {self.name: self.call(self._inbound_data)}
        outbound_data = _flatten(data, sep=self._sep) if self.flatten else data
        self.propagate(outbound_data)
        return outbound_data

    def __call__(self, *inbound):
        self.inbound = list(inbound)
        for node in self.inbound:
            node.subscribe(self)
        return self

    @abstractmethod
    def call(self, inbound_data: dict):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def has_next(self):
        raise NotImplementedError()

    def refresh(self):
        self.reset()
        for source in self.outbound:
            source.refresh()


class Value(Node):

    def __init__(self, name: str, value: float):
        super().__init__(name)
        self.value = value

    def call(self, inbound_data: dict):
        return self.value

    def has_next(self):
        return True

    def reset(self):
        pass


class Placeholder(Node):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name

    def call(self, inbound_data: dict):
        return inbound_data[self.name]

    def has_next(self):
        return True

    def reset(self):
        pass