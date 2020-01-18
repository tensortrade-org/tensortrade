
import collections

from abc import abstractmethod
from typing import List




class Node:

    def __init__(self, name: str):
        self._name = name
        self._inbound = []
        self._outbound = []
        self._inbound_data = {}
        self._call_count = 0
        self._flatten = False

    @property
    def name(self):
        return self._name

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

    def subscribe(self, source):
        self.outbound += [source]

    def push(self, inbound_data: dict):
        self._inbound_data.update(inbound_data)

    def propagate(self, outbound_data: dict):
        data = {self.name: outbound_data}
        for node in self.outbound:
            node.push(data)

    def next(self):
        self._call_count += 1
        if self._call_count < len(self.inbound):
            return
        self._call_count = 0
        outbound_data = self.call(self._inbound_data)
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
