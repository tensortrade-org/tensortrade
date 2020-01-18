
import collections

from typing import List

from tensortrade.base.core import Observable
from tensortrade.data.stream.node import Node


def _flatten(data, parent_key='', sep=':'):
    items = []
    for k, v in data.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class DataFeed(Observable):

    def __init__(self, nodes: List[Node], flatten: bool = True):
        super().__init__()
        self._names = [node.name for node in nodes]
        self._inputs = self.gather(nodes)
        self._data = {}
        self._flatten = flatten
        self.reset()

    @property
    def names(self):
        return self._names

    @property
    def inputs(self) -> List[Node]:
        return self._inputs

    @staticmethod
    def gather(nodes: List[Node]):
        starting = []
        for node in nodes:
            for start_node in node.gather():
                if start_node not in starting:
                    starting += [start_node]
        return starting

    def _next(self, node: Node):
        outbound_data = node.next()
        if outbound_data:
            self._data[node.name] = outbound_data
            for output_node in node.outbound:
                self._next(output_node)

    def next(self):
        self._data = {}
        for node in self.inputs:
            self._next(node)
        data = {name: self._data[name] for name in self.names}
        return _flatten(data) if self._flatten else data

    def has_next(self) -> bool:
        for node in self.inputs:
            if not node.has_next():
                return False
        return True

    def reset(self):
        for node in self.inputs:
            node.refresh()
