
import collections
import pandas as pd

from typing import List

from tensortrade.base.core import Observable
from tensortrade.data.stream.node import Node
from tensortrade.data import Array


def _flatten(data, parent_key='', sep=':/'):
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
        self._names = None
        self._nodes = self.remove_duplicates(nodes)
        self._inputs = self.gather(self._nodes)
        self._data = {}
        self._flatten = flatten
        self.reset()

    @property
    def names(self):
        if self._names:
            return self._names
        names = []
        for name in map(lambda n: n.name, self._nodes):
            for k in self._data.keys():
                if k.startswith(name):
                    names += [k]
        self._names = names
        return self._names

    @property
    def inputs(self) -> List[Node]:
        return self._inputs

    @staticmethod
    def remove_duplicates(nodes: List['Node']) -> List['Node']:
        node_list = []
        for node in nodes:
            if node not in node_list:
                node_list += [node]
        return node_list

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
            self._data.update(outbound_data)
            for output_node in node.outbound:
                self._next(output_node)

    def next(self):
        self._data = {}
        for node in self.inputs:
            self._next(node)
        feed_data = {name: self._data[name] for name in self.names}
        for listener in self.listeners:
            listener.on_next(feed_data)
        return feed_data

    def has_next(self) -> bool:
        for node in self.inputs:
            if not node.has_next():
                return False
        return True

    def __add__(self, other):
        if isinstance(other, DataFeed):
            feed = DataFeed(self._nodes + other._nodes)
            listeners = self.listeners + other.listeners
            for listener in listeners:
                feed.attach(listener)
            return feed

    def reset(self):
        for node in self.inputs:
            node.refresh()

    @staticmethod
    def from_frame(frame: pd.DataFrame) -> 'DataFeed':
        nodes = []
        for name in frame.columns:
            nodes += [Array(name, list(frame[name]))]
        return DataFeed(nodes)
