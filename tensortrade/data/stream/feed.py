# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd

from typing import List

from tensortrade.base.core import Observable
from tensortrade.data.stream import Stream, Node


class DataFeed(Observable):

    def __init__(self, nodes: List[Node]):
        super().__init__()

        self._names = None
        self._nodes = self.remove_duplicates(nodes)
        self._inputs = self.gather(self._nodes)
        self._data = {}
        self.reset()

    @property
    def names(self):
        if self._names:
            return self._names

        self._names = []
        for name in map(lambda n: n.name, self._nodes):
            for k in self._data.keys():
                if k.startswith(name):
                    self._names += [k]
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
            nodes += [Stream(name, list(frame[name]))]

        return DataFeed(nodes)
