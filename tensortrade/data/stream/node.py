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


from abc import abstractmethod
from typing import List


class Node:

    def __init__(self, name: str, sep: str = ":/"):
        self._name = name
        self._inputs = []
        self._inbound = []
        self._outbound = []
        self._inbound_data = {}
        self._call_count = 0
        self._data = {}
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

    def gather(self) -> List['Node']:
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
            pass

        self._call_count = 0

        outbound_data = {self.name: self.generate(self._inbound_data)}

        self.propagate(outbound_data)

        return outbound_data

    def __call__(self, *inbound):
        self.inbound = list(inbound)

        for node in self.inbound:
            node.subscribe(self)

        return self

    @abstractmethod
    def generate(self, inbound_data: dict):
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

    def generate(self, inbound_data: dict):
        return self.value

    def has_next(self):
        return True

    def reset(self):
        pass


class Placeholder(Node):

    def __init__(self, name: str):
        super().__init__(name)

        self.name = name

    def generate(self, inbound_data: dict):
        return inbound_data[self.name]

    def has_next(self):
        return True

    def reset(self):
        pass
