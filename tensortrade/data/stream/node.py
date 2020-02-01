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

"""
References:
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/module/module.py
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/base_layer.py
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/node.py
"""

from abc import abstractmethod
from tensortrade.base.core import Observable


class Node(Observable):

    def __init__(self, name: str):
        super().__init__()

        self._name = name
        self.inputs = []

        if len(Module.CONTEXTS) > 0:
            Module.CONTEXTS[-1].add_node(self)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: float):
        self._value = value

    def __call__(self, *inputs):
        self.inputs = []

        for node in inputs:
            if isinstance(node, Module):
                if not node.built:
                    with node:
                        node.build()

                    node.built = True

                self.inputs += node.flatten()
            else:
                self.inputs += [node]

        return self

    def run(self):
        self.value = self.forward()

    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()

    @abstractmethod
    def has_next(self):
        raise NotImplementedError()

    def __str__(self):
        return "<Node: name={}, type={}>".format(self.name,
                                                 str(self.__class__.__name__).lower())

    def __repr__(self):
        return str(self)


class Module(Node):

    CONTEXTS = []

    def __init__(self, name: str):
        super().__init__(name)

        self.submodules = []
        self.variables = []
        self.built = False

    def add_node(self, node: 'Node'):
        node.name = self.name + ":/" + node.name

        if isinstance(node, Module):
            self.submodules += [node]
        else:
            self.variables += [node]

    def build(self):
        pass

    def flatten(self):
        nodes = [node for node in self.variables]

        for module in self.submodules:
            nodes += module.flatten()

        return nodes

    def __enter__(self):
        self.CONTEXTS += [self]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.CONTEXTS.pop()
        return self

    def forward(self):
        return

    def reset(self):
        pass
