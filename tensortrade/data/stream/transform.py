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


import functools

from typing import Union, Callable

from .node import Node, Module


class BinOp(Node):

    def __init__(self, name: str, op):
        super().__init__(name)
        self.op = op

    def forward(self):
        return self.op(self.inputs[0].value, self.inputs[1].value)

    def has_next(self):
        return True

    def reset(self):
        pass


class Reduce(Node):

    def __init__(self,
                 name: str,
                 func: Callable[[float, float], float]):
        super().__init__(name)
        self.func = func

    def forward(self):
        return functools.reduce(self.func, [node.value for node in self.inputs])

    def has_next(self):
        return True

    def reset(self):
        pass


class Select(Node):

    def __init__(self, selector: Union[Callable[[str], bool], str]):
        if isinstance(selector, str):
            self.key = selector
            self.selector = lambda x: x.name == selector
        else:
            self.key = None
            self.selector = selector
        super().__init__(self.key or "select")
        self._node = None

    def forward(self):
        if not self._node:
            self._node = list(filter(self.selector, self.inputs))[0]
            self.name = self._node.name
        return self._node.value

    def has_next(self):
        return True

    def reset(self):
        pass


class Lambda(Node):

    def __init__(self, name: str, extract: Callable[[any], float], obj: any):
        super().__init__(name)
        self.extract = extract
        self.obj = obj

    def forward(self):
        return self.extract(self.obj)

    def has_next(self):
        return True

    def reset(self):
        pass


class Forward(Lambda):

    def __init__(self, node: 'Node'):
        super().__init__(
            name=node.name,
            extract=lambda x: x.value,
            obj=node
        )
        self(node)


class Condition(Module):

    def __init__(self, name: str, condition: Callable[['Node'], bool]):
        super().__init__(name)
        self.condition = condition

    def build(self):
        self.variables = list(filter(self.condition, self.inputs))

    def has_next(self):
        return True