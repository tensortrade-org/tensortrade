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

from .node import Node


class BinOp(Node):

    def __init__(self, name: str, op):
        super().__init__(name)
        self.op = op

    def forward(self, inbound_data: dict):
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

    def forward(self, inbound_data):
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

    def forward(self, inbound_data):
        if not self.key:
            self.key = list(filter(self.selector, inbound_data.keys()))[0]
            self.name = self.key
        return inbound_data[self.key]

    def has_next(self):
        return True

    def reset(self):
        pass


class Namespace(Node):

    def __init__(self, name: str):
        super().__init__(name)
        self.flatten = True

    def forward(self, inbound_data: dict):
        return inbound_data

    def has_next(self):
        return True

    def reset(self):
        pass


class Lambda(Node):

    def __init__(self, name: str, extract: Callable[[any], float], obj: any):
        super().__init__(name)

        self.extract = extract
        self.obj = obj

    def forward(self, inbound_data: dict):
        return self.extract(self.obj)

    def has_next(self):
        return True

    def reset(self):
        pass
