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


from typing import List
from tensortrade.data.stream.node import Node


class Stream(Node):

    def __init__(self, array: List[any] = None, name: str = None):
        super().__init__(name)
        self._array = array if array else []
        self._cursor = 0

    @property
    def generic_name(self) -> str:
        return "stream"

    def forward(self):
        v = self._array[self._cursor]
        self._cursor += 1
        return v

    def has_next(self) -> bool:
        if self._cursor < len(self._array):
            return True
        return False

    def reset(self):
        self._cursor = 0
