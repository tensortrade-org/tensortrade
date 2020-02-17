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
# limitations under the License


import random

from collections import namedtuple
from typing import List


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done'])


class ReplayMemory(object):

    def __init__(self, capacity: int, transition_type: namedtuple = Transition):
        self.capacity = capacity
        self.Transition = transition_type

        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(self.Transition(*args))
        else:
            self.memory[self.position] = self.Transition(*args)

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> List[namedtuple]:
        return random.sample(self.memory, batch_size)

    def head(self, batch_size) -> List[namedtuple]:
        return self.memory[:batch_size]

    def tail(self, batch_size) -> List[namedtuple]:
        return self.memory[-batch_size:]

    def __len__(self):
        return len(self.memory)
