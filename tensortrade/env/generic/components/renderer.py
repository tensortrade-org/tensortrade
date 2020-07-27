# Copyright 2020 The TensorTrade Authors.
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

from abc import ABCMeta, abstractmethod

from tensortrade.core import Component


class Renderer(Component, metaclass=ABCMeta):

    registered_name = "renderer"

    @abstractmethod
    def render(self, env: 'TradingEnv'):
        raise NotImplementedError()

    def save(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass
