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


from tensortrade.orders.criteria import Criteria


class Timed(Criteria):

    def __init__(self, duration):
        self.duration = duration

    def call(self, order: 'Order', exchange: 'Exchange'):
        return (order.clock.step - order.created_at) <= self.duration

    def __str__(self):
        return "<Timed: duration={}>".format(self.duration)

