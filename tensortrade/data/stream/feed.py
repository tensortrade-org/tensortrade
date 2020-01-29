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


from tensortrade.data.stream import Node


class DataFeed(Node):

    def __init__(self):
        super().__init__("")
        self.process = None
        self.compiled = False

    @staticmethod
    def _gather(node, vertices, edges):
        if node not in vertices:
            vertices += [node]
            for input_node in node.inputs:
                edges += [(input_node, node)]
            for input_node in node.inputs:
                DataFeed._gather(input_node, vertices, edges)
        return edges

    def gather(self):
        return self._gather(self, [], [])

    @staticmethod
    def toposort(edges):
        S = set([s for s, t in edges])
        T = set([t for s, t in edges])

        starting = list(S.difference(T))
        process = starting.copy()
        while len(starting) > 0:
            start = starting.pop()

            edges = list(filter(lambda e: e[0] != start, edges))

            S = set([s for s, t in edges])
            T = set([t for s, t in edges])

            starting += [v for v in S.difference(T) if v not in starting]

            if start not in process:
                process += [start]
        return process

    def compile(self):
        edges = self.gather()
        self.process = self.toposort(edges)

        self.compiled = True
        self.reset()

    def run(self):
        if not self.compiled:
            self.compile()
        for node in self.process:
            node.run()
        super().run()

    def forward(self):
        return {node.name: node.value for node in self.inputs}

    def next(self):
        self.run()

        for listener in self.listeners:
            listener.on_next(self.value)

        return self.value

    def has_next(self) -> bool:
        return all(node.has_next() for node in self.process)

    def __add__(self, other):
        if isinstance(other, DataFeed):
            nodes = list(set(self.inputs + other.inputs))
            feed = DataFeed()(*nodes)
            for listener in self.listeners + other.listeners:
                feed.attach(listener)
            return feed

    def reset(self):
        for node in self.process:
            node.reset()
