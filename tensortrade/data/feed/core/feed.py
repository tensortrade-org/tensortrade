

from tensortrade.data.feed.core.base import Stream, T


class DataFeed(Stream[T]):

    def __init__(self, streams=None):
        super().__init__()

        self.process = None
        self.compiled = False

        if streams:
            self.__call__(*streams)

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

    def reset(self) -> None:
        for node in self.process:
            node.reset()
