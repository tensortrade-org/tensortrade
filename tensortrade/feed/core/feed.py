

from typing import List

from tensortrade.feed.core.base import Stream, T


class DataFeed(Stream[T]):
    """A stream the compiles together streams to be run in an organized manner.

    Parameters
    ----------
    streams : `List[Stream]`
        A list of streams to be used in the data feed.
    """

    def __init__(self, streams: "List[Stream]") -> None:
        super().__init__()

        self.process = None
        self.compiled = False

        if streams:
            self.__call__(*streams)

    def compile(self) -> None:
        """Compiles all the given stream together.

        Organizes the order in which streams should be run to get valid output.
        """
        edges = self.gather()

        self.process = self.toposort(edges)
        self.compiled = True
        self.reset()

    def run(self) -> None:
        """Runs all the streams in processing order."""
        if not self.compiled:
            self.compile()

        for s in self.process:
            s.run()

        super().run()

    def forward(self) -> dict:
        return {s.name: s.value for s in self.inputs}

    def next(self) -> dict:
        self.run()
        return self.value

    def has_next(self) -> bool:
        return all(s.has_next() for s in self.process)

    def reset(self) -> None:
        for s in self.process:
            s.reset()
