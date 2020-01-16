

from typing import Dict, List

from tensortrade.data.source import DataSource


class DataFeed(DataSource):

    def __init__(self, sources: List[DataSource]):
        super().__init__()
        self._sources = sources

    def next(self) -> Dict[str, any]:
        data = {}
        for ds in self._sources:
            data.update(ds.next())
        return data

    def has_next(self) -> bool:
        for ds in self._sources:
            if not ds.has_next():
                return False
        return True

    def reset(self):
        for ds in self._sources:
            ds.reset()
