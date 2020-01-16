

from typing import Dict, List

from tensortrade.data.source import DataSource


class DataFeed:

    def __init__(self,
                 inputs: List[DataSource] = None,
                 outputs: List[DataSource] = None):
        super().__init__()
        self._inputs = inputs if inputs else []
        self._outputs = outputs if outputs else []

        self._both = []
        self._inputs_only = []
        self._outputs_only = []
        for source in self._inputs:
            if source in self._outputs:
                self._both += [source]
            else:
                self._inputs_only += [source]

        for source in self._outputs:
            if source not in self._inputs:
                self._outputs_only += [source]

    def next(self) -> Dict[str, any]:
        if len(self._outputs) != 0:

            for source in self._inputs_only:
                source.next()

            feed_data = {}
            for source in self._both:
                data = source.next()
                for k in data.keys():
                    if k in feed_data.keys():
                        data[source.name + k] = data[k]
                        del data[k]
                feed_data.update(data)

            for source in self._outputs_only:
                data = source.next()
                for k in data.keys():
                    if k in feed_data.keys():
                        data[source.name + k] = data[k]
                        del data[k]
                feed_data.update(data)

            return feed_data

        feed_data = {}
        for source in self._inputs:
            data = source.next()
            for k in data.keys():
                if k in feed_data.keys():
                    data['{' + source.name + '}' + ":" + str(k)] = data[k]
                    del data[k]
            feed_data.update(data)
        return feed_data

    def has_next(self) -> bool:
        for ds in self._inputs:
            if not ds.has_next():
                return False
        return True

    def reset(self):
        for ds in self._inputs:
            ds.refresh()
