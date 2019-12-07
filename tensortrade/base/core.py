import uuid


objects = {}


class Identifiable:

    def __init__(self):
        self._id = uuid.uuid4()
        objects[self.id] = self

    @property
    def id(self):
        return self._id


class TimeIndexed:

    now = None

    def reset(self):
        pass
