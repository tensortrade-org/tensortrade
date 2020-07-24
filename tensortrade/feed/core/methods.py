

class Methods:

    def __init__(self, stream):
        self.stream = stream

    @classmethod
    def _make_accessor(cls, stream):
        return cls(stream)

    @classmethod
    def register_method(cls, func, names):
        def method(self, *args, **kwargs):
            args = (self.stream,) + args
            return func(*args, **kwargs)
        for name in names:
            setattr(cls, name, method)
        return method
