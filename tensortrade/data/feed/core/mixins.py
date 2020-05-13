

class DataTypeMixin:

    @classmethod
    def register_method(cls, func, names):
        def method(self, *args, **kwargs):
            args = (self,) + args
            return func(*args, **kwargs)

        for name in names:
            setattr(cls, name, method)
