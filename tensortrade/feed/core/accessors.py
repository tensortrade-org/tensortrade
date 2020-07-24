

class CachedAccessor:
    """
    Custom property-like object.
    A descriptor for caching accessors.

    Parameters:
    ==========
        name : str
            Namespace that will be accessed under.
        accessor : cls
            Class with the extension methods.
    """

    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, instance, owner):
        if instance is None:
            return self._accessor
        accessor = self._accessor(instance)
        object.__setattr__(instance, self._name, accessor)
        return accessor
