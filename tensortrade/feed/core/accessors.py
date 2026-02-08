from typing import Generic, TypeVar, overload

_AccessorT = TypeVar("_AccessorT")


class CachedAccessor(Generic[_AccessorT]):
    """
    Custom property-like object.

    A descriptor for caching accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``df.foo``.
    accessor : cls
        Class with the extension methods.

    References
    ----------
    .. [1] https://github.com/pandas-dev/pandas/blob/v1.1.0/pandas/core/accessor.py#L285-L289
    """

    def __init__(self, name: str, accessor: type[_AccessorT]) -> None:
        self._name = name
        self._accessor = accessor

    @overload
    def __get__(self, instance: None, owner: type) -> type[_AccessorT]: ...

    @overload
    def __get__(self, instance: object, owner: type) -> _AccessorT: ...

    def __get__(self, instance, owner):
        if instance is None:
            return self._accessor
        accessor = self._accessor(instance)
        object.__setattr__(instance, self._name, accessor)
        return accessor
