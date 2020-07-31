"""This module hold the project level registry and provides methods to mutate
and change the registry.

Attributes
----------
MAJOR_COMPONENTS : List[str]
    The list of the major components that can be injected into.
"""

_REGISTRY = {}


MAJOR_COMPONENTS = [
    "actions",
    "rewards",
    "observer",
    "informer",
    "stopper",
    "renderer"
]


def registry() -> dict:
    """Gets the project level registry.

    Returns
    -------
    dict
        The project level registry.
    """
    return _REGISTRY


def register(component: 'Component', registered_name: str) -> None:
    """Registers a component into the registry

    Parameters
    ----------
    component : 'Component'
        The component to be registered.
    registered_name : str
        The name to be associated with the registered component.
    """
    _REGISTRY[component] = registered_name

