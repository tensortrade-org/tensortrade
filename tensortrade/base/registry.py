
import numpy as np


_REGISTRY = {}


MAJOR_COMPONENTS = [
    'exchanges',
    'actions',
    'rewards',
    'features',
    'slippage'
]


def get_major_component_names():
    return MAJOR_COMPONENTS


def get_registry():
    return _REGISTRY


def registered_names():
    return list(np.unique([_REGISTRY[i] for i in _REGISTRY.keys()]))


def register(component, registered_name: str):
    _REGISTRY[component] = registered_name


def get_registered_name(component):
    return _REGISTRY[component]
