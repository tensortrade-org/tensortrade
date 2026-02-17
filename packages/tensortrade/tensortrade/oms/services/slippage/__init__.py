from .random_slippage_model import RandomUniformSlippageModel
from .slippage_model import SlippageModel

_registry = {"uniform": RandomUniformSlippageModel}


def get(identifier: str) -> SlippageModel:
    """Gets the `SlippageModel` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `SlippageModel`

    Raises:
        KeyError: if identifier is not associated with any `SlippageModel`
    """
    if identifier not in _registry:
        raise KeyError(
            f"Identifier {identifier} is not associated with any `SlippageModel`."
        )
    return _registry[identifier]()
