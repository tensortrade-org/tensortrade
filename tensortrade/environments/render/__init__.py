import importlib
from .base_renderer import BaseRenderer
from .screen_logger import ScreenLogger
from .file_logger import FileLogger


_registry = {
    'screenlog': ScreenLogger,
    'filelog': FileLogger,
}


if importlib.util.find_spec("matplotlib") is not None:
    from .matplotlib_trading_chart import MatplotlibTradingChart

    _registry['matplotlib'] = MatplotlibTradingChart


if importlib.util.find_spec("plotly") is not None:
    from .plotly_stock_chart import PlotlyTradingChart

    _registry['plotly'] = PlotlyTradingChart


def get(identifier: str) -> BaseRenderer:
    """Gets the `BaseRenderer` that matches the identifier.

    Arguments:
        identifier: The identifier for the `RewardScheme`

    Raises:
        KeyError: if identifier is not associated with any `RewardScheme`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `BaseRenderer`.'.format(identifier))
    return _registry[identifier]()
