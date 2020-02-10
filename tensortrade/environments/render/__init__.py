import importlib


_registry = {}


if importlib.util.find_spec("matplotlib") is not None:
    from .matplotlib_trading_chart import MatplotlibTradingChart

    _registry['matplotlib'] = MatplotlibTradingChart


if importlib.util.find_spec("plotly") is not None:
    from .plotly_stock_chart import PlotlyTradingChart

    _registry['plotly'] = PlotlyTradingChart
