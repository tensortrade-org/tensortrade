import importlib

if importlib.util.find_spec("matplotlib") is not None:
    from .matplotlib_trading_chart import MatplotlibTradingChart

_registry = {
    'matplotlib': MatplotlibTradingChart
}
