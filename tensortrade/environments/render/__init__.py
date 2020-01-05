import importlib

if importlib.util.find_spec('matplotlib') is not None:
    from .matplotlib_trading_chart import MatplotlibTradingChart

if importlib.util.find_spec('pyglet') is not None:
    from .pyglet_trading_chart import PygletTradingChart

_registry = {
    'matplotlib': MatplotlibTradingChart,
    'pyglet': PygletTradingChart
}
