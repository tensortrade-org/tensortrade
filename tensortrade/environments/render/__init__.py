import importlib

if importlib.util.find_spec("matplotlib") is not None:
    from .matplotlib_trading_chart import MatplotlibTradingChart
from .plotly_stock_chart import PlotlyTradingChart

_registry = {
    'matplotlib': MatplotlibTradingChart,
    'plotly': PlotlyTradingChart
}
