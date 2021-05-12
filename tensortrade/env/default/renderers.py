# Copyright 2020 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import os
import sys
import logging
import importlib

from abc import abstractmethod
from datetime import datetime
from typing import Union, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd

from IPython.display import display, clear_output
from pandas.plotting import register_matplotlib_converters

from tensortrade.oms.orders import TradeSide
from tensortrade.env.generic import Renderer, TradingEnv


if importlib.util.find_spec("matplotlib"):
    import matplotlib.pyplot as plt

    from matplotlib import style

    style.use("ggplot")
    register_matplotlib_converters()

if importlib.util.find_spec("plotly"):
    import plotly.graph_objects as go

    from plotly.subplots import make_subplots


def _create_auto_file_name(filename_prefix: str,
                           ext: str,
                           timestamp_format: str = '%Y%m%d_%H%M%S') -> str:
    timestamp = datetime.now().strftime(timestamp_format)
    filename = filename_prefix + timestamp + '.' + ext
    return filename


def _check_path(path: str, auto_create: bool = True) -> None:
    if not path or os.path.exists(path):
        return

    if auto_create:
        os.mkdir(path)
    else:
        raise OSError(f"Path '{path}' not found.")


def _check_valid_format(valid_formats: list, save_format: str) -> None:
    if save_format not in valid_formats:
        raise ValueError("Acceptable formats are '{}'. Found '{}'".format("', '".join(valid_formats), save_format))


class BaseRenderer(Renderer):
    """The abstract base renderer to be subclassed when making a renderer
    the incorporates a `Portfolio`.
    """

    def __init__(self):
        super().__init__()
        self._max_episodes = None
        self._max_steps = None

    @staticmethod
    def _create_log_entry(episode: int = None,
                          max_episodes: int = None,
                          step: int = None,
                          max_steps: int = None,
                          date_format: str = "%Y-%m-%d %H:%M:%S %p") -> str:
        """
        Creates a log entry to be used by a renderer.

        Parameters
        ----------
        episode : int
            The current episode.
        max_episodes : int
            The maximum number of episodes that can occur.
        step : int
            The current step of the current episode.
        max_steps : int
            The maximum number of steps within an episode that can occur.
        date_format : str
            The format for logging the date.

        Returns
        -------
        str
            a log entry
        """
        log_entry = f"[{datetime.now().strftime(date_format)}]"

        if episode is not None:
            log_entry += f" Episode: {episode + 1}/{max_episodes if max_episodes else ''}"

        if step is not None:
            log_entry += f" Step: {step}/{max_steps if max_steps else ''}"

        return log_entry

    def render(self, env: 'TradingEnv', **kwargs):

        price_history = None
        if len(env.observer.renderer_history) > 0:
            price_history = pd.DataFrame(env.observer.renderer_history)

        performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')

        self.render_env(
            episode=kwargs.get("episode", None),
            max_episodes=kwargs.get("max_episodes", None),
            step=env.clock.step,
            max_steps=kwargs.get("max_steps", None),
            price_history=price_history,
            net_worth=performance.net_worth,
            performance=performance.drop(columns=['base_symbol']),
            trades=env.action_scheme.broker.trades
        )

    @abstractmethod
    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:
        """Renderers the current state of the environment.

        Parameters
        ----------
        episode : int
            The episode that the environment is being rendered for.
        max_episodes : int
            The maximum number of episodes that will occur.
        step : int
            The step of the current episode that is happening.
        max_steps : int
            The maximum number of steps that will occur in an episode.
        price_history : `pd.DataFrame`
            The history of instrument involved with the environment. The
            required columns are: date, open, high, low, close, and volume.
        net_worth : `pd.Series`
            The history of the net worth of the `portfolio`.
        performance : `pd.Series`
            The history of performance of the `portfolio`.
        trades : `OrderedDict`
            The history of trades for the current episode.
        """
        raise NotImplementedError()

    def save(self) -> None:
        """Saves the rendering of the `TradingEnv`.
        """
        pass

    def reset(self) -> None:
        """Resets the renderer.
        """
        pass


class EmptyRenderer(Renderer):
    """A renderer that does renders nothing.

    Needed to make sure that environment can function without requiring a
    renderer.
    """

    def render(self, env, **kwargs):
        pass


class ScreenLogger(BaseRenderer):
    """Logs information the screen of the user.

    Parameters
    ----------
    date_format : str
        The format for logging the date.
    """

    DEFAULT_FORMAT: str = "[%(asctime)-15s] %(message)s"

    def __init__(self, date_format: str = "%Y-%m-%d %-I:%M:%S %p"):
        super().__init__()
        self._date_format = date_format

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None):
        print(self._create_log_entry(episode, max_episodes, step, max_steps, date_format=self._date_format))


class FileLogger(BaseRenderer):
    """Logs information to a file.

    Parameters
    ----------
    filename : str
        The file name of the log file. If omitted, a file name will be
        created automatically.
    path : str
        The path to save the log files to. None to save to same script directory.
    log_format : str
        The log entry format as per Python logging. None for default. For
        more details, refer to https://docs.python.org/3/library/logging.html
    timestamp_format : str
        The format of the timestamp of the log entry. Node for default.
    """

    DEFAULT_LOG_FORMAT: str = '[%(asctime)-15s] %(message)s'
    DEFAULT_TIMESTAMP_FORMAT: str = '%Y-%m-%d %H:%M:%S'

    def __init__(self,
                 filename: str = None,
                 path: str = 'log',
                 log_format: str = None,
                 timestamp_format: str = None) -> None:
        super().__init__()
        _check_path(path)

        if not filename:
            filename = _create_auto_file_name('log_', 'log')

        self._logger = logging.getLogger(self.id)
        self._logger.setLevel(logging.INFO)

        if path:
            filename = os.path.join(path, filename)
        handler = logging.FileHandler(filename)
        handler.setFormatter(
            logging.Formatter(
                log_format if log_format is not None else self.DEFAULT_LOG_FORMAT,
                datefmt=timestamp_format if timestamp_format is not None else self.DEFAULT_TIMESTAMP_FORMAT
            )
        )
        self._logger.addHandler(handler)

    @property
    def log_file(self) -> str:
        """The filename information is being logged to. (str, read-only)
        """
        return self._logger.handlers[0].baseFilename

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None) -> None:
        log_entry = self._create_log_entry(episode, max_episodes, step, max_steps)
        self._logger.info(f"{log_entry} - Performance:\n{performance}")


class PlotlyTradingChart(BaseRenderer):
    """Trading visualization for TensorTrade using Plotly.

    Parameters
    ----------
    display : bool
        True to display the chart on the screen, False for not.
    height : int
        Chart height in pixels. Affects both display and saved file
        charts. Set to None for 100% height. Default is None.
    save_format : str
        A format to save the chart to. Acceptable formats are
        html, png, jpeg, webp, svg, pdf, eps. All the formats except for
        'html' require Orca. Default is None for no saving.
    path : str
        The path to save the char to if save_format is not None. The folder
        will be created if not found.
    filename_prefix : str
        A string that precedes automatically-created file name
        when charts are saved. Default 'chart_'.
    timestamp_format : str
        The format of the date shown in the chart title.
    auto_open_html : bool
        Works for save_format='html' only. True to automatically
        open the saved chart HTML file in the default browser, False otherwise.
    include_plotlyjs : Union[bool, str]
        Whether to include/load the plotly.js library in the saved
        file. 'cdn' results in a smaller file by loading the library online but
        requires an Internet connect while True includes the library resulting
        in much larger file sizes. False to not include the library. For more
        details, refer to https://plot.ly/python-api-reference/generated/plotly.graph_objects.Figure.html

    Notes
    -----
    Possible Future Enhancements:
        - Saving images without using Orca.
        - Limit displayed step range for the case of a large number of steps and let
          the shown part of the chart slide after filling that range to keep showing
          recent data as it's being added.

    References
    ----------
    .. [1] https://plot.ly/python-api-reference/generated/plotly.graph_objects.Figure.html
    .. [2] https://plot.ly/python/figurewidget/
    .. [3] https://plot.ly/python/subplots/
    .. [4] https://plot.ly/python/reference/#candlestick
    .. [5] https://plot.ly/python/#chart-events
    """

    def __init__(self,
                 display: bool = True,
                 height: int = None,
                 timestamp_format: str = '%Y-%m-%d %H:%M:%S',
                 save_format: str = None,
                 path: str = 'charts',
                 filename_prefix: str = 'chart_',
                 auto_open_html: bool = False,
                 include_plotlyjs: Union[bool, str] = 'cdn') -> None:
        super().__init__()
        self._height = height
        self._timestamp_format = timestamp_format
        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix
        self._include_plotlyjs = include_plotlyjs
        self._auto_open_html = auto_open_html

        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

        self.fig = None
        self._price_chart = None
        self._volume_chart = None
        self._performance_chart = None
        self._net_worth_chart = None
        self._base_annotations = None
        self._last_trade_step = 0
        self._show_chart = display

    def _create_figure(self, performance_keys: dict) -> None:
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.55, 0.15, 0.15, 0.15],
        )
        fig.add_trace(go.Candlestick(name='Price', xaxis='x1', yaxis='y1',
                                     showlegend=False), row=1, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)

        fig.add_trace(go.Bar(name='Volume', showlegend=False,
                             marker={'color': 'DodgerBlue'}),
                      row=2, col=1)

        for k in performance_keys:
            fig.add_trace(go.Scatter(mode='lines', name=k), row=3, col=1)

        fig.add_trace(go.Scatter(mode='lines', name='Net Worth', marker={'color': 'DarkGreen'}),
                      row=4, col=1)

        fig.update_xaxes(linecolor='Grey', gridcolor='Gainsboro')
        fig.update_yaxes(linecolor='Grey', gridcolor='Gainsboro')
        fig.update_xaxes(title_text='Price', row=1)
        fig.update_xaxes(title_text='Volume', row=2)
        fig.update_xaxes(title_text='Performance', row=3)
        fig.update_xaxes(title_text='Net Worth', row=4)
        fig.update_xaxes(title_standoff=7, title_font=dict(size=12))

        self.fig = go.FigureWidget(fig)
        self._price_chart = self.fig.data[0]
        self._volume_chart = self.fig.data[1]
        self._performance_chart = self.fig.data[2]
        self._net_worth_chart = self.fig.data[-1]

        self.fig.update_annotations({'font': {'size': 12}})
        self.fig.update_layout(template='plotly_white', height=self._height, margin=dict(t=50))
        self._base_annotations = self.fig.layout.annotations

    def _create_trade_annotations(self,
                                  trades: 'OrderedDict',
                                  price_history: 'pd.DataFrame') -> 'Tuple[go.layout.Annotation]':
        """Creates annotations of the new trades after the last one in the chart.

        Parameters
        ----------
        trades : `OrderedDict`
            The history of trades for the current episode.
        price_history : `pd.DataFrame`
            The price history of the current episode.

        Returns
        -------
        `Tuple[go.layout.Annotation]`
            A tuple of annotations used in the renderering process.
        """
        annotations = []
        for trade in reversed(trades.values()):
            trade = trade[0]

            tp = float(trade.price)
            ts = float(trade.size)

            if trade.step <= self._last_trade_step:
                break

            if trade.side.value == 'buy':
                color = 'DarkGreen'
                ay = 15
                qty = round(ts / tp, trade.quote_instrument.precision)

                text_info = dict(
                    step=trade.step,
                    datetime=price_history.iloc[trade.step - 1]['date'],
                    side=trade.side.value.upper(),
                    qty=qty,
                    size=ts,
                    quote_instrument=trade.quote_instrument,
                    price=tp,
                    base_instrument=trade.base_instrument,
                    type=trade.type.value.upper(),
                    commission=trade.commission
                )

            elif trade.side.value == 'sell':
                color = 'FireBrick'
                ay = -15
                # qty = round(ts * tp, trade.quote_instrument.precision)

                text_info = dict(
                    step=trade.step,
                    datetime=price_history.iloc[trade.step - 1]['date'],
                    side=trade.side.value.upper(),
                    qty=ts,
                    size=round(ts * tp, trade.base_instrument.precision),
                    quote_instrument=trade.quote_instrument,
                    price=tp,
                    base_instrument=trade.base_instrument,
                    type=trade.type.value.upper(),
                    commission=trade.commission
                )
            else:
                raise ValueError(f"Valid trade side values are 'buy' and 'sell'. Found '{trade.side.value}'.")

            hovertext = 'Step {step} [{datetime}]<br>' \
                        '{side} {qty} {quote_instrument} @ {price} {base_instrument} {type}<br>' \
                        'Total: {size} {base_instrument} - Comm.: {commission}'.format(**text_info)

            annotations += [go.layout.Annotation(
                x=trade.step - 1, y=tp,
                ax=0, ay=ay, xref='x1', yref='y1', showarrow=True,
                arrowhead=2, arrowcolor=color, arrowwidth=4,
                arrowsize=0.8, hovertext=hovertext, opacity=0.6,
                hoverlabel=dict(bgcolor=color)
            )]

        if trades:
            self._last_trade_step = trades[list(trades)[-1]][0].step

        return tuple(annotations)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: pd.DataFrame = None,
                   net_worth: pd.Series = None,
                   performance: pd.DataFrame = None,
                   trades: 'OrderedDict' = None) -> None:
        if price_history is None:
            raise ValueError("renderers() is missing required positional argument 'price_history'.")

        if net_worth is None:
            raise ValueError("renderers() is missing required positional argument 'net_worth'.")

        if performance is None:
            raise ValueError("renderers() is missing required positional argument 'performance'.")

        if trades is None:
            raise ValueError("renderers() is missing required positional argument 'trades'.")

        if not self.fig:
            self._create_figure(performance.keys())

        if self._show_chart:  # ensure chart visibility through notebook cell reruns
            display(self.fig)

        self.fig.layout.title = self._create_log_entry(episode, max_episodes, step, max_steps)
        self._price_chart.update(dict(
            open=price_history['open'],
            high=price_history['high'],
            low=price_history['low'],
            close=price_history['close']
        ))
        self.fig.layout.annotations += self._create_trade_annotations(trades, price_history)

        self._volume_chart.update({'y': price_history['volume']})

        for trace in self.fig.select_traces(row=3):
            trace.update({'y': performance[trace.name]})

        self._net_worth_chart.update({'y': net_worth})

        if self._show_chart:
            self.fig.show()

    def save(self) -> None:
        """Saves the current chart to a file.

        Notes
        -----
        All formats other than HTML require Orca installed and server running.
        """
        if not self._save_format:
            return
        else:
            valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
            _check_valid_format(valid_formats, self._save_format)

        _check_path(self._path)

        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=self._auto_open_html)
        else:
            self.fig.write_image(filename)

    def reset(self) -> None:
        self._last_trade_step = 0
        if self.fig is None:
            return

        self.fig.layout.annotations = self._base_annotations
        clear_output(wait=True)


class MatplotlibTradingChart(BaseRenderer):
    """ Trading visualization for TensorTrade using Matplotlib
    Parameters
    ---------
    display : bool
        True to display the chart on the screen, False for not.
    save_format : str
        A format to save the chart to. Acceptable formats are
        png, jpg, svg, pdf.
    path : str
        The path to save the char to if save_format is not None. The folder
        will be created if not found.
    filename_prefix : str
        A string that precedes automatically-created file name
        when charts are saved. Default 'chart_'.
    """
    def __init__(self,
                 display: bool = True,
                 save_format: str = None,
                 path: str = 'charts',
                 filename_prefix: str = 'chart_') -> None:
        super().__init__()
        self._volume_chart_height = 0.33

        self._df = None
        self.fig = None
        self._price_ax = None
        self._volume_ax = None
        self.net_worth_ax = None
        self._show_chart = display

        self._save_format = save_format
        self._path = path
        self._filename_prefix = filename_prefix

        if self._save_format and self._path and not os.path.exists(path):
            os.mkdir(path)

    def _create_figure(self) -> None:
        self.fig = plt.figure()

        self.net_worth_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
        self.price_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8,
                                         colspan=1, sharex=self.net_worth_ax)
        self.volume_ax = self.price_ax.twinx()
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)

    def _render_trades(self, step_range, trades) -> None:
        trades = [trade for sublist in trades.values() for trade in sublist]

        for trade in trades:
            if trade.step in range(sys.maxsize)[step_range]:
                date = self._df.index.values[trade.step]
                close = self._df['close'].values[trade.step]
                color = 'green'

                if trade.side is TradeSide.SELL:
                    color = 'red'

                self.price_ax.annotate(' ', (date, close),
                                       xytext=(date, close),
                                       size="large",
                                       arrowprops=dict(arrowstyle='simple', facecolor=color))

    def _render_volume(self, step_range, times) -> None:
        self.volume_ax.clear()

        volume = np.array(self._df['volume'].values[step_range])

        self.volume_ax.plot(times, volume,  color='blue')
        self.volume_ax.fill_between(times, volume, color='blue', alpha=0.5)

        self.volume_ax.set_ylim(0, max(volume) / self._volume_chart_height)
        self.volume_ax.yaxis.set_ticks([])

    def _render_price(self, step_range, times, current_step) -> None:
        self.price_ax.clear()

        self.price_ax.plot(times, self._df['close'].values[step_range], color="black")

        last_time = self._df.index.values[current_step]
        last_close = self._df['close'].values[current_step]
        last_high = self._df['high'].values[current_step]

        self.price_ax.annotate('{0:.2f}'.format(last_close), (last_time, last_close),
                               xytext=(last_time, last_high),
                               bbox=dict(boxstyle='round',
                                         fc='w', ec='k', lw=1),
                               color="black",
                               fontsize="small")

        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * self._volume_chart_height, ylim[1])

    # def _render_net_worth(self, step_range, times, current_step, net_worths, benchmarks):
    def _render_net_worth(self, step_range, times, current_step, net_worths) -> None:
        self.net_worth_ax.clear()
        self.net_worth_ax.plot(times, net_worths[step_range], label='Net Worth', color="g")
        self.net_worth_ax.legend()

        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        last_time = times[-1]
        last_net_worth = list(net_worths[step_range])[-1]

        self.net_worth_ax.annotate('{0:.2f}'.format(last_net_worth), (last_time, last_net_worth),
                                   xytext=(last_time, last_net_worth),
                                   bbox=dict(boxstyle='round',
                                             fc='w', ec='k', lw=1),
                                   color="black",
                                   fontsize="small")

        self.net_worth_ax.set_ylim(min(net_worths) / 1.25, max(net_worths) * 1.25)

    def render_env(self,
                   episode: int = None,
                   max_episodes: int = None,
                   step: int = None,
                   max_steps: int = None,
                   price_history: 'pd.DataFrame' = None,
                   net_worth: 'pd.Series' = None,
                   performance: 'pd.DataFrame' = None,
                   trades: 'OrderedDict' = None) -> None:
        if price_history is None:
            raise ValueError("renderers() is missing required positional argument 'price_history'.")

        if net_worth is None:
            raise ValueError("renderers() is missing required positional argument 'net_worth'.")

        if performance is None:
            raise ValueError("renderers() is missing required positional argument 'performance'.")

        if trades is None:
            raise ValueError("renderers() is missing required positional argument 'trades'.")

        if not self.fig:
            self._create_figure()

        if self._show_chart:
            plt.show(block=False)

        current_step = step - 1

        self._df = price_history
        if max_steps:
            window_size = max_steps
        else:
            window_size = 20

        current_net_worth = round(net_worth[len(net_worth)-1], 1)
        initial_net_worth = round(net_worth[0], 1)
        profit_percent = round((current_net_worth - initial_net_worth) / initial_net_worth * 100, 2)

        self.fig.suptitle('Net worth: $' + str(current_net_worth) +
                          ' | Profit: ' + str(profit_percent) + '%')

        window_start = max(current_step - window_size, 0)
        step_range = slice(window_start, current_step)

        times = self._df.index.values[step_range]

        if len(times) > 0:
            # self._render_net_worth(step_range, times, current_step, net_worths, benchmarks)
            self._render_net_worth(step_range, times, current_step, net_worth)
            self._render_price(step_range, times, current_step)
            self._render_volume(step_range, times)
            self._render_trades(step_range, trades)

        self.price_ax.set_xticklabels(times, rotation=45, horizontalalignment='right')

        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)
        plt.pause(0.001)

    def save(self) -> None:
        """Saves the rendering of the `TradingEnv`.
        """
        if not self._save_format:
            return
        else:
            valid_formats = ['png', 'jpeg', 'svg', 'pdf']
            _check_valid_format(valid_formats, self._save_format)

        _check_path(self._path)
        filename = _create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(self._path, filename)
        self.fig.savefig(filename, format=self._save_format)

    def reset(self) -> None:
        """Resets the renderer.
        """
        self.fig = None
        self._price_ax = None
        self._volume_ax = None
        self.net_worth_ax = None
        self._df = None


_registry = {
    "screen-log": ScreenLogger,
    "file-log": FileLogger,
    "plotly": PlotlyTradingChart,
    "matplot": MatplotlibTradingChart
}


def get(identifier: str) -> 'BaseRenderer':
    """Gets the `BaseRenderer` that matches the identifier.

    Parameters
    ----------
    identifier : str
        The identifier for the `BaseRenderer`

    Returns
    -------
    `BaseRenderer`
        The renderer associated with the `identifier`.

    Raises
    ------
    KeyError:
        Raised if identifier is not associated with any `BaseRenderer`
    """
    if identifier not in _registry.keys():
        msg = f"Identifier {identifier} is not associated with any `BaseRenderer`."
        raise KeyError(msg)
    return _registry[identifier]()
