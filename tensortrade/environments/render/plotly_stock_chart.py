# Copyright 2019 The TensorTrade Authors.
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
from typing import Union
from collections import OrderedDict
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display, clear_output

from tensortrade.environments.render import BaseRenderer
from tensortrade.environments.utils.helpers import create_auto_file_name, check_path


class PlotlyTradingChart(BaseRenderer):
    """
    Trading visualization for TensorTrade using Plotly.

    Possible Future Enhancements:
    - Saving images without using Orca.
    - Limit displayed step range for the case of a large number of steps and let
      the shown part of the chart slide after filling that range to keep showing
      recent data as it's being added.

    References:
    https://plot.ly/python-api-reference/generated/plotly.graph_objects.Figure.html
    https://plot.ly/python/figurewidget/
    https://plot.ly/python/subplots/
    https://plot.ly/python/reference/#candlestick
    https://plot.ly/python/#chart-events
    """
    def __init__(self, display: bool = True, height: int=None,
                 timestamp_format: str = '%Y-%m-%d %H:%M:%S',
                 save_format: str=None, path: str = 'charts',
                 filename_prefix='chart_', auto_open_html=False,
                 include_plotlyjs: Union[bool, str]='cdn'
                 ):
        """
        Arguments:
            display: True to display the chart on the screen, False for not.
            height: Chart height in pixels. Affects both display and saved file
                charts. Set to None for 100% height. Default is None.
            save_format: A format to save the chart to. Acceptable formats are
                html, png, jpeg, webp, svg, pdf, eps. All the formats except for
                'html' require Orca. Default is None for no saving.
            path: The path to save the char to if save_format is not None. The folder
                will be created if not found.
            filename_prefix: A string that precedes automatically-created file name
                when charts are saved. Default 'chart_'.
            timestamp_format: The format of the date shown in the chart title.
            auto_open_html: Works for save_format='html' only. True to automatically
                open the saved chart HTML file in the default browser, False otherwise.
            include_plotlyjs: Whether to include/load the plotly.js library in the saved
                file. 'cdn' results in a smaller file by loading the library online but
                requires an Internet connect while True includes the library resulting
                in much larger file sizes. False to not include the library. For more
                details, refer to https://plot.ly/python-api-reference/generated/plotly.graph_objects.Figure.html
        """
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
        self._show_chart = True

    def _create_figure(self, performance_keys):
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

        fig.add_trace(go.Scatter(mode='lines', name='Net Worth', marker={ 'color': 'DarkGreen' }),
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

    def _create_trade_annotations(self, trades: OrderedDict, price_history: pd.DataFrame):
        """Creates annotations of the new trades after the last one in the chart."""
        annotations = []
        for trade in reversed(trades.values()):
            trade = trade[0]

            if trade.step <= self._last_trade_step:
                break

            if trade.side.value == 'buy':
                color = 'DarkGreen'
                ay = 15
            elif trade.side.value == 'sell':
                color = 'FireBrick'
                ay = -15
            else:
                raise ValueError(f"Valid trade side values are 'buy' and 'sell'. Found '{trade.side.value}'.")

            hovertext = 'Step {step} [{datetime}]<br>{side} {qty} {quote_instrument} @ {price} {base_instrument} {type}<br>Total: {size} {base_instrument} - Comm.: {commission}'.format(
                step=trade.step,
                datetime=price_history.iloc[trade.step - 1]['datetime'],
                side=trade.side.value.upper(),
                qty=round(trade.size/trade.price, trade.exchange_pair.pair.quote.precision),
                size=trade.size,
                quote_instrument=trade.quote_instrument,
                price=float(trade.price),
                base_instrument=trade.base_instrument,
                type=trade.type.value.upper(),
                commission=trade.commission
            )
            annotation = go.layout.Annotation(
                x=trade.step - 1, y=float(trade.price),
                ax=0, ay=ay, xref='x1', yref='y1', showarrow=True,
                arrowhead=2, arrowcolor=color, arrowwidth=4,
                arrowsize=0.8, hovertext=hovertext, opacity=0.6,
                hoverlabel=dict(bgcolor=color)
                )
            annotations.append(annotation)

        if trades:
            self._last_trade_step = trades[list(trades)[-1]][0].step

        return tuple(annotations)

    def render(self, episode: int = None, max_episodes: int = None,
               step: int = None, max_steps: int = None,
               price_history: pd.DataFrame = None, net_worth: pd.Series = None,
               performance: pd.DataFrame = None, trades: 'OrderedDict' = None
               ):
        if price_history is None:
            raise ValueError("render() is missing required positional argument 'price_history'.")

        if net_worth is None:
            raise ValueError("render() is missing required positional argument 'net_worth'.")

        if performance is None:
            raise ValueError("render() is missing required positional argument 'performance'.")

        if trades is None:
            raise ValueError("render() is missing required positional argument 'trades'.")

        if not self.fig:
            self._create_figure(performance.keys())

        if self._show_chart:  # ensure chart visibility through notebook cell reruns
            display(self.fig)
            self._show_chart = False

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

    def reset(self):
        self._last_trade_step = 0
        if self.fig is None:
            return

        self.fig.layout.annotations = self._base_annotations
        clear_output(wait=True)
        self._show_chart = True

    def save(self):
        """Saves the current chart to a file.

        Note:
        All formats other than HTML require Orca installed and server running.

        Arguments:
            filename: str, primary part of the image file name (e.g. "mountain" in "mountain.png".
            path: The path to save the chart to.
            format: The chart format to save.
        """
        if not self._save_format:
            return
        else:
            valid_formats = ['html', 'png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
            if self._save_format not in valid_formats:
                raise ValueError("Acceptable formats are '{}'. Found '{}'".format("', '".join(valid_formats), self._save_format))

        check_path(self._path)

        filename = create_auto_file_name(self._filename_prefix, self._save_format)
        filename = os.path.join(self._path, filename)
        if self._save_format == 'html':
            self.fig.write_html(file=filename, include_plotlyjs='cdn', auto_open=self._auto_open_html)
        else:
            self.fig.write_image(filename)
