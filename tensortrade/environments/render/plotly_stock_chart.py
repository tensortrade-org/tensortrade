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
from datetime import datetime
from collections import OrderedDict
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display, clear_output

from tensortrade.environments.render import BaseRenderer


class PlotlyTradingChart(BaseRenderer):
    """
    Trading visualization for TensorTrade using Plotly.

    Possible Future Enhancements:
    - Make the renderes modular with standard interface (render, save, ...).
      Create a renderer and configure one or more renders and attach them to the
      environement. All available renderers to be used for output.
    - Saving images without using Orca.
    - Let the environmemt pass only newly added data points since last render
      to add to the existing ones for performance. Alternatively, find data diff
      internally and add new ones (hint: panads .from_dict() could be used for
      better performance here).
    - Limit displayed step range for the case of a large number of steps and let
      the shown part of the chart slide after filling that range to keep showing
      recent data when added.

    References:
    https://plot.ly/python/figurewidget/
    https://plot.ly/python/subplots/
    https://plot.ly/python/reference/#candlestick
    https://plot.ly/python/#chart-events
    """
    def __init__(self, height: int = 800, datefmt: str = '%Y-%m-%d %H:%M:%S'):
        self._datefmt = datefmt
        self._height = height

        self.fig = None
        self._price_chart = None
        self._volume_chart = None
        self._performance_chart = None
        self._net_worth_chart = None
        self._base_annotations = None
        self._last_trade_step = 0
        self._show_chart = True

    @property
    def can_save(self):
        return False  # to change after testing the saving functionality

    @property
    def can_reset(self):
        return True

    def _create_figure(self, performance_keys):
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.55, 0.15, 0.15, 0.15],
        )
        fig.add_trace(go.Candlestick(name='Price', xaxis='x1', yaxis='y1',
                                     showlegend=False), row=1, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False)

        fig.add_trace(go.Bar(name='Volume', showlegend=False,
                             marker={ 'color': 'DodgerBlue' }),
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

        self.fig.update_annotations({'font': { 'size': 12 } })
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
                # hovertext = 'Buy'
                ay = 15
            elif trade.side.value == 'sell':
                color = 'FireBrick'
                # hovertext = 'Sell'
                ay = -15
            else:
                raise ValueError(f"Valid trade side values are 'buy' and 'sell'. Found '{trade.side.value}'.")

            hovertext = 'Step {step} [{datetime}]<br>{side} {qty} {quote_instrument} @ {price} {base_instrument} {type}<br>Total: {size} {quote_instrument} - Comm: {commission}'.format(
                step=trade.step,
                datetime=price_history.iloc[trade.step - 1]['datetime'],
                side=trade.side.value.upper(),
                qty=round(trade.size/trade.price, trade.pair.quote.precision),
                size=trade.size,
                quote_instrument=trade.quote_instrument,
                price=trade.price,
                base_instrument=trade.base_instrument,
                type=trade.type.value.upper(),
                commission=trade.commission
            )
            annotation = go.layout.Annotation(
                x=trade.step - 1, y=trade.price,
                ax=0, ay=ay, xref='x1', yref='y1', showarrow=True,
                arrowhead=2, arrowcolor=color, arrowwidth=4,
                arrowsize=0.8, hovertext=hovertext, opacity=0.6,
                hoverlabel=dict(bgcolor=color)
                )
            annotations.append(annotation)

        if trades:
            self._last_trade_step = trades[list(trades)[-1]][0].step
        return tuple(annotations)

    def _create_title(self, episode, max_episodes, step, max_steps):
        """Creates the chart title. Override or assign this method to callable
        to change the title format.
        """
        return 'Episode: {}/{} - Step: {}/{} @ {}'.format(
            episode + 1,
            max_episodes,
            step,
            max_steps,
            datetime.now().strftime(self._datefmt)
        )

    def render(self, episode: int, max_episodes: int, step: int, max_steps: int,
               price_history: pd.DataFrame, net_worth: pd.Series,
               performance: pd.DataFrame, trades
               ):
        if self.fig is None:
            self._create_figure(performance.keys())

        if self._show_chart:  # ensure chart visibility through notebook cell reruns
            display(self.fig)
            self._show_chart = False

        self.fig.layout.title = self._create_title(episode, max_episodes, step, max_steps)
        self._price_chart.update(dict(
            open=price_history['open'],
            high=price_history['high'],
            low=price_history['low'],
            close=price_history['close']
        ))
        self.fig.layout.annotations += self._create_trade_annotations(trades, price_history)

        self._volume_chart.update({ 'y': price_history['volume'] })

        for trace in self.fig.select_traces(row=3):
            trace.update({ 'y': performance[trace.name] })

        self._net_worth_chart.update({ 'y': net_worth })

    def reset(self):
        self._last_trade_step = 0
        if self.fig is None:
            return

        self.fig.layout.annotations = self._base_annotations
        clear_output(wait=True)
        self._show_chart = True


    def save(self, filename: str, path: str = 'charts', format: str = 'png'):
        """Saves the current chart to a file.

        Note:
        This feature requires Orca to be installed and the server running.

        Arguments:
            filename: str, primary part of the image file name (e.g. "mountain" in "mountain.png".
            path: str, path to save to.
            fmt: str, the image format to save.
        """
        valid_formats = ['png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        if format not in valid_formats:
            raise ValueError("Acceptable formats are '{}'. Found '{}'".format("', '".join(valid_formats), format))

        if path is not None and not os.path.exists(path):
            os.mkdir(path)

        pathname = os.path.join(path, filename + '.' + format)
        self.fig.write_image(pathname)
