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
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class PlotlyTradingChart:
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
    def __init__(self, height: int = 800):
        self._height = height

        self.fig = None
        self._price_chart = None
        self._volume_chart = None
        self._performance_chart = None
        self._net_worth_chart = None
        self._base_annotations = None

    def _create_figure(self, performance_keys):
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                row_heights=[0.55, 0.15, 0.15, 0.15],
                # subplot_titles = ['', 'Volume', 'Performance', 'Net Worth']
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
            display(self.fig)  # Jupyter notebook function

    def _create_trade_annotations(self, trades, max_step):
        annotations = []
        for trade in trades:
            side = trade['side'].value
            if side == 'buy':
                color = 'DarkGreen'
                hovertext = 'Buy'
                ay = 15
            elif side == 'sell':
                color = 'FireBrick'
                hovertext = 'Sell'
                ay = -15
            else:
                raise ValueError(f"Trade side '{side}' is invalid. Valid values are 'buy' and 'sell'.")

            hovertext += ' {} @ {}'.format(trade['size'], trade['price'])
            annotation = go.layout.Annotation(
                x=trade['step'] - 1, y=trade['price'],
                ax=0, ay=ay, xref='x1', yref='y1', showarrow=True,
                arrowhead=2, arrowcolor=color, arrowwidth=4,
                arrowsize=0.8, hovertext=hovertext, opacity=0.6
                )
            annotations.append(annotation)

        return tuple(annotations)

    def render(self, title: str, price_history: pd.DataFrame, net_worth: pd.Series, performance: pd.DataFrame, trades):
        if self.fig is None:
            self._create_figure(performance.keys())

        self._price_chart.update(dict(
            open=price_history['open'],
            high=price_history['high'],
            low=price_history['low'],
            close=price_history['close']
        ))
        ann = self._create_trade_annotations(trades, price_history.shape[0])
        self.fig.layout.annotations = self._base_annotations + ann

        self._volume_chart.update({ 'y': price_history['volume'] })

        for trace in self.fig.select_traces(row=3):
            trace.update({ 'y': performance[trace.name] })

        self._net_worth_chart.update({ 'y': net_worth })  # net worth
        self.fig.layout.title = title

    def save(self, filename: str, directory: str = 'charts', fmt: str = 'png'):
        """Saves the current chart to a file.

        Note:
        This feature requires Orca to be installed and the server running.

        Arguments:
            filename: str, primary part of the image file name (e.g. "mountain" in "mountain.png".
            directory: str, directory to save to.
            fmt: str, the image format to save.
        """
        valid_formats = ['png', 'jpeg', 'webp', 'svg', 'pdf', 'eps']
        if fmt not in valid_formats:
            raise ValueError("Acceptable formats are '{}'. Found '{}'".format("', '".join(valid_formats), fmt))

        if directory is not None and not os.path.exists(directory):
            os.mkdir(directory)

        path = os.path.join(directory, filename + '.' + fmt)
        self.fig.write_image(path)
