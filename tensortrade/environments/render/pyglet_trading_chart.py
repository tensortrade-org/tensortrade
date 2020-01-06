import pyglet
from gym.envs.classic_control import rendering

pyglet.text.Label.render = pyglet.text.Label.draw

RED = (.5, 0, 0)
GREEN = (0, .5, 0)
BRIGHT_RED = (1, 0, 0)
BRIGHT_GREEN = (0, 1, 0)
BLACK = (0, 0, 0)
MARGIN = 5


def new_label(text, xpos, ypos, anchor_x='center'):
    label = pyglet.text.Label(f'{text}', font_name='Courier',
                              font_size=10, color=(0, 0, 0, 255),
                              x=xpos, y=ypos,
                              anchor_x=anchor_x, anchor_y='center')
    return label


class PygletTradingChart:
    # pylint: disable=attribute-defined-outside-init
    """An OHLCV trading visualization using
    pyglet made to render gym environments."""

    def __init__(self, width=960, height=540):
        super().__init__()
        self.viewer = rendering.Viewer(width, height)
        self.width = width
        self.height = height
        self.chart_xstart = MARGIN
        self.chart_xend = width - MARGIN
        chart_height = height * .75 - 2 * MARGIN
        self.chart_ystart = MARGIN
        self.chart_yend = chart_height - MARGIN
        self.chart_height = self.chart_yend - self.chart_ystart
        self.chart_width = self.chart_xend - self.chart_xstart
        self.price_label_width = self.chart_width * .06
        self.volume_chart_height = self.chart_height * .08
        self.candles_margin = min(MARGIN, self.volume_chart_height * .25)
        self.net_worths_chart_height = height * .25 - 2 * MARGIN
        self.markers = []
        self._draw_axis()

    def render(self, dataframe: 'DataFrame', net_worths: 'List[float]',
               trades: 'List[Tuple[int, float, float]]'):
        window_size = len(dataframe)
        if window_size <= 1:
            return
        self._render_chart(dataframe, trades, window_size)
        self._render_net_worths(net_worths.values, window_size)
        return self.viewer.render()

    def _render_net_worths(self, net_worths, window_size):
        initial = round(net_worths[0], 2)
        net_worth = round(net_worths[-1], 2)
        profit_percent = (net_worth - initial) / initial

        grid = (self.width - MARGIN * 2) / window_size
        label = new_label(f'Net worth: ${net_worth} '
                          f'| Profit: {profit_percent:.2%}',
                          self.width / 2, self.height - 10)
        self.viewer.add_onetime(label)
        if len(net_worths) > 1:
            max_ = max(net_worths)
            min_ = min(net_worths)
            values = (net_worths - min_) / (max_ - min_)
            values *= self.net_worths_chart_height
            start = max(1, len(net_worths) - window_size)
            for i in range(start, len(net_worths)):
                previous = values[i - 1] + self.chart_yend + 2 * MARGIN
                current = values[i] + self.chart_yend + 2 * MARGIN
                i -= start
                xaxis = rendering.Line(start=(i * grid, previous),
                                       end=((i + 1) * grid, current))
                self.viewer.add_onetime(xaxis)

    def _render_chart(self, dataframe, trades, window_size):
        width = self.chart_width - self.price_label_width
        grid = width / (window_size + 1)
        candle_width = width / (window_size * 1.5)
        cols = ['open', 'high', 'low', 'close']
        min_, max_ = self._min_max(dataframe[cols])
        tmp = self._scale(dataframe[cols], min_, max_)
        tmp['volume'] = dataframe['volume'] / dataframe['volume'].max()
        for i, (index, values) in enumerate(tmp.iterrows(), 1):
            sign = None
            for trade in trades:
                if trade[0] == index:
                    sign = (trade[2], self._scale(trade[1], min_, max_))
                    break
            self._draw_candle(i * grid + self.chart_xstart, values['open'],
                              values['high'], values['low'], values['close'],
                              candle_width, sign)
            color = BLACK
            if i > 1:
                pos = i - 1
                close = dataframe['close']
                if close[pos] > close[pos - 1]:
                    color = GREEN
                elif close[pos] < close[pos - 1]:
                    color = RED
            self._draw_volume_bar(i * grid + self.chart_xstart,
                                  self.chart_ystart, values['volume'],
                                  candle_width, color)
        xpos = self.chart_xend - self.price_label_width + 2
        label = new_label(dataframe['low'].min(), xpos, tmp['low'].min(), 'left')
        self.viewer.add_onetime(label)
        label = new_label(dataframe['high'].max(), xpos, tmp['high'].max(), 'left')
        self.viewer.add_onetime(label)
        for price, color in self.markers:
            scaled_marker_price = self._scale(price, min_, max_)
            self._draw_marker_price(scaled_marker_price, color)
        self.markers = []

    def add_onetime_marker(self, price, color=BLACK):
        self.markers.append((price, color))

    def _draw_candle(self, pos, open_, high, low, close, width, trade):
        half_width = width / 2
        left, right = pos - half_width, pos + half_width
        if open_ != close:
            open_close = rendering.FilledPolygon([
                (left, open_),
                (left, close),
                (right, close),
                (right, open_)
            ])
        else:
            open_close = rendering.Line(start=(left, open_), end=(right, open_))
        line = rendering.Line(start=(pos, low), end=(pos, high))
        if open_ > close:
            open_close.set_color(*RED)
            line.set_color(*RED)
        elif open_ < close:
            open_close.set_color(*GREEN)
            line.set_color(*GREEN)
        else:
            open_close.set_color(*BLACK)
            line.set_color(*BLACK)
        self.viewer.add_onetime(line)
        self.viewer.add_onetime(open_close)

        if trade:
            value = trade[1]
            if trade[0] == 'buy':
                sign = rendering.FilledPolygon([
                    (left, value),
                    (right, value),
                    ((left + right) / 2, value + width)
                ])
                sign.set_color(0, 1, 0)
            else:
                sign = rendering.FilledPolygon([
                    (left, value),
                    (right, value),
                    ((left + right) / 2, value - width)
                ])
                sign.set_color(1, 0, 0)
            self.viewer.add_onetime(sign)

    def _draw_volume_bar(self, pos, ybase, height, width, color):
        half_width = width / 2
        max_height = self.volume_chart_height
        left, right = pos - half_width, pos + half_width
        vol_bar = rendering.FilledPolygon([(left, ybase),
                                           (left, ybase + max_height * height),
                                           (right, ybase + max_height * height),
                                           (right, ybase)])
        vol_bar.set_color(*color)
        self.viewer.add_onetime(vol_bar)

    def _draw_axis(self):
        # BOTTOM
        xaxis = rendering.Line(start=(self.chart_xstart, self.chart_ystart),
                               end=(self.chart_xend, self.chart_ystart))
        self.viewer.add_geom(xaxis)
        # TOP
        xaxis = rendering.Line(start=(self.chart_xstart, self.chart_yend),
                               end=(self.chart_xend, self.chart_yend))
        self.viewer.add_geom(xaxis)
        # LEFT
        yaxis = rendering.Line(start=(self.chart_xstart, self.chart_ystart),
                               end=(self.chart_xstart, self.chart_yend))
        self.viewer.add_geom(yaxis)
        # MID RIGHT
        yaxis = rendering.Line(start=(self.chart_xend - self.price_label_width,
                                      self.chart_ystart),
                               end=(self.chart_xend - self.price_label_width,
                                    self.chart_yend))
        self.viewer.add_geom(yaxis)
        # RIGHT
        yaxis = rendering.Line(start=(self.chart_xend, self.chart_ystart),
                               end=(self.chart_xend, self.chart_yend))
        self.viewer.add_geom(yaxis)

    def _min_max(self, dataframe):
        min_ = dataframe.min().min()
        max_ = dataframe.max().max()
        for price, _ in self.markers:
            min_ = min(min_, price)
            max_ = max(max_, price)
        amplitude = (max_ - min_) / ((min_ + max_) / 2) / 2
        fee = 0.00075
        if amplitude < fee:
            min_ -= min_ * (fee - amplitude)
            max_ += max_ * (fee - amplitude)
        return min_, max_

    def _scale(self, value, min_, max_):
        return ((value - min_) / (max_ - min_)
                * (self.chart_height - self.volume_chart_height
                   - self.candles_margin * 2)
                + self.chart_ystart + self.volume_chart_height + self.candles_margin)

    def _draw_marker_price(self, scaled_price, color):
        order = rendering.Line(start=(self.chart_xstart, scaled_price),
                               end=(self.chart_xend - self.price_label_width,
                                    scaled_price))
        order.set_color(*color)
        self.viewer.add_onetime(order)
