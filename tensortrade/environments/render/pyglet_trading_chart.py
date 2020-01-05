import pyglet
from gym.envs.classic_control import rendering

pyglet.text.Label.render = pyglet.text.Label.draw

RED = (.5, 0, 0)
GREEN = (0, .5, 0)
BRIGHT_RED = (1, 0, 0)
BRIGHT_GREEN = (0, 1, 0)
BLACK = (0, 0, 0)


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

    def __init__(self):
        super().__init__()
        self.viewer = rendering.Viewer(1280, 720)
        self.xstart = 0
        self.xend = 1280
        self.ystart = 200
        self.yend = 720
        self.height = self.yend - self.ystart
        self.width = self.xend - self.xstart
        self.price_label_width = self.width * .06
        self.volume_graph_height = self.height * .08
        self.candles_margin = min(5, self.volume_graph_height * .25)
        self.markers = []
        self._draw_axis()

    def render(self, dataframe: 'DataFrame',
               trades: 'List[Tuple[int, float, float]]'):
        n_candles = len(dataframe)
        width = self.width - self.price_label_width
        grid = width / (n_candles + 1)
        candle_width = width / (n_candles * 1.5)
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
            self._draw_candle(i * grid + self.xstart, values['open'],
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
            self._draw_volume_bar(i * grid + self.xstart,
                                  self.ystart, values['volume'],
                                  candle_width, color)
        xpos = self.xend - self.price_label_width / 2
        label = new_label(dataframe['low'].min(), xpos, tmp['low'].min())
        self.viewer.add_onetime(label)
        label = new_label(dataframe['high'].max(), xpos, tmp['high'].max())
        self.viewer.add_onetime(label)
        for price, color in self.markers:
            scaled_marker_price = self._scale(price, min_, max_)
            self._draw_marker_price(scaled_marker_price, color)
        self.markers = []
        return self.viewer.render()

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
        max_height = self.volume_graph_height
        left, right = pos - half_width, pos + half_width
        vol_bar = rendering.FilledPolygon([(left, ybase),
                                           (left, ybase + max_height * height),
                                           (right, ybase + max_height * height),
                                           (right, ybase)])
        vol_bar.set_color(*color)
        self.viewer.add_onetime(vol_bar)

    def _draw_axis(self):
        xaxis = rendering.Line(start=(self.xstart, self.ystart),
                               end=(self.xend, self.ystart))
        self.viewer.add_geom(xaxis)
        yaxis = rendering.Line(start=(self.xend - self.price_label_width, self.ystart),
                               end=(self.xend - self.price_label_width, self.yend))
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
                * (self.height - self.volume_graph_height
                   - self.candles_margin * 2)
                + self.ystart + self.volume_graph_height + self.candles_margin)

    def _draw_marker_price(self, scaled_price, color):
        order = rendering.Line(start=(self.xstart, scaled_price),
                               end=(self.xend - self.price_label_width,
                                    scaled_price))
        order.set_color(*color)
        self.viewer.add_onetime(order)
