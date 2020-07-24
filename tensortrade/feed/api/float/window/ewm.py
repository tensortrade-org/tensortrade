
import numpy as np

from typing import List

from tensortrade.feed import Stream
from tensortrade.feed import Float


class ExponentialWeightedMovingAverage(Stream[float]):

    def __init__(self, alpha, adjust, ignore_na, min_periods):
        super().__init__()
        self.alpha = alpha
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.min_periods = max(min_periods, 1)

        self.i = 0
        self.n = 0

        self.avg = None
        self.factor = 1 - alpha
        self.new_wt = 1 if self.adjust else self.alpha
        self.old_wt = 1

    def forward(self) -> float:
        value = self.inputs[0].value
        if self.avg is None:
            is_observation = (value == value)
            self.n += int(is_observation)
            self.avg = value
            return self.avg if self.n >= self.min_periods else np.nan

        is_observation = (value == value)
        self.n += is_observation

        if self.avg == self.avg:

            if is_observation or not self.ignore_na:

                self.old_wt *= self.factor
                if is_observation:
                    # avoid numerical errors on constant series
                    if self.avg != value:
                        num = self.old_wt * self.avg + self.new_wt * value
                        den = self.old_wt + self.new_wt
                        self.avg = num / den

                    if self.adjust:
                        self.old_wt += self.new_wt
                    else:
                        self.old_wt = 1

        elif is_observation:
            self.avg = value

        return self.avg if self.n >= self.min_periods else np.nan

    def has_next(self) -> bool:
        return True

    def reset(self):
        self.i = 0
        self.n = 0

        self.avg = None
        self.old_wt = 1


class ExponentialWeightedMovingCovariance(Stream[float]):

    def __init__(self, alpha, adjust, ignore_na, min_periods, bias):
        super().__init__()
        self.alpha = alpha
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.min_periods = min_periods
        self.bias = bias

        self.i = 0
        self.n = 0

        self.minp = max(self.min_periods, 1)

        self.avg = None
        self.factor = 1 - alpha
        self.new_wt = 1 if self.adjust else self.alpha
        self.old_wt = 1

        self.mean_x = None
        self.mean_y = None

        self.cov = 0
        self.sum_wt = 1
        self.sum_wt2 = 1
        self.old_wt = 1

    def forward(self) -> float:
        v1 = self.inputs[0].value
        v2 = self.inputs[1].value
        if self.mean_x is None and self.mean_y is None:
            self.mean_x = v1
            self.mean_y = v2
            is_observation = (self.mean_x == self.mean_x) and (self.mean_y == self.mean_y)
            self.n += int(is_observation)
            if not is_observation:
                self.mean_x = np.nan
                self.mean_y = np.nan
            return (0. if self.bias else np.nan) if self.n >= self.minp else np.nan

        is_observation = (v1 == v1) and (v2 == v2)
        self.n += is_observation

        if self.mean_x == self.mean_x:
            if is_observation or not self.ignore_na:
                self.sum_wt *= self.factor
                self.sum_wt2 *= (self.factor * self.factor)
                self.old_wt *= self.factor
                if is_observation:
                    old_mean_x = self.mean_x
                    old_mean_y = self.mean_y

                    # avoid numerical errors on constant streams
                    wt_sum = self.old_wt + self.new_wt

                    if self.mean_x != v1:
                        self.mean_x = ((self.old_wt * old_mean_x) + (self.new_wt * v1)) / wt_sum

                    # avoid numerical errors on constant series
                    if self.mean_y != v2:
                        self.mean_y = ((self.old_wt * old_mean_y) + (self.new_wt * v2)) / wt_sum

                    d1 = old_mean_x - self.mean_x
                    d2 = old_mean_y - self.mean_y

                    d3 = v1 - self.mean_x
                    d4 = v2 - self.mean_y

                    t1 = self.old_wt * (self.cov + d1 * d2)
                    t2 = self.new_wt * d3 * d4
                    self.cov = (t1 + t2) / wt_sum

                    self.sum_wt += self.new_wt
                    self.sum_wt2 += self.new_wt * self.new_wt
                    self.old_wt += self.new_wt
                    if not self.adjust:
                        self.sum_wt /= self.old_wt
                        self.sum_wt2 /= self.old_wt * self.old_wt
                        self.old_wt = 1

        elif is_observation:
            self.mean_x = v1
            self.mean_y = v2

        if self.n >= self.minp:
            if not self.bias:
                numerator = self.sum_wt * self.sum_wt
                denominator = numerator - self.sum_wt2
                if denominator > 0:
                    output = ((numerator / denominator) * self.cov)
                else:
                    output = np.nan
            else:
                output = self.cov
        else:
            output = np.nan

        return output

    def has_next(self) -> bool:
        return True

    def reset(self):
        self.avg = None
        self.new_wt = 1 if self.adjust else self.alpha
        self.old_wt = 1

        self.mean_x = None
        self.mean_y = None

        self.cov = 0
        self.sum_wt = 1
        self.sum_wt2 = 1
        self.old_wt = 1


class EWM(Stream[List[float]]):

    def __init__(
            self,
            com=None,
            span=None,
            halflife=None,
            alpha=None,
            min_periods=0,
            adjust=True,
            ignore_na=False):
        super().__init__()
        self.com = com
        self.span = span
        self.halflife = halflife

        self.min_periods = min_periods
        self.adjust = adjust
        self.ignore_na = ignore_na

        if alpha:
            assert 0 < alpha <= 1
            self.alpha = alpha
        elif com:
            assert com >= 0
            self.alpha = 1 / (1 + com)
        elif span:
            assert span >= 1
            self.alpha = 2 / (1 + span)
        elif halflife:
            assert halflife > 0
            self.alpha = 1 - np.exp(np.log(0.5) / halflife)

        self.history = []
        self.weights = []

    def forward(self):
        value = self.inputs[0].value
        if self.ignore_na:
            if not np.isnan(value):
                self.history += [value]

                # Compute weights
                if not self.adjust and len(self.weights) > 0:
                    self.weights[-1] *= self.alpha
                self.weights += [(1 - self.alpha) ** len(self.history)]
        else:
            self.history += [value]

            # Compute weights
            if not self.adjust and len(self.weights) > 0:
                self.weights[-1] *= self.alpha
            self.weights += [(1 - self.alpha)**len(self.history)]

        return self.history, self.weights

    def has_next(self) -> bool:
        return True

    def mean(self) -> "Stream[float]":
        return ExponentialWeightedMovingAverage(
            alpha=self.alpha,
            min_periods=self.min_periods,
            adjust=self.adjust,
            ignore_na=self.ignore_na
        )(self.inputs[0]).astype("float")

    def var(self, bias=False) -> "Stream[float]":
        return ExponentialWeightedMovingCovariance(
            alpha=self.alpha,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            min_periods=self.min_periods,
            bias=bias
        )(self.inputs[0], self.inputs[0]).astype("float")

    def std(self, bias=False) -> "Stream[float]":
        return self.var(bias).sqrt()


@Float.register(["ewm"])
def ewm(s: "Stream[float]",
        com=None,
        span=None,
        halflife=None,
        alpha=None,
        min_periods=0,
        adjust=True,
        ignore_na=False) -> "Stream[List[float]]":
    return EWM(
        com=com,
        span=span,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na
    )(s)
