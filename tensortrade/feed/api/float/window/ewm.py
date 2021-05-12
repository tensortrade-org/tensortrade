"""
ewm.py contains functions and classes for exponential weighted moving stream
operations.
"""

from typing import List, Tuple

import numpy as np

from tensortrade.feed.core.base import Stream
from tensortrade.feed.api.float import Float


class ExponentialWeightedMovingAverage(Stream[float]):
    r"""A stream operator that computes an exponential weighted moving average
    on a given float stream.

    Parameters
    ----------
    alpha : float
        The smoothing factor :math:`\alpha` directly,
        :math:`0 < \alpha \leq 1`.
    adjust : bool
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average).
    ignore_na : bool
        Ignore missing values when calculating weights.
    min_periods : int
        Minimum number of observations in window required to have a value
        (otherwise result is NA).

    References
    ----------
    .. [1] https://github.com/pandas-dev/pandas/blob/d9fff2792bf16178d4e450fe7384244e50635733/pandas/_libs/window/aggregations.pyx#L1801
    """

    def __init__(self,
                 alpha: float,
                 adjust: bool,
                 ignore_na: bool,
                 min_periods: int) -> None:
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

    def reset(self) -> None:
        self.i = 0
        self.n = 0

        self.avg = None
        self.old_wt = 1
        super().reset()


class ExponentialWeightedMovingCovariance(Stream[float]):
    r"""A stream operator that computes an exponential weighted moving average
    on a given float stream.

    Parameters
    ----------
    alpha : float
        The smoothing factor :math:`\alpha` directly,
        :math:`0 < \alpha \leq 1`.
    adjust : bool
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average).
    ignore_na : bool
        Ignore missing values when calculating weights.
    min_periods : int
        Minimum number of observations in window required to have a value
        (otherwise result is NA).
    bias : bool
        Use a standard estimation bias correction
    """

    def __init__(self,
                 alpha: float,
                 adjust: bool,
                 ignore_na: bool,
                 min_periods: int,
                 bias: bool) -> None:
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

    def reset(self) -> None:
        self.avg = None
        self.new_wt = 1 if self.adjust else self.alpha
        self.old_wt = 1

        self.mean_x = None
        self.mean_y = None

        self.cov = 0
        self.sum_wt = 1
        self.sum_wt2 = 1
        self.old_wt = 1
        super().reset()


class EWM(Stream[List[float]]):
    r"""Provide exponential weighted (EW) functions.

    Exactly one parameter: `com`, `span`, `halflife`, or `alpha` must be
    provided.

    Parameters
    ----------
    com : float, optional
        Specify decay in terms of center of mass,
        :math:`\alpha = 1 / (1 + com)`, for :math:`com \geq 0`.
    span : float, optional
        Specify decay in terms of span,
        :math:`\alpha = 2 / (span + 1)`, for :math:`span \geq 1`.
    halflife : float, str, timedelta, optional
        Specify decay in terms of half-life,
        :math:`\alpha = 1 - \exp\left(-\ln(2) / halflife\right)`, for
        :math:`halflife > 0`.
        If ``times`` is specified, the time unit (str or timedelta) over which an
        observation decays to half its value. Only applicable to ``mean()``
        and halflife value will not apply to the other functions.
    alpha : float, optional
        Specify smoothing factor :math:`\alpha` directly,
        :math:`0 < \alpha \leq 1`.
    min_periods : int, default 0
        Minimum number of observations in window required to have a value
        (otherwise result is NA).
    adjust : bool, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average).
        - When ``adjust=True`` (default), the EW function is calculated using weights
          :math:`w_i = (1 - \alpha)^i`. For example, the EW moving average of the series
          [:math:`x_0, x_1, ..., x_t`] would be:
        .. math::
            y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ... + (1 -
            \alpha)^t x_0}{1 + (1 - \alpha) + (1 - \alpha)^2 + ... + (1 - \alpha)^t}
        - When ``adjust=False``, the exponentially weighted function is calculated
          recursively:
        .. math::
            \begin{split}
                y_0 &= x_0\\
                y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,
            \end{split}
    ignore_na : bool, default False
        Ignore missing values when calculating weights.
        - When ``ignore_na=False`` (default), weights are based on absolute positions.
        - When ``ignore_na=True``, weights are based on relative positions.

    See Also
    --------
    .. [1] https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ewm.html

    References
    ----------
    .. [1] https://github.com/pandas-dev/pandas/blob/d9fff2792bf16178d4e450fe7384244e50635733/pandas/core/window/ewm.py#L65
    """

    def __init__(
            self,
            com: float = None,
            span: float = None,
            halflife: float = None,
            alpha: float = None,
            min_periods: int = 0,
            adjust: bool = True,
            ignore_na: bool = False):
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

    def forward(self) -> "Tuple[List[float], List[float]]":
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
        """Computes the exponential weighted moving average.

        Returns
        -------
        `Stream[float]`
            The exponential weighted moving average stream based on the
            underlying stream of values.
        """
        return ExponentialWeightedMovingAverage(
            alpha=self.alpha,
            min_periods=self.min_periods,
            adjust=self.adjust,
            ignore_na=self.ignore_na
        )(self.inputs[0]).astype("float")

    def var(self, bias=False) -> "Stream[float]":
        """Computes the exponential weighted moving variance.

        Returns
        -------
        `Stream[float]`
            The exponential weighted moving variance stream based on the
            underlying stream of values.
        """
        return ExponentialWeightedMovingCovariance(
            alpha=self.alpha,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            min_periods=self.min_periods,
            bias=bias
        )(self.inputs[0], self.inputs[0]).astype("float")

    def std(self, bias=False) -> "Stream[float]":
        """Computes the exponential weighted moving standard deviation.

        Returns
        -------
        `Stream[float]`
            The exponential weighted moving standard deviation stream based on
            the underlying stream of values.
        """
        return self.var(bias).sqrt()

    def reset(self) -> None:
        self.history = []
        self.weights = []
        super().reset()


@Float.register(["ewm"])
def ewm(s: "Stream[float]",
        com: float = None,
        span: float = None,
        halflife: float = None,
        alpha: float = None,
        min_periods: int = 0,
        adjust: bool = True,
        ignore_na: bool = False) -> "Stream[Tuple[List[float], List[float]]]":
    r"""Computes the weights and values in order to perform an exponential
    weighted moving operation.

    Parameters
    ----------
    s : `Stream[float]`
        A float stream.
    com : float, optional
        Specify decay in terms of center of mass,
        :math:`\alpha = 1 / (1 + com)`, for :math:`com \geq 0`.
    span : float, optional
        Specify decay in terms of span,
        :math:`\alpha = 2 / (span + 1)`, for :math:`span \geq 1`.
    halflife : float, optional
        Specify decay in terms of half-life,
        :math:`\alpha = 1 - \exp\left(-\ln(2) / halflife\right)`, for
        :math:`halflife > 0`.
    alpha : float, optional
        Specify smoothing factor :math:`\alpha` directly,
        :math:`0 < \alpha \leq 1`.
    min_periods : int, default 0
        Minimum number of observations in window required to have a value
        (otherwise result is NA).
    adjust : bool, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average).
    ignore_na : bool, default False
        Ignore missing values when calculating weights.

    Returns
    -------
    `Stream[Tuple[List[float], List[float]]]`
        A stream of weights and values to be used for computation of exponential
        weighted moving operations.
    """
    return EWM(
        com=com,
        span=span,
        halflife=halflife,
        alpha=alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na
    )(s)
