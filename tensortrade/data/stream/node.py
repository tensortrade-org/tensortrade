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
# limitations under the License.

"""
References:
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/module/module.py
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/base_layer.py
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/node.py
"""

import math
import functools

import numpy as np

from abc import abstractmethod
from typing import Union, Callable

from tensortrade.base.core import Observable


class Node(Observable):

    names = {}

    def __init__(self, name: str = None):
        super().__init__()

        if not name:
            name = self.generic_name

            if name in Node.names.keys():
                Node.names[name] += 1
                name += ":" + str(Node.names[name] - 1)
            else:
                Node.names[name] = 0

        self._name = name
        self.inputs = []

        if len(Module.CONTEXTS) > 0:
            Module.CONTEXTS[-1].add_node(self)

    @property
    def generic_name(self) -> str:
        return "node"

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def rename(self, name: str, preserve_ns: bool = True) -> 'Node':
        if preserve_ns and len(Module.CONTEXTS) > 0:
            module_name = Module.CONTEXTS[-1].name
            name = module_name + ":/" + name
        self._name = name
        return self

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __call__(self, *inputs):
        self.inputs = []

        for node in inputs:
            if isinstance(node, Module):
                if not node.built:
                    with node:
                        node.build()

                    node.built = True

                self.inputs += node.flatten()
            else:
                self.inputs += [node]

        return self

    def run(self):
        self.value = self.forward()

        for listener in self.listeners:
            listener.on_next(self.value)

    def apply(self, func: Callable[[any], any]) -> 'Node':
        name = "Apply({},{})".format(self.name, func.__name__)
        return Apply(func=func, name=name)(self)

    def pow(self, power: float) -> 'Node':
        name = "Pow({},{})".format(self.name, power)
        return self.apply(lambda x: np.power(x, power)).rename(name)

    def sqrt(self) -> 'Node':
        name = "Sqrt({})".format(self.name)
        return self.apply(np.sqrt).rename(name)

    def square(self) -> 'Node':
        name = "Square({})".format(self.name)
        return self.pow(2).rename(name)

    def log(self) -> 'Node':
        name = "Log({})".format(self.name)
        return self.apply(np.log).rename(name)

    def abs(self) -> 'Node':
        name = "Abs({})".format(self.name)
        return self.apply(np.abs).rename(name)

    def ceil(self) -> 'Node':
        name = "Ceil({})".format(self.name)
        return self.apply(np.ceil).rename(name)

    def floor(self) -> 'Node':
        name = "Floor({})".format(self.name)
        return self.apply(np.floor).rename(name)

    def max(self, other) -> 'Node':
        assert isinstance(other, Node)
        name = "Max({},{})".format(self.name, other.name)
        return BinOp(np.maximum)(self, other).rename(name)

    def min(self, other) -> 'Node':
        assert isinstance(other, Node)
        name = "Min({},{})".format(self.name, other.name)
        return BinOp(np.minimum)(self, other).rename(name)

    def clamp_min(self, c_min: float):
        name = "ClampMin({},{})".format(self.name, c_min)
        return BinOp(np.maximum)(self, Constant(c_min)).rename(name)

    def clamp_max(self, c_max: float):
        name = "ClampMax({},{})".format(self.name, c_max)
        return BinOp(np.minimum)(self, Constant(c_max)).rename(name)

    def clamp(self, c_min: float, c_max: float):
        name = "Clamp({},{},{})".format(self.name, c_min, c_max)
        return self.clamp_min(c_min).clamp_max(c_max).rename(name)

    def __add__(self, other):
        if np.isscalar(other):
            other = Constant(other, "Constant({})".format(other))
            name = "Add({},{})".format(self.name, other.name)
            return BinOp(np.add, name)(self, other)
        assert isinstance(other, Node)
        name = "Add({},{})".format(self.name, other.name)
        return BinOp(np.add, name)(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if np.isscalar(other):
            other = Constant(other, "Constant({})".format(other))
            name = "Subtract({},{})".format(self.name, other.name)
            return BinOp(np.subtract, name)(self, other)
        assert isinstance(other, Node)
        name = "Subtract({},{})".format(self.name, other.name)
        return BinOp(np.subtract, name)(self, other)

    def __rsub__(self, other):
        if not np.isscalar(other):
            raise Exception("Invalid node operation.")
        other = Constant(other, "Constant({})".format(other))
        name = "Subtract({},{})".format(other.name, self.name)
        return BinOp(np.subtract, name)(other, self)

    def __mul__(self, other):
        if np.isscalar(other):
            other = Constant(other, "Constant({})".format(other))
            name = "Multiply({},{})".format(self.name, other.name)
            return BinOp(np.multiply, name)(self, other)
        assert isinstance(other, Node)
        name = "Multiply({},{})".format(self.name, other.name)
        return BinOp(np.multiply, name)(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if np.isscalar(other):
            other = Constant(other, "Constant({})".format(other))
            name = "Divide({},{})".format(self.name, other.name)
            return BinOp(np.divide, name)(self, other)
        assert isinstance(other, Node)
        name = "Divide({},{})".format(self.name, other.name)
        return BinOp(np.divide, name)(self, other)

    def __rtruediv__(self, other):
        if not np.isscalar(other):
            raise Exception("Invalid node operation.")
        other = Constant(other, "Constant({})".format(other))
        name = "Divide({},{})".format(other.name, self.name)
        return BinOp(np.divide, name)(other, self)

    def __pow__(self, power, modulo=None):
        return self.pow(power)

    def __abs__(self):
        return self.abs()

    def __ceil__(self):
        return self.ceil()

    def __floor__(self):
        return self.floor()

    def __neg__(self):
        name = "Neg({})".format(self.name)
        return self.apply(np.negative).rename(name)

    def lag(self, lag: int = 1):
        name = "Lag({},{})".format(self.name, lag)
        return Lag(lag=lag, name=name)(self)

    def freeze(self):
        name = "Freeze({})".format(self.name)
        return Freeze(name)(self)

    def diff(self, periods: int = 1):
        name = "Diff({},{})".format(self.name, periods)
        return (self - self.lag(periods)).rename(name)

    def cumsum(self):
        name = "CumSum({})".format(self.name)
        return CumSum(name)(self)

    def cumprod(self):
        name = "CumProd({})".format(self.name)
        return CumProd(name)(self)

    def cummin(self, skip_na: bool = True):
        name = "CumMin({})".format(self.name)
        return CumMin(skip_na=skip_na, name=name)(self)

    def cummax(self, skip_na: bool = True):
        name = "CumMax({})".format(self.name)
        return CumMax(skip_na=skip_na, name=name)(self)

    def pct_change(self, periods: int = 1):
        name = "PercentChange({},{})".format(self.name, periods)
        return ((self / self.lag(periods)) - Constant(1)).rename(name)

    def fillna(self, fill_value: float = 0):
        name = "FillNa({},{})".format(self.name, fill_value)
        return FillNa(fill_value, name)(self)

    def ffill(self):
        name = "ForwardFill({})".format(self.name)
        return ForwardFill(name)(self)

    def expanding(self, warmup: int = 1):
        return Expanding(warmup)(self)

    def rolling(self, window: int, warmup: int = 0):
        return Rolling(window, warmup)(self)

    def ewm(self,
            com: float = None,
            span: float = None,
            halflife: float = None,
            alpha: float = None,
            warmup: int = 0,
            adjust: bool = True,
            ignore_na: bool = False):
        return EWM(com, span, halflife, alpha, warmup, adjust, ignore_na)(self)

    def is_na(self):
        return not self.value or np.isnan(self.value)

    def is_finite(self):
        return not self.value or np.isfinite(self.value)

    @abstractmethod
    def forward(self):
        raise NotImplementedError()

    @abstractmethod
    def has_next(self):
        raise NotImplementedError()

    def reset(self):
        for listener in self.listeners:
            if hasattr(listener, "reset"):
                listener.reset()

    def __str__(self):
        return "<Node: name={}, type={}>".format(self.name,
                                                 str(self.__class__.__name__).lower())

    def __repr__(self):
        return str(self)


class Module(Node):

    CONTEXTS = []

    def __init__(self, name: str = None):
        super(Module, self).__init__(name)

        self.submodules = []
        self.variables = []
        self.built = False

    @property
    def generic_name(self) -> str:
        return "module"

    def add_node(self, node: 'Node'):
        node.name = self.name + ":/" + node.name

        if isinstance(node, Module):
            self.submodules += [node]
        else:
            self.variables += [node]

    def build(self):
        pass

    def __enter__(self):
        self.CONTEXTS += [self]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.CONTEXTS.pop()
        return self

    def flatten(self):
        nodes = [node for node in self.variables]

        for module in self.submodules:
            nodes += module.flatten()

        return nodes

    def forward(self):
        return


class Constant(Node):

    def __init__(self, constant: float, name: str = None):
        super().__init__(name)
        self.constant = constant

    def forward(self):
        return self.constant

    def has_next(self):
        return True


class Lag(Node):

    def __init__(self, lag: int = 1, name: str = None):
        super().__init__(name)
        self.lag = lag
        self.runs = 0
        self.history = []

    @property
    def generic_name(self) -> str:
        return "lag"

    def forward(self):
        node = self.inputs[0]
        if self.runs < self.lag:
            self.runs += 1
            self.history.insert(0, node.value)
            return np.nan

        self.history.insert(0, node.value)
        return self.history.pop()

    def has_next(self):
        return True

    def reset(self):
        super().reset()
        self.runs = 0
        self.history = []


class BinOp(Node):

    def __init__(self, op, name: str = None):
        super().__init__(name)
        self.op = op

    def generic_name(self) -> str:
        return str(self.op.__name__)

    def forward(self):
        return self.op(self.inputs[0].value, self.inputs[1].value)

    def has_next(self):
        return True


class Reduce(Node):

    def __init__(self,
                 func: Callable[[float, float], float],
                 name: str = None):
        super().__init__(name)
        self.func = func

    @property
    def generic_name(self) -> str:
        return "reduce"

    def forward(self):
        return functools.reduce(self.func, [node.value for node in self.inputs])

    def has_next(self):
        return True


class Select(Node):

    def __init__(self, selector: Union[Callable[[str], bool], str]):
        if isinstance(selector, str):
            self.key = selector
            self.selector = lambda x: x.name == selector
        else:
            self.key = None
            self.selector = selector

        super().__init__(self.key)
        self._node = None

    @property
    def generic_name(self) -> str:
        return "select"

    def forward(self):
        if not self._node:
            self._node = list(filter(self.selector, self.inputs))[0]
            self.name = self._node.name

        return self._node.value

    def has_next(self):
        return True


class Lambda(Node):

    def __init__(self, extract: Callable[[any], any], obj: any, name: str = None):
        super().__init__(name)
        self.extract = extract
        self.obj = obj

    @property
    def generic_name(self) -> str:
        return "lambda"

    def forward(self):
        return self.extract(self.obj)

    def has_next(self):
        return True


class Apply(Node):

    def __init__(self, func: Callable[[any], any], name: str = None):
        super().__init__(name)
        self.name = name
        self.func = func

    @property
    def generic_name(self) -> str:
        return "apply"

    def forward(self):
        node = self.inputs[0]
        return self.func(node.value)

    def has_next(self):
        return True


class Forward(Lambda):

    def __init__(self, node: 'Node'):
        super().__init__(
            name=node.name,
            extract=lambda x: x.value,
            obj=node
        )
        self(node)


class Condition(Module):

    def __init__(self, condition: Callable[['Node'], bool], name: str = None):
        super().__init__(name)
        self.condition = condition

    def build(self):
        self.variables = list(filter(self.condition, self.inputs))

    def has_next(self):
        return True


class ForwardFill(Node):

    def __init__(self, name: str = None):
        super().__init__(name)
        self.previous = None

    @property
    def generic_name(self):
        return "ffill"

    def forward(self):
        node = self.inputs[0]
        if not self.previous or np.isfinite(node.value):
            self.previous = node.value
        return self.previous

    def has_next(self):
        return True


class FillNa(Node):

    def __init__(self, fill_value: float = 0, name: str = None):
        super().__init__(name)
        self.fill_value = fill_value

    def forward(self):
        node = self.inputs[0]
        if node.is_na():
            return self.fill_value
        return node.value

    def has_next(self):
        return True


class CumSum(Node):

    def __init__(self, name: str = None):
        super().__init__(name)
        self.csum = 0

    def forward(self):
        node = self.inputs[0]
        self.csum += node.value
        return self.csum

    def has_next(self):
        return True


class CumProd(Node):

    def __init__(self, name: str = None):
        super().__init__(name)
        self.cprod = 1

    def forward(self):
        node = self.inputs[0]
        self.cprod *= node.value
        return self.cprod

    def has_next(self):
        return True


class CumMax(Node):

    def __init__(self, skip_na: bool = True, name: str = None):
        super().__init__(name)
        self.skip_na = skip_na
        self.cmax = None

    def forward(self):
        node = self.inputs[0]
        if self.skip_na:
            if (self.cmax is None) or (not node.is_na() and self.is_na()):
                self.cmax = node.value
            else:
                if not node.is_na() and node.value > self.cmax:
                    self.cmax = node.value
        else:
            if self.cmax is None:
                self.cmax = node.value
            elif node.value > self.cmax:
                self.cmax = node.value
        return self.cmax

    def has_next(self):
        return True


class CumMin(Node):

    def __init__(self, skip_na: bool = True, name: str = None):
        super().__init__(name)
        self.skip_na = skip_na
        self.cmin = None

    def forward(self):
        node = self.inputs[0]
        if self.skip_na:
            if (self.cmin is None) or (not node.is_na() and self.is_na()):
                self.cmin = node.value
            else:
                if not node.is_na() and node.value < self.cmin:
                    self.cmin = node.value
        else:
            if self.cmin is None:
                self.cmin = node.value
            elif node.value < self.cmin:
                self.cmin = node.value
        return self.cmin

    def has_next(self):
        return True


class Freeze(Node):

    def __init__(self, name: str = None):
        super().__init__(name)
        self.freeze_value = None

    @property
    def generic_name(self):
        return "freeze"

    def forward(self):
        node = self.inputs[0]
        if not self.freeze_value:
            self.freeze_value = node.value
        return self.freeze_value

    def has_next(self):
        return True

    def reset(self):
        self.freeze_value = None


class Expanding(Node):

    def __init__(self, warmup: int = 1, name: str = None):
        super().__init__(name)
        self.warmup = warmup
        self.history = []

    @property
    def generic_name(self):
        return "expanding"

    def forward(self):
        node = self.inputs[0]
        self.history += [node.value]
        return self.history

    def has_next(self):
        return True

    def agg(self, func):
        name = "Expanding{}({})".format(func.__name__.capitalize(), self.name)
        return RollingNode(func, name)(self)

    def count(self):
        name = "ExpandingCount({})".format(self.name)
        return RollingCount(name)(self)

    def sum(self):
        name = "ExpandingSum({})".format(self.name)
        return self.agg(np.sum).rename(name)

    def mean(self):
        name = "ExpandingMean({})".format(self.name)
        return self.agg(np.mean).rename(name)

    def var(self):
        name = "ExpandingVar({})".format(self.name)
        return self.agg(lambda x: np.var(x, ddof=1)).rename(name)

    def median(self):
        name = "ExpandingMedian({})".format(self.name)
        return self.agg(np.median).rename(name)

    def std(self):
        name = "ExpandingSD({})".format(self.name)
        return self.agg(lambda x: np.std(x, ddof=1)).rename(name)

    def min(self):
        name = "ExpandingMin({})".format(self.name)
        return self.agg(np.min).rename(name)

    def max(self):
        name = "ExpandingMax({})".format(self.name)
        return self.agg(np.max).rename(name)


class ExpandingNode(Node):

    def __init__(self, func, name: str = None):
        super().__init__(name)
        self.func = func

    def forward(self):
        expanding = self.inputs[0]
        history = expanding.value
        if len(history) < expanding.warmup:
            return np.nan
        return self.func(history)

    def has_next(self):
        return True


class ExpandingCount(ExpandingNode):

    def __init__(self, name: str = None):
        super().__init__(lambda w: (~np.isnan(w)).sum(), name)

    def forward(self):
        expanding = self.inputs[0]
        return self.func(expanding.value)


class Rolling(Node):

    def __init__(self, window: int, warmup: int = 0, name: str = None):
        super().__init__(name)
        assert warmup <= window
        self.window = window
        self.warmup = warmup
        self.history = []

    @property
    def generic_name(self):
        return "rolling"

    def forward(self):
        node = self.inputs[0]
        self.history.insert(0, node.value)
        if len(self.history) > self.window:
            self.history.pop()
        return self.history

    def has_next(self):
        return True

    def agg(self, func):
        name = "Rolling{}({},{})".format(func.__name__.capitalize(), self.name, self.window)
        return RollingNode(func, name)(self)

    def count(self):
        name = "RollingCount({},{})".format(self.name, self.window)
        return RollingCount(name)(self)

    def sum(self):
        name = "RollingSum({},{})".format(self.name, self.window)
        return self.agg(np.sum).rename(name)

    def mean(self):
        name = "RollingMean({},{})".format(self.name, self.window)
        return self.agg(np.mean).rename(name)

    def var(self):
        name = "RollingVar({},{})".format(self.name, self.window)
        return self.agg(np.var).rename(name)

    def median(self):
        name = "RollingMedian({},{})".format(self.name, self.window)
        return self.agg(np.median).rename(name)

    def std(self):
        name = "RollingSD({},{})".format(self.name, self.window)
        return self.var().sqrt().rename(name)

    def min(self):
        name = "RollingMin({},{})".format(self.name, self.window)
        return self.agg(np.min).rename(name)

    def max(self):
        name = "RollingMax({},{})".format(self.name, self.window)
        return self.agg(np.max).rename(name)


class RollingNode(Node):

    def __init__(self, func, name: str = None):
        super().__init__(name)
        self.func = func

    def forward(self):
        rolling = self.inputs[0]
        history = rolling.value
        if len(history) < rolling.warmup:
            return np.nan
        return self.func(history)

    def has_next(self):
        return True


class RollingCount(RollingNode):

    def __init__(self, name: str = None):
        super().__init__(lambda w: (~np.isnan(w)).sum(), name)

    def forward(self):
        rolling = self.inputs[0]
        return self.func(rolling.value)


class EWM(Node):

    def __init__(self,
                 com: float = None,
                 span: float = None,
                 halflife: float = None,
                 alpha: float = None,
                 warmup: int = 0,
                 adjust: bool = True,
                 ignore_na: bool = False,
                 name: str = None):
        super().__init__(name)
        self.com = com
        self.span = span
        self.halflife = halflife

        self.warmup = warmup
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
            self.alpha = 1 - math.exp(math.log(0.5) / halflife)

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

    def has_next(self):
        return True

    def debias_factor(self):
        name = "EWM:DebiasFactor({},{})".format(self.name, self.alpha)
        return DebiasFactor(name)(self)

    def mean(self):
        name = "EWM:Mean({},{})".format(self.name, self.alpha)
        return ExponentialWeightedMovingAverage(name)(self)

    def var(self, bias: bool = False):
        name = "EWM:Var({},{})".format(self.name, self.alpha)
        return ExponentialWeightedMovingVariance(bias, name)(self, self.inputs[0])

    def std(self, bias: bool = False):
        name = "EWM:SD({},{})".format(self.name, self.alpha)
        return self.var(bias).sqrt().rename(name)


class ExponentialWeightedMovingAverage(Node):

    def __init__(self, name: str = None):
        super().__init__(name)

    def forward(self):
        ewm = self.inputs[0]
        history, weights = ewm.value

        x = np.array(history)
        w = np.array(weights)

        # Compute average
        if not ewm.ignore_na:
            mask = ~np.isnan(x)
            w = w[mask[::-1]]
            x = x[mask]

        v = (w[::-1] * x).sum() / w.sum()

        return v if len(x) >= ewm.warmup else np.nan

    def has_next(self):
        return True


class DebiasFactor(Node):

    def __init__(self, name: str = None):
        super().__init__(name)

    def forward(self):
        ewm = self.inputs[0]
        w = np.array(ewm.weights)
        a = w.sum()**2
        b = (w**2).sum()
        return a / (a - b)

    def has_next(self):
        return True


class ExponentialWeightedMovingVariance(Module):

    def __init__(self, bias: bool = False, name: str = None):
        super().__init__(name)
        self.bias = bias
        self.variance = None

    def build(self):
        ewm, node = self.inputs
        t1 = (node**2).ewm(
            alpha=ewm.alpha,
            warmup=ewm.warmup,
            adjust=ewm.adjust,
            ignore_na=ewm.ignore_na
        ).mean()
        t2 = ewm.mean()**2
        biased_variance = t1 - t2

        if self.bias:
            self.variance = biased_variance.rename(self.name)
        else:
            self.variance = ewm.debias_factor()*biased_variance
            self.variance.name = self.name

    def flatten(self):
        return [self.variance]

    def forward(self):
        return self.variance.value

    def has_next(self):
        return True
