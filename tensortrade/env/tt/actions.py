
from abc import abstractmethod
from typing import Union, List
from itertools import product


from gym.spaces import Space, Discrete

from tensortrade.env.generic import ActionScheme
from tensortrade.base import Clock
from tensortrade.oms.orders import (
    Broker,
    Order,
    TradeSide,
    TradeType,
    OrderListener,
    risk_managed_order
)


class TensorTradeActionScheme(ActionScheme):

    def __init__(self):
        super().__init__()
        self.portfolio = None
        self.broker = Broker()

    @property
    def clock(self) -> Clock:
        return self._clock

    @clock.setter
    def clock(self, clock):
        self._clock = clock

        components = [self.portfolio] + self.portfolio.exchanges
        for c in components:
            c.clock = clock
        self.broker.clock = clock

    def perform(self, env, action):
        orders = self.get_orders(action, self.portfolio)

        if orders:
            if not isinstance(orders, list):
                orders = [orders]

            for order in orders:
                self.broker.submit(order)

        self.broker.update()

    @abstractmethod
    def get_orders(self, action, portfolio):
        raise NotImplementedError()

    def reset(self):
        self.portfolio.reset()
        self.broker.reset()


class SimpleOrders(TensorTradeActionScheme):
    """A discrete action scheme that determines actions based on a list of
    trading pairs, order criteria, and trade sizes.

    Parameters:
    ===========
        criteria : List[OrderCriteria]
            A list of order criteria to select from when submitting an order.
            (e.g. MarketOrder, LimitOrder w/ price, StopLoss, etc.)
        trade_sizes : List[float]
            A list of trade sizes to select from when submitting an order.
            (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable.
            '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
        durations : List[int]
            A list of durations to select from when submitting an order.
        trade_type : TradeType
            A type of trade to make.
        order_listener : OrderListener
            A callback class to use for listening to steps of the order process.
    """

    def __init__(self,
                 criteria: Union[List['OrderCriteria'], 'OrderCriteria'] = None,
                 trade_sizes: Union[List[float], int] = 10,
                 durations: Union[List[int], int] = None,
                 trade_type: TradeType = TradeType.MARKET,
                 order_listener: OrderListener = None):
        super().__init__()
        criteria = self.default('criteria', criteria)
        self.criteria = criteria if isinstance(criteria, list) else [criteria]

        trade_sizes = self.default('trade_sizes', trade_sizes)
        if isinstance(trade_sizes, list):
            self.trade_sizes = trade_sizes
        else:
            self.trade_sizes = [(x + 1) / trade_sizes for x in range(trade_sizes)]

        durations = self.default('durations', durations)
        self.durations = durations if isinstance(durations, list) else [durations]

        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

        self._action_space = None
        self.actions = None

    @property
    def action_space(self) -> Space:
        if not self._action_space:
            self.actions = list(product(self.criteria,
                                        self.trade_sizes,
                                        self.durations,
                                        [TradeSide.BUY, TradeSide.SELL]))
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self, action: int, portfolio: 'Portfolio') -> Order:
        if action == 0:
            return None

        (exchange_pair, (criteria, proportion, duration, side)) = self.actions[action]

        instrument = side.instrument(exchange_pair.pair)
        wallet = portfolio.get_wallet(exchange_pair.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)

        quantity = (size * instrument).quantize()

        if size < 10 ** -instrument.precision:
            return None

        order = Order(
            step=self.clock.step,
            side=side,
            trade_type=self._trade_type,
            exchange_pair=exchange_pair,
            price=exchange_pair.price,
            quantity=quantity,
            criteria=criteria,
            end=self.clock.step + duration if duration else None,
            portfolio=portfolio
        )

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return order


class ManagedRiskOrders(TensorTradeActionScheme):
    """A discrete action scheme that determines actions based on managing risk,
       through setting a follow-up stop loss and take profit on every order.

    Parameters:
    ===========
        stop_loss_percentages : List[float]
            A list of possible stop loss percentages for each order.
        take_profit_percentages : List[float]
            A list of possible take profit percentages for each order.
        trade_sizes : List[float]
            A list of trade sizes to select from when submitting an order.
            (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable.
            '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
        durations : List[int]
            A list of durations to select from when submitting an order.
        trade_type : TradeType
            A type of trade to make.
        order_listener : OrderListener
            A callback class to use for listening to steps of the order process.
    """

    def __init__(self,
                 stop_loss_percentages: Union[List[float], float] = [0.02, 0.04, 0.06],
                 take_profit_percentages: Union[List[float], float] = [0.01, 0.02, 0.03],
                 trade_sizes: Union[List[float], int] = 10,
                 durations: Union[List[int], int] = None,
                 trade_type: TradeType = TradeType.MARKET,
                 order_listener: OrderListener = None):
        super().__init__()
        stop_loss_percentages = self.default('stop_loss_percentages', stop_loss_percentages)
        self.stop_loss_percentages = stop_loss_percentages if isinstance(
            stop_loss_percentages, list) else [stop_loss_percentages]

        take_profit_percentages = self.default('take_profit_percentages', take_profit_percentages)
        self.take_profit_percentages = take_profit_percentages if isinstance(
            take_profit_percentages, list) else [take_profit_percentages]

        trade_sizes = self.default('trade_sizes', trade_sizes)
        if isinstance(trade_sizes, list):
            self.trade_sizes = trade_sizes
        else:
            self.trade_sizes = [(x + 1) / trade_sizes for x in range(trade_sizes)]

        durations = self.default('durations', durations)
        self.durations = durations if isinstance(durations, list) else [durations]

        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

        self._action_space = None
        self.actions = None

    @property
    def action_space(self) -> Space:
        if not self._action_space:
            self.actions = list(product(self.stop_loss_percentages,
                                        self.take_profit_percentages,
                                        self.trade_sizes,
                                        self.durations,
                                        [TradeSide.BUY, TradeSide.SELL]))
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self, action: int, portfolio: 'Portfolio') -> Order:

        if action == 0:
            return None

        (exchange_pair, (stop_loss, take_profit, proportion, duration, side)) = self.actions[action]

        instrument = side.instrument(exchange_pair.pair)
        wallet = portfolio.get_wallet(exchange_pair.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)
        quantity = (size * instrument).quantize()

        if size < 10 ** -exchange_pair.pair.base.precision:
            return None

        params = {
            'step': self.clock.step,
            'side': side,
            'exchange_pair': exchange_pair,
            'price': exchange_pair.price,
            'quantity': quantity,
            'down_percent': stop_loss,
            'up_percent': take_profit,
            'portfolio': portfolio,
            'trade_type': self._trade_type,
            'end': self.clock.step + duration if duration else None
        }

        order = risk_managed_order(**params)

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return order


_registry = {
    'simple': SimpleOrders,
    'managed-risk': ManagedRiskOrders,
}


def get(identifier: str) -> ActionScheme:
    """Gets the `ActionScheme` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `ActionScheme`

    Raises:
        KeyError: if identifier is not associated with any `ActionScheme`
    """
    if identifier not in _registry.keys():
        raise KeyError(f"Identifier {identifier} is not associated with any `ActionScheme`.")

    return _registry[identifier]()
