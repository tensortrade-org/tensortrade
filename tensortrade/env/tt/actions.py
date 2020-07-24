
from abc import abstractmethod
from typing import Union, List
from itertools import product


from gym.spaces import Discrete

from tensortrade.env.generic import ActionScheme
from tensortrade.base import Clock
from tensortrade.oms.orders import Broker, Order, TradeSide, TradeType, OrderListener, risk_managed_order


class TensorTradeActionScheme(ActionScheme):

    def __init__(self):
        super().__init__()
        self.portfolio = None
        self.broker = Broker()

    def set_portfolio(self, portfolio):
        self.portfolio = portfolio

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



class SimpleOrders(ActionScheme):
    """A discrete action scheme that determines actions based on a list of
    trading pairs, order criteria, and trade sizes."""

    def __init__(self,
                 criteria: Union[List['OrderCriteria'], 'OrderCriteria'] = None,
                 trade_sizes: Union[List[float], int] = 10,
                 trade_type: TradeType = TradeType.MARKET,
                 durations: Union[List[int], int] = None,
                 order_listener: OrderListener = None):
        """
        Arguments:
            pairs: A list of trading pairs to select from when submitting an order.
            (e.g. TradingPair(BTC, USD), TradingPair(ETH, BTC), etc.)
            criteria: A list of order criteria to select from when submitting an order.
            (e.g. MarketOrder, LimitOrder w/ price, StopLoss, etc.)
            trade_sizes: A list of trade sizes to select from when submitting an order.
            (e.g. '[1, 1/3]' = 100% or 33% of balance is tradeable. '4' = 25%, 50%, 75%, or 100% of balance is tradeable.)
            order_listener (optional): An optional listener for order events executed by this action scheme.
        """
        self.criteria = self.default('criteria', criteria)
        self.trade_sizes = self.default('trade_sizes', trade_sizes)
        self.durations = self.default('durations', durations)
        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

    @property
    def criteria(self) -> List['OrderCriteria']:
        """A list of order criteria to select from when submitting an order.
        (e.g. MarketOrderCriteria, LimitOrderCriteria, StopLossCriteria, CustomCriteria, etc.)
        """
        return self._criteria

    @criteria.setter
    def criteria(self, criteria: Union[List['OrderCriteria'], 'OrderCriteria']):
        self._criteria = criteria if isinstance(criteria, list) else [criteria]

    @property
    def trade_sizes(self) -> List[float]:
        """A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable. '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
        """
        return self._trade_sizes

    @trade_sizes.setter
    def trade_sizes(self, trade_sizes: Union[List[float], int]):
        self._trade_sizes = trade_sizes if isinstance(trade_sizes, list) else [
            (x + 1) / trade_sizes for x in range(trade_sizes)]

    @property
    def durations(self) -> List[int]:
        """A list of durations to select from when submitting an order."""
        return self._durations

    @durations.setter
    def durations(self, durations: Union[List[int], int]):
        self._durations = durations if isinstance(durations, list) else [durations]

    def compile(self):
        self.actions = list(product(self._criteria,
                                    self._trade_sizes,
                                    self._durations,
                                    [TradeSide.BUY, TradeSide.SELL]))
        self.actions = list(product(self.exchange_pairs, self.actions))
        self.actions = [None] + self.actions

        self._action_space = Discrete(len(self.actions))

    def get_order(self, action: int, portfolio: 'Portfolio') -> Order:
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


class ManagedRiskOrders(ActionScheme):
    """A discrete action scheme that determines actions based on managing risk,
       through setting a follow-up stop loss and take profit on every order.
    """

    def __init__(self,
                 stop_loss_percentages: Union[List[float], float] = [0.02, 0.04, 0.06],
                 take_profit_percentages: Union[List[float], float] = [0.01, 0.02, 0.03],
                 trade_sizes: Union[List[float], int] = 10,
                 durations: Union[List[int], int] = None,
                 trade_type: TradeType = TradeType.MARKET,
                 order_listener: OrderListener = None):
        """
        Arguments:
            pairs: A list of trading pairs to select from when submitting an order.
            (e.g. TradingPair(BTC, USD), TradingPair(ETH, BTC), etc.)
            stop_loss_percentages: A list of possible stop loss percentages for each order.
            take_profit_percentages: A list of possible take profit percentages for each order.
            trade_sizes: A list of trade sizes to select from when submitting an order.
            (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable. '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
            order_listener (optional): An optional listener for order events executed by this action scheme.
        """
        super().__init__()
        self.stop_loss_percentages = self.default('stop_loss_percentages', stop_loss_percentages)
        self.take_profit_percentages = self.default(
            'take_profit_percentages', take_profit_percentages)
        self.trade_sizes = self.default('trade_sizes', trade_sizes)
        self.durations = self.default('durations', durations)
        self._trade_type = self.default('trade_type', trade_type)
        self._order_listener = self.default('order_listener', order_listener)

    @property
    def stop_loss_percentages(self) -> List[float]:
        """A list of order percentage losses to select a stop loss from when submitting an order.
        (e.g. 0.01 = sell if price drops 1%, 0.15 = 15%, etc.)
        """
        return self._stop_loss_percentages

    @stop_loss_percentages.setter
    def stop_loss_percentages(self, stop_loss_percentages: Union[List[float], float]):
        self._stop_loss_percentages = stop_loss_percentages if isinstance(
            stop_loss_percentages, list) else [stop_loss_percentages]

    @property
    def take_profit_percentages(self) -> List[float]:
        """A list of order percentage gains to select a take profit from when submitting an order.
        (e.g. 0.01 = sell if price rises 1%, 0.15 = 15%, etc.)
        """
        return self._take_profit_percentages

    @take_profit_percentages.setter
    def take_profit_percentages(self, take_profit_percentages: Union[List[float], float]):
        self._take_profit_percentages = take_profit_percentages if isinstance(
            take_profit_percentages, list) else [take_profit_percentages]

    @property
    def trade_sizes(self) -> List[float]:
        """A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable. '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
        """
        return self._trade_sizes

    @trade_sizes.setter
    def trade_sizes(self, trade_sizes: Union[List[float], int]):
        self._trade_sizes = trade_sizes if isinstance(trade_sizes, list) else [
            (x + 1) / trade_sizes for x in range(trade_sizes)]

    @property
    def durations(self) -> List[int]:
        """A list of durations to select from when submitting an order."""
        return self._durations

    @durations.setter
    def durations(self, durations: Union[List[int], int]):
        self._durations = durations if isinstance(durations, list) else [durations]

    def compile(self):
        self.actions = list(product(self._stop_loss_percentages,
                                    self._take_profit_percentages,
                                    self._trade_sizes,
                                    self._durations,
                                    [TradeSide.BUY, TradeSide.SELL]))
        self.actions = list(product(self.exchange_pairs, self.actions))
        self.actions = [None] + self.actions

        self._action_space = Discrete(len(self.actions))

    def get_order(self, action: int, portfolio: 'Portfolio') -> Order:

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

    def reset(self):
        pass


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
