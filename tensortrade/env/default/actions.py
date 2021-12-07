import logging
from abc import abstractmethod
from itertools import product
from typing import Union, List, Any

from gym.spaces import Space, Discrete

from tensortrade.core import Clock
from tensortrade.env.generic import ActionScheme, TradingEnv
from tensortrade.oms.instruments import ExchangePair
from tensortrade.oms.orders import (
    Broker,
    Order,
    OrderListener,
    OrderSpec,
    proportion_order,
    risk_managed_order,
    TradeSide,
    TradeType
)
from tensortrade.oms.wallets import Portfolio


class TensorTradeActionScheme(ActionScheme):
    """An abstract base class for any `ActionScheme` that wants to be
    compatible with the built in OMS.

    The structure of the action scheme is built to make sure that action space
    can be used with the system, provided that the user defines the methods to
    interpret that action.

    Attributes
    ----------
    portfolio : 'Portfolio'
        The portfolio object to be used in defining actions.
    broker : 'Broker'
        The broker object to be used for placing orders in the OMS.

    Methods
    -------
    perform(env,portfolio)
        Performs the action on the given environment.
    get_orders(action,portfolio)
        Gets the list of orders to be submitted for the given action.
    """

    def __init__(self) -> None:
        super().__init__()
        self.portfolio: 'Portfolio' = None
        self.broker: 'Broker' = Broker()

    @property
    def clock(self) -> 'Clock':
        """The reference clock from the environment. (`Clock`)

        When the clock is set for the we also set the clock for the portfolio
        as well as the exchanges defined in the portfolio.

        Returns
        -------
        `Clock`
            The environment clock.
        """
        return self._clock

    @clock.setter
    def clock(self, clock: 'Clock') -> None:
        self._clock = clock

        components = [self.portfolio] + self.portfolio.exchanges
        for c in components:
            c.clock = clock
        self.broker.clock = clock

    def perform(self, env: 'TradingEnv', action: Any) -> None:
        """Performs the action on the given environment.

        Under the TT action scheme, the subclassed action scheme is expected
        to provide a method for getting a list of orders to be submitted to
        the broker for execution in the OMS.

        Parameters
        ----------
        env : 'TradingEnv'
            The environment to perform the action on.
        action : Any
            The specific action selected from the action space.
        """
        orders = self.get_orders(action, self.portfolio)

        for order in orders:
            if order:
                logging.info('Step {}: {} {}'.format(order.step, order.side, order.quantity))
                self.broker.submit(order)

        self.broker.update()

    @abstractmethod
    def get_orders(self, action: Any, portfolio: 'Portfolio') -> 'List[Order]':
        """Gets the list of orders to be submitted for the given action.

        Parameters
        ----------
        action : Any
            The action to be interpreted.
        portfolio : 'Portfolio'
            The portfolio defined for the environment.

        Returns
        -------
        List[Order]
            A list of orders to be submitted to the broker.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Resets the action scheme."""
        self.portfolio.reset()
        self.broker.reset()


class BSH(TensorTradeActionScheme):
    """A simple discrete action scheme where the only options are to buy, sell,
    or hold.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base intrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    """

    registered_name = "bsh"

    def __init__(self, cash: 'Wallet', asset: 'Wallet'):
        super().__init__()
        self.cash = cash
        self.asset = asset

        self.listeners = []
        self.action = 0

    @property
    def action_space(self):
        return Discrete(2)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'Order':
        order = None

        if abs(action - self.action) > 0:
            src = self.cash if self.action == 0 else self.asset
            tgt = self.asset if self.action == 0 else self.cash

            if src.balance == 0:  # We need to check, regardless of the proposed order, if we have balance in 'src'
                return []  # Otherwise just return an empty order list

            order = proportion_order(portfolio, src, tgt, 1.0)
            self.action = action

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self.action = 0


class SimpleOrders(TensorTradeActionScheme):
    """A discrete action scheme that determines actions based on a list of
    trading pairs, order criteria, and trade sizes.

    Parameters
    ----------
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
    min_order_pct : float
        The minimum value when placing an order, calculated in percent over net_worth.
    min_order_abs : float
        The minimum value when placing an order, calculated in absolute order value.
    """

    def __init__(self,
                 criteria: 'Union[List[OrderCriteria], OrderCriteria]' = None,
                 trade_sizes: 'Union[List[float], int]' = 10,
                 durations: 'Union[List[int], int]' = None,
                 trade_type: 'TradeType' = TradeType.MARKET,
                 order_listener: 'OrderListener' = None,
                 min_order_pct: float = 0.02,
                 min_order_abs: float = 0.00) -> None:
        super().__init__()
        self.min_order_pct = min_order_pct
        self.min_order_abs = min_order_abs
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
            self.actions = product(
                self.criteria,
                self.trade_sizes,
                self.durations,
                [TradeSide.BUY, TradeSide.SELL]
            )
            self.actions = list(self.actions)
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self,
                   action: int,
                   portfolio: 'Portfolio') -> 'List[Order]':

        if action == 0:
            return []

        (ep, (criteria, proportion, duration, side)) = self.actions[action]

        instrument = side.instrument(ep.pair)
        wallet = portfolio.get_wallet(ep.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)

        quantity = (size * instrument).quantize()

        if size < 10 ** -instrument.precision \
                or size < self.min_order_pct * portfolio.net_worth \
                or size < self.min_order_abs:
            return []

        order = Order(
            step=self.clock.step,
            side=side,
            trade_type=self._trade_type,
            exchange_pair=ep,
            price=ep.price,
            quantity=quantity,
            criteria=criteria,
            end=self.clock.step + duration if duration else None,
            portfolio=portfolio
        )

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]


class ManagedRiskOrders(TensorTradeActionScheme):
    """A discrete action scheme that determines actions based on managing risk,
       through setting a follow-up stop loss and take profit on every order.

    Parameters
    ----------
    stop : List[float]
        A list of possible stop loss percentages for each order.
    take : List[float]
        A list of possible take profit percentages for each order.
    trade_sizes : List[float]
        A list of trade sizes to select from when submitting an order.
        (e.g. '[1, 1/3]' = 100% or 33% of balance is tradable.
        '4' = 25%, 50%, 75%, or 100% of balance is tradable.)
    durations : List[int]
        A list of durations to select from when submitting an order.
    trade_type : `TradeType`
        A type of trade to make.
    order_listener : OrderListener
        A callback class to use for listening to steps of the order process.
    min_order_pct : float
        The minimum value when placing an order, calculated in percent over net_worth.
    min_order_abs : float
        The minimum value when placing an order, calculated in absolute order value.
    """

    def __init__(self,
                 stop: 'List[float]' = [0.02, 0.04, 0.06],
                 take: 'List[float]' = [0.01, 0.02, 0.03],
                 trade_sizes: 'Union[List[float], int]' = 10,
                 durations: 'Union[List[int], int]' = None,
                 trade_type: 'TradeType' = TradeType.MARKET,
                 order_listener: 'OrderListener' = None,
                 min_order_pct: float = 0.02,
                 min_order_abs: float = 0.00) -> None:
        super().__init__()
        self.min_order_pct = min_order_pct
        self.min_order_abs = min_order_abs
        self.stop = self.default('stop', stop)
        self.take = self.default('take', take)

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
    def action_space(self) -> 'Space':
        if not self._action_space:
            self.actions = product(
                self.stop,
                self.take,
                self.trade_sizes,
                self.durations,
                [TradeSide.BUY, TradeSide.SELL]
            )
            self.actions = list(self.actions)
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'List[Order]':

        if action == 0:
            return []

        (ep, (stop, take, proportion, duration, side)) = self.actions[action]

        side = TradeSide(side)

        instrument = side.instrument(ep.pair)
        wallet = portfolio.get_wallet(ep.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = (balance * proportion)
        size = min(balance, size)
        quantity = (size * instrument).quantize()

        if size < 10 ** -instrument.precision \
                or size < self.min_order_pct * portfolio.net_worth \
                or size < self.min_order_abs:
            return []

        params = {
            'side': side,
            'exchange_pair': ep,
            'price': ep.price,
            'quantity': quantity,
            'down_percent': stop,
            'up_percent': take,
            'portfolio': portfolio,
            'trade_type': self._trade_type,
            'end': self.clock.step + duration if duration else None
        }

        order = risk_managed_order(**params)

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]


_registry = {
    'bsh': BSH,
    'simple': SimpleOrders,
    'managed-risk': ManagedRiskOrders,
}


def get(identifier: str) -> 'ActionScheme':
    """Gets the `ActionScheme` that matches with the identifier.

    Parameters
    ----------
    identifier : str
        The identifier for the `ActionScheme`.

    Returns
    -------
    'ActionScheme'
        The action scheme associated with the `identifier`.

    Raises
    ------
    KeyError:
        Raised if the `identifier` is not associated with any `ActionScheme`.
    """
    if identifier not in _registry.keys():
        raise KeyError(f"Identifier {identifier} is not associated with any `ActionScheme`.")
    return _registry[identifier]()
