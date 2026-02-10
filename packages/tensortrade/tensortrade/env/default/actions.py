import logging
from abc import abstractmethod
from collections import deque
from itertools import product
from typing import Any

import numpy as np
from gymnasium.spaces import Discrete, Space

from tensortrade.core import Clock
from tensortrade.env.generic import ActionScheme, TradingEnv
from tensortrade.oms.instruments import ExchangePair
from tensortrade.oms.orders import (
    Broker,
    Order,
    OrderListener,
    TradeSide,
    TradeType,
    proportion_order,
    risk_managed_order,
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
        self.portfolio: Portfolio = None
        self.broker: Broker = Broker()

    @property
    def clock(self) -> "Clock":
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
    def clock(self, clock: "Clock") -> None:
        self._clock = clock

        components = [self.portfolio] + self.portfolio.exchanges
        for c in components:
            c.clock = clock
        self.broker.clock = clock

    def perform(self, env: "TradingEnv", action: Any) -> None:
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
                logging.info(f"Step {order.step}: {order.side} {order.quantity}")
                self.broker.submit(order)

        self.broker.update()

    @abstractmethod
    def get_orders(self, action: Any, portfolio: "Portfolio") -> "list[Order]":
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
    """A simple discrete action scheme where the only options are to hold, buy,
    or sell.

    Actions
    -------
    0 : Hold — do nothing
    1 : Buy  — convert cash to asset (no-op if already holding asset)
    2 : Sell — convert asset to cash (no-op if already holding cash)

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    """

    registered_name = "bsh"

    def __init__(self, cash: "Wallet", asset: "Wallet"):
        super().__init__()
        self.cash = cash
        self.asset = asset

        self.listeners = []
        self._position = 0  # 0 = in cash, 1 = in asset

    @property
    def action_space(self):
        return Discrete(3)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":
        order = None

        if action == 1 and self._position == 0:
            # Buy: cash -> asset
            if self.cash.balance.as_float() > 0:
                order = proportion_order(portfolio, self.cash, self.asset, 1.0)
                self._position = 1
        elif action == 2 and self._position == 1 and self.asset.balance.as_float() > 0:
            # Sell: asset -> cash
            order = proportion_order(portfolio, self.asset, self.cash, 1.0)
            self._position = 0

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self._position = 0


# ---------------------------------------------------------------------------
# Group 1: Risk Management
# ---------------------------------------------------------------------------


class TrailingStopBSH(TensorTradeActionScheme):
    """BSH with a trailing stop-loss that auto-sells when the price drops a
    percentage below its peak since entry.

    Actions
    -------
    0 : Hold
    1 : Buy (cash -> asset)
    2 : Sell (asset -> cash)

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    stop_pct : float
        The trailing stop percentage (default 0.05 = 5%).
    """

    registered_name = "trailing-stop-bsh"

    def __init__(self, cash: "Wallet", asset: "Wallet", stop_pct: float = 0.05):
        super().__init__()
        self.cash = cash
        self.asset = asset
        self.stop_pct = stop_pct

        self.listeners: list[OrderListener] = []
        self._position = 0
        self._entry_price: float = 0.0
        self._peak_price: float = 0.0

    @property
    def action_space(self) -> Space:
        return Discrete(3)

    def attach(self, listener: "OrderListener"):
        self.listeners += [listener]
        return self

    def _get_exchange_pair(self) -> "ExchangePair":
        pair = self.cash.instrument / self.asset.instrument
        return ExchangePair(self.cash.exchange, pair)

    def _current_price(self) -> float:
        return float(self._get_exchange_pair().price)

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":
        order = None
        effective_action = action

        if self._position == 1:
            price = self._current_price()
            self._peak_price = max(self._peak_price, price)

            # Check trailing stop
            if price <= self._peak_price * (1 - self.stop_pct):
                effective_action = 2  # Force sell

        if effective_action == 1 and self._position == 0:
            if self.cash.balance.as_float() > 0:
                order = proportion_order(portfolio, self.cash, self.asset, 1.0)
                self._position = 1
                price = self._current_price()
                self._entry_price = price
                self._peak_price = price
        elif (
            effective_action == 2
            and self._position == 1
            and self.asset.balance.as_float() > 0
        ):
            order = proportion_order(portfolio, self.asset, self.cash, 1.0)
            self._position = 0
            self._entry_price = 0.0
            self._peak_price = 0.0

        for listener in self.listeners:
            listener.on_action(effective_action)

        return [order]

    def reset(self):
        super().reset()
        self._position = 0
        self._entry_price = 0.0
        self._peak_price = 0.0


class BracketBSH(TensorTradeActionScheme):
    """BSH with bracket orders: auto-sells when price hits stop-loss or
    take-profit levels relative to the entry price.

    Actions
    -------
    0 : Hold
    1 : Buy (cash -> asset)
    2 : Sell (asset -> cash, also serves as manual emergency exit)

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    stop_loss_pct : float
        Stop-loss percentage below entry (default 0.05 = 5%).
    take_profit_pct : float
        Take-profit percentage above entry (default 0.10 = 10%).
    """

    registered_name = "bracket-bsh"

    def __init__(
        self,
        cash: "Wallet",
        asset: "Wallet",
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.10,
    ):
        super().__init__()
        self.cash = cash
        self.asset = asset
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.listeners: list[OrderListener] = []
        self._position = 0
        self._entry_price: float = 0.0

    @property
    def action_space(self) -> Space:
        return Discrete(3)

    def attach(self, listener: "OrderListener"):
        self.listeners += [listener]
        return self

    def _get_exchange_pair(self) -> "ExchangePair":
        pair = self.cash.instrument / self.asset.instrument
        return ExchangePair(self.cash.exchange, pair)

    def _current_price(self) -> float:
        return float(self._get_exchange_pair().price)

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":
        order = None
        effective_action = action

        if self._position == 1:
            price = self._current_price()
            # Check stop-loss
            if price <= self._entry_price * (
                1 - self.stop_loss_pct
            ) or price >= self._entry_price * (1 + self.take_profit_pct):
                effective_action = 2

        if effective_action == 1 and self._position == 0:
            if self.cash.balance.as_float() > 0:
                order = proportion_order(portfolio, self.cash, self.asset, 1.0)
                self._position = 1
                self._entry_price = self._current_price()
        elif (
            effective_action == 2
            and self._position == 1
            and self.asset.balance.as_float() > 0
        ):
            order = proportion_order(portfolio, self.asset, self.cash, 1.0)
            self._position = 0
            self._entry_price = 0.0

        for listener in self.listeners:
            listener.on_action(effective_action)

        return [order]

    def reset(self):
        super().reset()
        self._position = 0
        self._entry_price = 0.0


class DrawdownBudgetBSH(TensorTradeActionScheme):
    """BSH with a drawdown budget: force-sells and locks out new buys when
    the portfolio's drawdown from peak equity exceeds a threshold.

    Actions
    -------
    0 : Hold
    1 : Buy (blocked during lockout)
    2 : Sell

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    max_drawdown_pct : float
        Maximum allowed drawdown from equity peak (default 0.10 = 10%).
    lockout_steps : int
        Number of steps to block buys after a drawdown breach (default 10).
    """

    registered_name = "drawdown-budget-bsh"

    def __init__(
        self,
        cash: "Wallet",
        asset: "Wallet",
        max_drawdown_pct: float = 0.10,
        lockout_steps: int = 10,
    ):
        super().__init__()
        self.cash = cash
        self.asset = asset
        self.max_drawdown_pct = max_drawdown_pct
        self.lockout_steps = lockout_steps

        self.listeners: list[OrderListener] = []
        self._position = 0
        self._equity_peak: float = 0.0
        self._lockout_remaining: int = 0

    @property
    def action_space(self) -> Space:
        return Discrete(3)

    def attach(self, listener: "OrderListener"):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":
        order = None
        effective_action = action

        net_worth = portfolio.net_worth
        if net_worth is not None:
            self._equity_peak = max(self._equity_peak, net_worth)

        # Check drawdown breach
        if (
            self._equity_peak > 0
            and net_worth is not None
            and (self._equity_peak - net_worth) / self._equity_peak
            > self.max_drawdown_pct
        ):
            if self._position == 1:
                effective_action = 2  # Force sell
            self._lockout_remaining = self.lockout_steps

        # During lockout, suppress buys
        if self._lockout_remaining > 0 and effective_action == 1:
            effective_action = 0  # Hold instead

        # Decrement lockout after check
        if self._lockout_remaining > 0:
            self._lockout_remaining -= 1

        if effective_action == 1 and self._position == 0:
            if self.cash.balance.as_float() > 0:
                order = proportion_order(portfolio, self.cash, self.asset, 1.0)
                self._position = 1
        elif (
            effective_action == 2
            and self._position == 1
            and self.asset.balance.as_float() > 0
        ):
            order = proportion_order(portfolio, self.asset, self.cash, 1.0)
            self._position = 0

        for listener in self.listeners:
            listener.on_action(effective_action)

        return [order]

    def reset(self):
        super().reset()
        self._position = 0
        self._equity_peak = 0.0
        self._lockout_remaining = 0


# ---------------------------------------------------------------------------
# Group 2: Anti-Whipsaw (inherit from BSH)
# ---------------------------------------------------------------------------


class CooldownBSH(BSH):
    """BSH with a cooldown period after selling: buy signals are suppressed
    for ``cooldown_steps`` steps after a sell.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    cooldown_steps : int
        Number of steps to block buys after a sell (default 5).
    """

    registered_name = "cooldown-bsh"

    def __init__(self, cash: "Wallet", asset: "Wallet", cooldown_steps: int = 5):
        super().__init__(cash, asset)
        self.cooldown_steps = cooldown_steps
        self._cooldown_remaining: int = 0

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":
        effective_action = action

        # Suppress buys during cooldown
        if self._cooldown_remaining > 0 and effective_action == 1:
            effective_action = 0

        # Decrement cooldown after check
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # Detect sell to start cooldown (position 1 → 0)
        was_holding = self._position == 1

        orders = super().get_orders(effective_action, portfolio)

        # If we just sold, start cooldown
        if was_holding and self._position == 0:
            self._cooldown_remaining = self.cooldown_steps

        return orders

    def reset(self):
        super().reset()
        self._cooldown_remaining = 0


class HoldMinimumBSH(BSH):
    """BSH with a minimum hold period: sell signals are suppressed for
    ``min_hold_steps`` steps after a buy.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    min_hold_steps : int
        Number of steps to block sells after a buy (default 5).
    """

    registered_name = "hold-minimum-bsh"

    def __init__(self, cash: "Wallet", asset: "Wallet", min_hold_steps: int = 5):
        super().__init__(cash, asset)
        self.min_hold_steps = min_hold_steps
        self._hold_remaining: int = 0

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":
        effective_action = action

        # Suppress sells during hold minimum
        if self._hold_remaining > 0 and effective_action == 2:
            effective_action = 0

        # Decrement hold timer after check
        if self._hold_remaining > 0:
            self._hold_remaining -= 1

        # Detect buy to start hold timer (position 0 → 1)
        was_cash = self._position == 0

        orders = super().get_orders(effective_action, portfolio)

        # If we just bought, start hold timer
        if was_cash and self._position == 1:
            self._hold_remaining = self.min_hold_steps

        return orders

    def reset(self):
        super().reset()
        self._hold_remaining = 0


class ConfirmationBSH(BSH):
    """BSH with confirmation: buy/sell signals must repeat for
    ``confirmation_steps`` consecutive steps before executing.

    A hold (action=0) or change in signal resets the counter.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    confirmation_steps : int
        Consecutive identical signals required to act (default 3).
    """

    registered_name = "confirmation-bsh"

    def __init__(self, cash: "Wallet", asset: "Wallet", confirmation_steps: int = 3):
        super().__init__(cash, asset)
        self.confirmation_steps = confirmation_steps
        self._pending_action: int = 0
        self._confirmation_count: int = 0

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":
        effective_action = 0  # Default to hold

        if action == 0:
            # Hold resets counter
            self._pending_action = 0
            self._confirmation_count = 0
        elif action == self._pending_action:
            # Same signal — increment
            self._confirmation_count += 1
        else:
            # New signal — start fresh
            self._pending_action = action
            self._confirmation_count = 1

        if self._confirmation_count >= self.confirmation_steps:
            effective_action = self._pending_action
            # Reset after executing
            self._pending_action = 0
            self._confirmation_count = 0

        return super().get_orders(effective_action, portfolio)

    def reset(self):
        super().reset()
        self._pending_action = 0
        self._confirmation_count = 0


# ---------------------------------------------------------------------------
# Group 3: Position Sizing
# ---------------------------------------------------------------------------


class ScaledEntryBSH(TensorTradeActionScheme):
    """BSH variant with DCA-style scaled entries and exits.

    Actions
    -------
    0 : Hold
    1 : Buy one tranche (equal DCA portions)
    2 : Sell all
    3 : Sell one tranche

    Note: Incompatible with PBR (Discrete(4)). Use SimpleProfit or
    RiskAdjustedReturns instead.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    num_tranches : int
        Number of tranches for scaling in/out (default 3).
    """

    registered_name = "scaled-entry-bsh"

    def __init__(self, cash: "Wallet", asset: "Wallet", num_tranches: int = 3):
        super().__init__()
        self.cash = cash
        self.asset = asset
        self.num_tranches = num_tranches

        self.listeners: list[OrderListener] = []
        self._position = 0  # 0=cash, 1=any asset
        self._tranches_in: int = 0

    @property
    def action_space(self) -> Space:
        return Discrete(4)

    def attach(self, listener: "OrderListener"):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":
        order = None

        if action == 1 and self._tranches_in < self.num_tranches:
            # Buy one tranche
            remaining = self.num_tranches - self._tranches_in
            proportion = 1.0 / remaining
            if self.cash.balance.as_float() > 0:
                order = proportion_order(portfolio, self.cash, self.asset, proportion)
                self._tranches_in += 1
                self._position = 1

        elif action == 2 and self._tranches_in > 0:
            # Sell all
            if self.asset.balance.as_float() > 0:
                order = proportion_order(portfolio, self.asset, self.cash, 1.0)
                self._tranches_in = 0
                self._position = 0

        elif action == 3 and self._tranches_in > 0:
            # Sell one tranche
            proportion = 1.0 / self._tranches_in
            if self.asset.balance.as_float() > 0:
                order = proportion_order(portfolio, self.asset, self.cash, proportion)
                self._tranches_in -= 1
                if self._tranches_in == 0:
                    self._position = 0

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self._position = 0
        self._tranches_in = 0


class PartialTakeProfitBSH(TensorTradeActionScheme):
    """BSH variant that allows partial position exits for taking profit
    while keeping exposure.

    Actions
    -------
    0 : Hold
    1 : Buy all (cash -> asset)
    2 : Sell partial (first_sell_proportion of remaining position)
    3 : Sell all (full exit)

    Note: Incompatible with PBR (Discrete(4)). Use SimpleProfit or
    RiskAdjustedReturns instead.

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    first_sell_proportion : float
        Proportion to sell on first partial exit (default 0.5).
    """

    registered_name = "partial-tp-bsh"

    def __init__(
        self, cash: "Wallet", asset: "Wallet", first_sell_proportion: float = 0.5
    ):
        super().__init__()
        self.cash = cash
        self.asset = asset
        self.first_sell_proportion = first_sell_proportion

        self.listeners: list[OrderListener] = []
        self._position = 0  # 0=cash, 1=full, 2=half

    @property
    def action_space(self) -> Space:
        return Discrete(4)

    def attach(self, listener: "OrderListener"):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":
        order = None

        if action == 1 and self._position == 0:
            # Buy all
            if self.cash.balance.as_float() > 0:
                order = proportion_order(portfolio, self.cash, self.asset, 1.0)
                self._position = 1

        elif action == 2 and self._position == 1:
            # Sell partial from full position
            if self.asset.balance.as_float() > 0:
                order = proportion_order(
                    portfolio, self.asset, self.cash, self.first_sell_proportion
                )
                self._position = 2

        elif action == 2 and self._position == 2:
            # Sell remaining from half position
            if self.asset.balance.as_float() > 0:
                order = proportion_order(portfolio, self.asset, self.cash, 1.0)
                self._position = 0

        elif action == 3 and self._position > 0 and self.asset.balance.as_float() > 0:
            # Sell all from any position
            order = proportion_order(portfolio, self.asset, self.cash, 1.0)
            self._position = 0

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self._position = 0


class VolatilitySizedBSH(TensorTradeActionScheme):
    """BSH variant that sizes buy orders based on recent price volatility.

    Position size is ``target_risk / realized_volatility``, clamped to
    ``[min_size, max_size]``. Sell is always 100% of position.

    Actions
    -------
    0 : Hold
    1 : Buy (volatility-sized)
    2 : Sell (all)

    Parameters
    ----------
    cash : `Wallet`
        The wallet to hold funds in the base instrument.
    asset : `Wallet`
        The wallet to hold funds in the quote instrument.
    window : int
        Lookback window for volatility calculation (default 20).
    target_risk : float
        Target risk per trade as a fraction (default 0.02).
    min_size : float
        Minimum position size as proportion (default 0.1).
    max_size : float
        Maximum position size as proportion (default 1.0).
    """

    registered_name = "volatility-bsh"

    def __init__(
        self,
        cash: "Wallet",
        asset: "Wallet",
        window: int = 20,
        target_risk: float = 0.02,
        min_size: float = 0.1,
        max_size: float = 1.0,
    ):
        super().__init__()
        self.cash = cash
        self.asset = asset
        self.window = window
        self.target_risk = target_risk
        self.min_size = min_size
        self.max_size = max_size

        self.listeners: list[OrderListener] = []
        self._position = 0
        self._price_buffer: deque[float] = deque(maxlen=window + 1)

    @property
    def action_space(self) -> Space:
        return Discrete(3)

    def attach(self, listener: "OrderListener"):
        self.listeners += [listener]
        return self

    def _get_exchange_pair(self) -> "ExchangePair":
        pair = self.cash.instrument / self.asset.instrument
        return ExchangePair(self.cash.exchange, pair)

    def _current_price(self) -> float:
        return float(self._get_exchange_pair().price)

    def _compute_volatility(self) -> float:
        """Compute realized volatility from log returns."""
        if len(self._price_buffer) < 2:
            return 0.0
        prices = np.array(self._price_buffer)
        log_returns = np.diff(np.log(prices))
        if len(log_returns) == 0:
            return 0.0
        return float(np.std(log_returns))

    def perform(self, env: "TradingEnv", action: int) -> None:
        """Override perform to record price each step."""
        price = self._current_price()
        self._price_buffer.append(price)

        orders = self.get_orders(action, self.portfolio)
        for order in orders:
            if order:
                logging.info(f"Step {order.step}: {order.side} {order.quantity}")
                self.broker.submit(order)
        self.broker.update()

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":
        order = None
        effective_action = action

        if effective_action == 1 and self._position == 0:
            vol = self._compute_volatility()
            if vol > 0:
                position_size = self.target_risk / vol
            else:
                position_size = self.max_size
            position_size = float(np.clip(position_size, self.min_size, self.max_size))

            if self.cash.balance.as_float() > 0:
                order = proportion_order(
                    portfolio, self.cash, self.asset, position_size
                )
                self._position = 1

        elif (
            effective_action == 2
            and self._position == 1
            and self.asset.balance.as_float() > 0
        ):
            order = proportion_order(portfolio, self.asset, self.cash, 1.0)
            self._position = 0

        for listener in self.listeners:
            listener.on_action(effective_action)

        return [order]

    def reset(self):
        super().reset()
        self._position = 0
        self._price_buffer.clear()


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

    def __init__(
        self,
        criteria: "list[OrderCriteria] | OrderCriteria" = None,
        trade_sizes: "list[float] | int" = 10,
        durations: "list[int] | int" = None,
        trade_type: "TradeType" = TradeType.MARKET,
        order_listener: "OrderListener" = None,
        min_order_pct: float = 0.02,
        min_order_abs: float = 0.00,
    ) -> None:
        super().__init__()
        self.min_order_pct = min_order_pct
        self.min_order_abs = min_order_abs
        criteria = self.default("criteria", criteria)
        self.criteria = criteria if isinstance(criteria, list) else [criteria]

        trade_sizes = self.default("trade_sizes", trade_sizes)
        if isinstance(trade_sizes, list):
            self.trade_sizes = trade_sizes
        else:
            self.trade_sizes = [(x + 1) / trade_sizes for x in range(trade_sizes)]

        durations = self.default("durations", durations)
        self.durations = durations if isinstance(durations, list) else [durations]

        self._trade_type = self.default("trade_type", trade_type)
        self._order_listener = self.default("order_listener", order_listener)

        self._action_space = None
        self.actions = None

    @property
    def action_space(self) -> Space:
        if not self._action_space:
            self.actions = product(
                self.criteria,
                self.trade_sizes,
                self.durations,
                [TradeSide.BUY, TradeSide.SELL],
            )
            self.actions = list(self.actions)
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":

        if action == 0:
            return []

        (ep, (criteria, proportion, duration, side)) = self.actions[action]

        instrument = side.instrument(ep.pair)
        wallet = portfolio.get_wallet(ep.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = balance * proportion
        size = min(balance, size)

        quantity = (size * instrument).quantize()

        if (
            size < 10**-instrument.precision
            or size < self.min_order_pct * portfolio.net_worth
            or size < self.min_order_abs
        ):
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
            portfolio=portfolio,
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

    def __init__(
        self,
        stop: "list[float] | None" = None,
        take: "list[float] | None" = None,
        trade_sizes: "list[float] | int" = 10,
        durations: "list[int] | int" = None,
        trade_type: "TradeType" = TradeType.MARKET,
        order_listener: "OrderListener" = None,
        min_order_pct: float = 0.02,
        min_order_abs: float = 0.00,
    ) -> None:
        super().__init__()
        self.min_order_pct = min_order_pct
        self.min_order_abs = min_order_abs
        if stop is None:
            stop = [0.02, 0.04, 0.06]
        if take is None:
            take = [0.01, 0.02, 0.03]
        self.stop = self.default("stop", stop)
        self.take = self.default("take", take)

        trade_sizes = self.default("trade_sizes", trade_sizes)
        if isinstance(trade_sizes, list):
            self.trade_sizes = trade_sizes
        else:
            self.trade_sizes = [(x + 1) / trade_sizes for x in range(trade_sizes)]

        durations = self.default("durations", durations)
        self.durations = durations if isinstance(durations, list) else [durations]

        self._trade_type = self.default("trade_type", trade_type)
        self._order_listener = self.default("order_listener", order_listener)

        self._action_space = None
        self.actions = None

    @property
    def action_space(self) -> "Space":
        if not self._action_space:
            self.actions = product(
                self.stop,
                self.take,
                self.trade_sizes,
                self.durations,
                [TradeSide.BUY, TradeSide.SELL],
            )
            self.actions = list(self.actions)
            self.actions = list(product(self.portfolio.exchange_pairs, self.actions))
            self.actions = [None] + self.actions

            self._action_space = Discrete(len(self.actions))
        return self._action_space

    def get_orders(self, action: int, portfolio: "Portfolio") -> "list[Order]":

        if action == 0:
            return []

        (ep, (stop, take, proportion, duration, side)) = self.actions[action]

        side = TradeSide(side)

        instrument = side.instrument(ep.pair)
        wallet = portfolio.get_wallet(ep.exchange.id, instrument=instrument)

        balance = wallet.balance.as_float()
        size = balance * proportion
        size = min(balance, size)
        quantity = (size * instrument).quantize()

        if (
            size < 10**-instrument.precision
            or size < self.min_order_pct * portfolio.net_worth
            or size < self.min_order_abs
        ):
            return []

        params = {
            "side": side,
            "exchange_pair": ep,
            "price": ep.price,
            "quantity": quantity,
            "down_percent": stop,
            "up_percent": take,
            "portfolio": portfolio,
            "trade_type": self._trade_type,
            "end": self.clock.step + duration if duration else None,
        }

        order = risk_managed_order(**params)

        if self._order_listener is not None:
            order.attach(self._order_listener)

        return [order]


_registry = {
    "bsh": BSH,
    "trailing-stop-bsh": TrailingStopBSH,
    "bracket-bsh": BracketBSH,
    "drawdown-budget-bsh": DrawdownBudgetBSH,
    "cooldown-bsh": CooldownBSH,
    "hold-minimum-bsh": HoldMinimumBSH,
    "confirmation-bsh": ConfirmationBSH,
    "scaled-entry-bsh": ScaledEntryBSH,
    "partial-tp-bsh": PartialTakeProfitBSH,
    "volatility-bsh": VolatilitySizedBSH,
    "simple": SimpleOrders,
    "managed-risk": ManagedRiskOrders,
}


def get(identifier: str) -> "ActionScheme":
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
    if identifier not in _registry:
        raise KeyError(
            f"Identifier {identifier} is not associated with any `ActionScheme`."
        )
    return _registry[identifier]()
