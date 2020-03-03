
import pandas as pd

from tensortrade.base import Clock, InsufficientFunds
from tensortrade.wallets import Wallet
from tensortrade.instruments import Quantity, precise
from tensortrade.exchanges import ExchangeOptions
from tensortrade.orders import Order, Trade, TradeType, TradeSide


def contain_price(price: float, options: 'ExchangeOptions') -> float:
    return max(min(price, options.max_trade_price), options.min_trade_price)


def contain_size(size: float, options: 'ExchangeOptions') -> float:
    return max(min(size, options.max_trade_size), options.min_trade_size)


def execute_buy_order(order: 'Order',
                      base_wallet: 'Wallet',
                      quote_wallet: 'Wallet',
                      current_price: float,
                      options: 'ExchangeOptions',
                      clock: 'Clock') -> 'Trade':
    if order.type == TradeType.LIMIT and order.price < current_price:
        return None

    price = contain_price(current_price, options)
    filled_size = order.remaining_size

    if order.type == TradeType.MARKET:
        scale = order.price / max(price, order.price)

        filled_size = scale * order.remaining_size
        filled_size = precise(filled_size, order.pair.base.precision)

    size = contain_size(filled_size, options)
    base_quantity = Quantity(order.pair.base, size, order.path_id)

    commission = base_wallet.withdraw(options.commission * base_quantity, "COMMISSION FOR BUY")
    withdrawn = base_wallet.withdraw(base_quantity - commission, "FILL BUY ORDER")
    converted = withdrawn.convert(order.exchange_pair)

    quote_wallet.deposit(converted, "BOUGHT {} @ {}".format(order.exchange_pair, price))

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=TradeSide.BUY,
        trade_type=order.type,
        quantity=converted,
        price=price,
        commission=commission
    )

    return trade


def execute_sell_order(order: 'Order',
                       base_wallet: 'Wallet',
                       quote_wallet: 'Wallet',
                       current_price: float,
                       options: 'ExchangeOptions',
                       clock: 'Clock') -> 'Trade':
    if order.type == TradeType.LIMIT and order.price > current_price:
        return None

    price = contain_price(current_price, options)
    filled_size = contain_size(order.remaining_size, options)

    base_amount = Quantity(base_wallet.instrument, filled_size, order.path_id)

    quote_amount = base_amount.convert(order.exchange_pair)

    withdrawn = quote_wallet.withdraw(quote_amount, "FILL SELL ORDER")
    converted = withdrawn.convert(order.exchange_pair)
    base_wallet.deposit(converted, 'SOLD {} @ {}'.format(order.exchange_pair, price))

    commission = options.commission * converted
    base_wallet.withdraw(commission, "COMMISSION FOR SELL")

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=TradeSide.SELL,
        trade_type=order.type,
        quantity=withdrawn - commission.convert(order.exchange_pair),
        price=price,
        commission=commission
    )

    return trade


def execute_order(order: 'Order',
                  base_wallet: 'Wallet',
                  quote_wallet: 'Wallet',
                  current_price: float,
                  options: 'Options',
                  clock: 'Clock') -> 'Trade':
    kwargs = {"order": order,
              "base_wallet": base_wallet,
              "quote_wallet": quote_wallet,
              "current_price": current_price,
              "options": options,
              "clock": clock}

    if order.is_buy:
        trade = execute_buy_order(**kwargs)
    elif order.is_sell:
        trade = execute_sell_order(**kwargs)
    else:
        trade = None

    return trade
