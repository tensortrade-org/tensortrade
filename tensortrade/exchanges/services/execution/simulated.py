
from tensortrade.base import Clock
from tensortrade.base.exceptions import InsufficientFunds
from tensortrade.wallets import Wallet
from tensortrade.instruments import Quantity
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
                      exchange_id: str,
                      clock: 'Clock') -> 'Trade':
    price = contain_price(current_price, options)

    if order.type == TradeType.LIMIT and order.price < current_price:
        return None

    commission = Quantity(order.pair.base, order.size * options.commission, order.path_id)
    size = contain_size(order.size - commission.size, options)

    if order.type == TradeType.MARKET:
        scale = order.price / price
        size = contain_size(scale * order.size - commission.size, options)

    base_wallet -= commission.reason("COMMISSION FOR BUY")

    try:
        quantity = Quantity(order.pair.base, size, order.path_id)
        base_wallet -= quantity.reason("REMOVE FROM LOCKED TO FILL ORDER")
    except InsufficientFunds:
        balance = base_wallet.locked[order.path_id]
        quantity = Quantity(order.pair.base, balance.size, order.path_id)
        base_wallet -= quantity.reason("REMOVE FROM LOCKED TO FILL ORDER (INSUFFICIENT FUNDS)")

    quote_size = (order.price / price) * (size / price)

    filled_quantity = Quantity(order.pair.quote, quote_size, order.path_id)
    quote_wallet += filled_quantity.reason("BOUGHT @ {} {}".format(price, order.pair))

    trade = Trade(order_id=order.id,
                  exchange_id=exchange_id,
                  step=clock.step,
                  pair=order.pair,
                  side=TradeSide.BUY,
                  trade_type=order.type,
                  quantity=quantity,
                  price=price,
                  commission=commission)

    return trade


def execute_sell_order(order: 'Order',
                       base_wallet: 'Wallet',
                       quote_wallet: 'Wallet',
                       current_price: float,
                       options: 'ExchangeOptions',
                       exchange_id: str,
                       clock: 'Clock') -> 'Trade':
    price = contain_price(current_price, options)

    if order.type == TradeType.LIMIT and order.price > current_price:
        return None

    commission = Quantity(order.pair.base, order.size * options.commission, order.path_id)
    size = contain_size(order.size - commission.size, options)

    try:
        quote_size = (size / price) * (price / order.price)
        quote_quantity = Quantity(order.pair.quote, quote_size, order.path_id)
        quote_wallet -= quote_quantity.reason("REMOVE FROM LOCKED TO FILL ORDER")
    except InsufficientFunds:
        balance = quote_wallet.locked[order.path_id]
        quote_size = balance.size
        remove_quantity = Quantity(order.pair.quote, quote_size, order.path_id)
        quote_wallet -= remove_quantity.reason("REMOVE FROM LOCKED TO FILL ORDER (INSUFFICIENT FUNDS)")

    base_size = (quote_size * price) / (price / order.price)
    quantity = Quantity(order.pair.base, base_size, order.path_id)

    base_wallet += quantity.reason("SOLD @ {} {}".format(price, order.pair))
    base_wallet -= commission.reason("COMMISSION FOR SELL")

    trade = Trade(order_id=order.id,
                  exchange_id=exchange_id,
                  step=clock.step,
                  pair=order.pair,
                  side=TradeSide.SELL,
                  trade_type=order.type,
                  quantity=quantity,
                  price=price,
                  commission=commission)

    return trade


def execute_order(order: 'Order',
                  base_wallet: 'Wallet',
                  quote_wallet: 'Wallet',
                  current_price: float,
                  options: 'Options',
                  exchange_id: str,
                  clock: 'Clock') -> 'Trade':

    if order.is_buy:
        trade = execute_buy_order(
            order=order,
            base_wallet=base_wallet,
            quote_wallet=quote_wallet,
            current_price=current_price,
            options=options,
            exchange_id=exchange_id,
            clock=clock
        )
    elif order.is_sell:
        trade = execute_sell_order(
            order=order,
            base_wallet=base_wallet,
            quote_wallet=quote_wallet,
            current_price=current_price,
            options=options,
            exchange_id=exchange_id,
            clock=clock
        )
    else:
        trade = None

    return trade
