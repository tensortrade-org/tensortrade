
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
    if order.type == TradeType.LIMIT and order.price < current_price:
        return None

    price = contain_price(current_price, options)

    if order.type == TradeType.MARKET:
        scale = order.price / max(price, order.price)
        commission = Quantity(order.pair.base, scale * order.size *
                              options.commission, order.path_id)
        size = contain_size(scale * (order.size - commission.size), options)
    else:
        commission = Quantity(order.pair.base, order.size * options.commission, order.path_id)
        size = contain_size(order.size - commission.size, options)

    quantity = Quantity(order.pair.base, size, order.path_id)
    filled_quantity = quantity.convert(quote_wallet.instrument, price)

    base_wallet.withdraw(quantity, "FILL BUY ORDER")
    quote_wallet.deposit(filled_quantity, "BOUGHT {} @ {}".format(order.exchange_pair, price))
    base_wallet.withdraw(commission, "COMMISSION FOR BUY")

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
    if order.type == TradeType.LIMIT and order.price > current_price:
        return None

    price = contain_price(current_price, options)
    commission = Quantity(base_wallet.instrument, order.size * options.commission, order.path_id)
    size = contain_size(order.size, options)

    quantity = Quantity(base_wallet.instrument, size, order.path_id)
    filled_quantity = quantity.convert(quote_wallet.instrument, price)

    quote_wallet.withdraw(filled_quantity, "FILL SELL ORDER")
    base_wallet.deposit(quantity, 'SOLD {} @ {}'.format(order.exchange_pair, price))
    base_wallet.withdraw(quantity, 'COMMISSION FOR SELL')

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
