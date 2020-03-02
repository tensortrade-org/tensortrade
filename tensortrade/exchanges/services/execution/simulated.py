
from tensortrade.base import Clock
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
                      clock: 'Clock') -> 'Trade':
    if order.type == TradeType.LIMIT and order.price < current_price:
        return None

    price = contain_price(current_price, options)
    locked_quantity = base_wallet.locked.get(order.path_id)
    filled_size = order.remaining_size

    if order.type == TradeType.MARKET:
        scale = order.price / max(price, order.price)
        filled_size = scale * order.remaining_size

    filled_size = min(locked_quantity.size, filled_size)
    commission = Quantity(order.pair.base,
                          filled_size * options.commission,
                          order.path_id)
    size = contain_size(filled_size - commission.size, options)

    base_quantity = Quantity(order.pair.base, size, order.path_id)
    quote_quantity = base_quantity.convert(order.exchange_pair)

    base_wallet.withdraw(base_quantity, "FILL BUY ORDER")
    quote_wallet.deposit(quote_quantity, "BOUGHT {} @ {}".format(order.exchange_pair, price))
    base_wallet.withdraw(commission, "COMMISSION FOR BUY")

    trade = Trade(order_id=order.id,
                  step=clock.step,
                  exchange_pair=order.exchange_pair,
                  side=TradeSide.BUY,
                  trade_type=order.type,
                  quantity=quote_quantity,
                  price=price,
                  commission=commission)

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
    locked_quote_quantity = quote_wallet.locked.get(order.path_id)
    locked_base_quantity = locked_quote_quantity.convert(order.exchange_pair)
    filled_size = min(locked_base_quantity.size, order.remaining_size)

    commission = Quantity(base_wallet.instrument,
                          filled_size * options.commission,
                          order.path_id)
    size = contain_size(filled_size - commission.size, options)

    base_quantity = Quantity(base_wallet.instrument, size, order.path_id)
    quote_quantity = Quantity(quote_wallet.instrument, filled_size, order.path_id)

    quote_wallet.withdraw(quote_quantity, "FILL SELL ORDER")
    base_wallet.deposit(base_quantity, 'SOLD {} @ {}'.format(order.exchange_pair, price))
    base_wallet.withdraw(base_quantity, 'COMMISSION FOR SELL')

    trade = Trade(order_id=order.id,
                  step=clock.step,
                  exchange_pair=order.exchange_pair,
                  side=TradeSide.SELL,
                  trade_type=order.type,
                  quantity=quote_quantity,
                  price=price,
                  commission=commission)

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
