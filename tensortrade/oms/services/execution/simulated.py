
from decimal import Decimal

from tensortrade.core import Clock
from tensortrade.oms.wallets import Wallet
from tensortrade.oms.exchanges import ExchangeOptions
from tensortrade.oms.instruments import Instrument, Quantity
from tensortrade.oms.orders import Order, Trade, TradeType, TradeSide, Position


def execute_buy_order(order: 'Order',
                      base_wallet: 'Wallet',
                      quote_wallet: 'Wallet',
                      current_price: float,
                      options: 'ExchangeOptions',
                      clock: 'Clock') -> 'Trade':
    """Executes a buy order on the exchange.

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """
    if order.type == TradeType.LIMIT and order.price < current_price:
        return None

    filled = order.remaining.contain(order.exchange_pair)

    if order.type == TradeType.MARKET:
        scale = order.price / max(current_price, order.price)
        filled = scale * filled

    commission = options.commission * filled
    quantity = filled - commission

    if commission.size < Decimal(10)**-quantity.instrument.precision:
        order.cancel("COMMISSION IS LESS THAN PRECISION.")
        return None

    # Todo: Fill Order with Position-Data got by Broker
    if quantity.instrument == order.exchange_pair.pair.base:
        instrument = order.exchange_pair.pair.quote
        converted_size = quantity.size / order.price
    else:
        instrument = order.exchange_pair.pair.base
        converted_size = quantity.size * order.price
    converted = Quantity(instrument, converted_size, quantity.path_id).quantize()
    position = Position(id = order.id, # to be replaced by borker-id
                        order = order,
                        quantity = converted,
                        )
    quote_wallet.add_position(position)

    transfer = Wallet.transfer(
        source=base_wallet,
        target=quote_wallet,
        quantity=quantity,
        commission=commission,
        exchange_pair=order.exchange_pair,
        reason="BUY"
    )

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=order.side,
        trade_type=order.type,
        quantity=transfer.quantity,
        price=transfer.price,
        commission=transfer.commission
    )

    return trade


def execute_sell_order(order: 'Order',
                       base_wallet: 'Wallet',
                       quote_wallet: 'Wallet',
                       current_price: float,
                       options: 'ExchangeOptions',
                       clock: 'Clock') -> 'Trade':
    """Executes a sell order on the exchange.

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """
    if order.type == TradeType.LIMIT and order.price < current_price:
        return None

    filled = order.remaining.contain(order.exchange_pair)

    if order.type == TradeType.MARKET:
        scale = order.price / max(current_price, order.price)
        filled = scale * filled

    commission = options.commission * filled
    quantity = filled - commission

    if commission.size < Decimal(10)**-quantity.instrument.precision:
        order.cancel("COMMISSION IS LESS THAN PRECISION.")
        return None

    # Todo: Fill Order with Position-Data got by Broker
    if quantity.instrument == order.exchange_pair.pair.base:
        instrument = order.exchange_pair.pair.quote
        converted_size = quantity.size / order.price
    else:
        instrument = order.exchange_pair.pair.base
        converted_size = quantity.size * order.price
    converted = Quantity(instrument, converted_size, quantity.path_id).quantize()
    position = Position(id = order.id, # to be replaced by borker-id
                        order = order,
                        quantity = converted,
                        )
    quote_wallet.add_position(position)

    transfer = Wallet.transfer(
        source=base_wallet,
        target=quote_wallet,
        quantity=quantity,
        commission=commission,
        exchange_pair=order.exchange_pair,
        reason="SELL"
    )

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=TradeSide.SELL,
        trade_type=order.type,
        quantity=transfer.quantity,
        price=transfer.price,
        commission=transfer.commission
    )

    return trade

def execute_close_order(order: 'Order',
                       base_wallet: 'Wallet',
                       quote_wallet: 'Wallet',
                       current_price: float,
                       options: 'ExchangeOptions',
                       clock: 'Clock') -> 'Trade':
    """Executes a sell order on the exchange.

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """
    if order.type == TradeType.LIMIT and order.price > current_price:
        return None

    filled = order.remaining.contain(order.exchange_pair)

    commission = Quantity(instrument=filled.instrument, size=0)
    quantity = filled - commission

    '''
    # Transfer Funds from Quote Wallet to Base Wallet
    transfer = Wallet.transfer(
        source=quote_wallet,
        target=base_wallet,
        quantity=quantity,
        commission=commission,
        exchange_pair=order.exchange_pair,
        reason="CLOSE"
    )
    '''

    # to be replaced by broker
    position = quote_wallet.get_positions()[order.path_id]
    if position:
        # mark position as closed
        position.is_open = False

    trade = Trade(
        order_id=order.id,
        step=clock.step,
        exchange_pair=order.exchange_pair,
        side=order.side,
        trade_type=order.type,
        quantity=order.quantity,
        price=order.price,
        commission=commission
    )

    return trade


def execute_order(order: 'Order',
                  base_wallet: 'Wallet',
                  quote_wallet: 'Wallet',
                  current_price: float,
                  options: 'Options',
                  clock: 'Clock') -> 'Trade':
    """Executes an order on the exchange.

    Parameters
    ----------
    order : `Order`
        The order that is being filled.
    base_wallet : `Wallet`
        The wallet of the base instrument.
    quote_wallet : `Wallet`
        The wallet of the quote instrument.
    current_price : float
        The current price of the exchange pair.
    options : `ExchangeOptions`
        The exchange options.
    clock : `Clock`
        The clock for the trading process..

    Returns
    -------
    `Trade`
        The executed trade that was made.
    """
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
    elif order.is_close:
        trade = execute_close_order(**kwargs)
    else:
        trade = None

    return trade
