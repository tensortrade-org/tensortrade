

from tensortrade.orders import Order, OrderSpec
from tensortrade.orders.criteria import Stop, Limit
from tensortrade.trades import TradeSide, TradeType


def market_order(side: 'TradeSide',
                 pair: 'TradingPair',
                 price: float,
                 size: float,
                 portfolio: 'Portfolio'):

    order = Order(
        side=side,
        trade_type=TradeType.MARKET,
        pair=pair,
        price=price,
        quantity=(size * pair.base),
        portfolio=portfolio
    )
    return order


def limit_order(side: 'TradeSide',
                pair: 'TradingPair',
                price: float,
                size: float,
                portfolio: 'Portfolio'):

    order = Order(
        side=side,
        trade_type=TradeType.LIMIT,
        pair=pair,
        price=price,
        quantity=(size * pair.base),
        portfolio=portfolio
    )
    return order


def hidden_limit_order(side: 'TradeSide',
                       pair: 'TradingPair',
                       price: float,
                       size: float,
                       portfolio: 'Portfolio'):

    order = Order(
        side=side,
        trade_type=TradeType.MARKET,
        pair=pair,
        price=price,
        quantity=(size * pair.base),
        portfolio=portfolio,
        criteria=Limit(limit_price=price)
    )

    return order


def risk_managed_order(side: 'TradeSide',
                       trade_type: 'TradeType',
                       pair: 'TradingPair',
                       price: float,
                       size: float,
                       down_percent: float,
                       up_percent: float,
                       portfolio: 'Portfolio'):

    order = Order(side=side,
                  trade_type=trade_type,
                  pair=pair,
                  price=price,
                  quantity=(size * pair.base),
                  portfolio=portfolio)

    risk_criteria = Stop("down", down_percent) ^ Stop("up", up_percent)

    risk_management = OrderSpec(side=TradeSide.SELL if side == TradeSide.BUY else TradeSide.BUY,
                                trade_type=TradeType.MARKET,
                                pair=pair,
                                criteria=risk_criteria)

    order += risk_management

    return order

