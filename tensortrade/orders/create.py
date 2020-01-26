# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License


from tensortrade.orders import Order, OrderSpec
from tensortrade.orders.criteria import Stop, Limit
from tensortrade.trades import TradeSide, TradeType


def market_order(step: int,
                 side: 'TradeSide',
                 pair: 'TradingPair',
                 price: float,
                 size: float,
                 portfolio: 'Portfolio'):
    instrument = pair.base if side == TradeSide.BUY else pair.quote
    order = Order(step=step,
                  side=side,
                  trade_type=TradeType.MARKET,
                  pair=pair,
                  price=price,
                  quantity=(size * instrument),
                  portfolio=portfolio
                  )

    return order


def limit_order(step: int,
                side: 'TradeSide',
                pair: 'TradingPair',
                price: float,
                size: float,
                portfolio: 'Portfolio',
                ttl_in_seconds: int = None,
                ttl_in_steps: int = None):
    instrument = pair.base if side == TradeSide.BUY else pair.quote
    order = Order(step=step,
                  side=side,
                  trade_type=TradeType.LIMIT,
                  pair=pair,
                  price=price,
                  quantity=(size * instrument),
                  ttl_in_seconds=ttl_in_seconds,
                  ttl_in_steps=ttl_in_steps,
                  portfolio=portfolio
                  )

    return order


def hidden_limit_order(step: int,
                       side: 'TradeSide',
                       pair: 'TradingPair',
                       price: float,
                       size: float,
                       portfolio: 'Portfolio',
                       ttl_in_seconds: int = None,
                       ttl_in_steps: int = None):
    instrument = pair.base if side == TradeSide.BUY else pair.quote
    order = Order(step=step,
                  side=side,
                  trade_type=TradeType.MARKET,
                  pair=pair,
                  price=price,
                  quantity=(size * instrument),
                  ttl_in_seconds=ttl_in_seconds,
                  ttl_in_steps=ttl_in_steps,
                  portfolio=portfolio,
                  criteria=Limit(limit_price=price)
                  )

    return order


def risk_managed_order(step: int,
                       side: 'TradeSide',
                       trade_type: 'TradeType',
                       pair: 'TradingPair',
                       price: float,
                       size: float,
                       down_percent: float,
                       up_percent: float,
                       portfolio: 'Portfolio',
                       ttl_in_seconds: int = None,
                       ttl_in_steps: int = None):
    instrument = pair.base if side == TradeSide.BUY else pair.quote
    order = Order(step=step,
                  side=side,
                  trade_type=trade_type,
                  pair=pair,
                  price=price,
                  ttl_in_seconds=ttl_in_seconds,
                  ttl_in_steps=ttl_in_steps,
                  quantity=(size * instrument),
                  portfolio=portfolio)

    risk_criteria = Stop("down", down_percent) ^ Stop("up", up_percent)
    risk_management = OrderSpec(side=TradeSide.SELL if side == TradeSide.BUY else TradeSide.BUY,
                                trade_type=TradeType.MARKET,
                                pair=pair,
                                criteria=risk_criteria)

    order += risk_management

    return order
