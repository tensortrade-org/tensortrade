from typing import Dict

from trader.environments import TradeType
from trader.actions import ActionStrategy
from trader.exchanges import AssetExchange


class SimpleContinuousStrategy(ActionStrategy):
    def __init__(self, asset_symbol: str = 'BTC'):
        super(SimpleContinuousStrategy, self).__init__(action_space_shape=(1, 1),
                                                       continuous_action_space=True)

        self.asset_symbol = asset_symbol

    def _buy_price(self, current_price, commission_percent, base_precision):
        price_adjustment = 1 + (commission_percent / 100)
        return round(current_price * price_adjustment, base_precision)

    def _sell_price(self, current_price, commission_percent, base_precision):
        price_adjustment = 1 + (commission_percent / 100)
        return round(current_price * price_adjustment, base_precision)

    def suggest_trade(self, action: int | tuple, balance: float, assets_held: Dict[str, float], exchange: AssetExchange):
        action_type, trade_amount = action
        trade_type = TradeType(int(action_type * 3))

        current_price = exchange.current_price(symbol=self.asset_symbol)
        commission_percent = exchange.commission_percent
        base_precision = exchange.base_precision
        asset_precision = exchange.asset_precision

        price = current_price
        amount_to_trade = 0

        if trade_type == TradeType.BUY:
            price = self._buy_price(current_price, commission_percent, base_precision)
            amount_to_trade = round(balance * trade_amount / price, asset_precision)

        elif trade_type == TradeType.SELL:
            price = self._sell_price(current_price, commission_percent, base_precision)
            amount_to_trade = round(assets_held[self.asset_symbol] * trade_amount, asset_precision)

        return self.asset_symbol, trade_type, amount_to_trade, price
