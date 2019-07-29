from typing import Dict

from trader.environments import TradeType
from trader.actions import ActionStrategy
from trader.exchanges import AssetExchange


class SimpleDiscreteStrategy(ActionStrategy):
    def __init__(self, n_bins: int, asset_symbol: str = 'BTC', commission_percent: float = 0.3):
        super(SimpleDiscreteStrategy, self).__init__(action_space_shape=n_bins * len(TradeType),
                                                     continuous_action_space=False)

        self.n_bins = n_bins
        self.asset_symbol = asset_symbol

    def _buy_price(self, current_price, commission_percent, base_precision):
        price_adjustment = 1 + (commission_percent / 100)
        return round(current_price * price_adjustment, base_precision)

    def _sell_price(self, current_price, commission_percent, base_precision):
        price_adjustment = 1 + (commission_percent / 100)
        return round(current_price * price_adjustment, base_precision)

    def suggest_trade(self, action: int | tuple, balance: float, assets_held: Dict[str, float], exchange: AssetExchange):
        trade_type = TradeType(action % len(TradeType))
        trade_amount = float(1 / (action % self.n_bins + 1))

        current_price = exchange.current_price(symbol=self.asset_symbol)
        commission_percent = exchange.commission_percent
        base_precision = exchange.base_precision
        asset_precision = exchange.asset_precision

        amount_to_trade = 0
        price = current_price

        if trade_type == TradeType.BUY:
            price = self._buy_price(current_price, commission_percent, base_precision)
            amount_to_trade = round(self.balance * trade_amount / price, self.asset_precision)

        elif trade_type == TradeType.SELL:
            price = self._sell_price(current_price, commission_percent, base_precision)
            amount_to_trade = round(assets_held[self.asset_symbol] * trade_amount, self.asset_precision)

        return self.asset_symbol, trade_type, amount_to_trade, price
