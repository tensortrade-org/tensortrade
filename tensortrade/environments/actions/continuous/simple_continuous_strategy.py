from typing import Union

from tensortrade.environments.actions import ActionStrategy, TradeType
from tensortrade.exchanges import AssetExchange


class SimpleContinuousStrategy(ActionStrategy):
    """Simple continuous strategy, that executes trades on a continuous basis."""

    def __init__(self, base_symbol: str = 'USD', asset_symbol: str = 'BTC'):
        """
            # Arguments
                base_symbol: optional the symbol that defines the notional value
                asset_symbol: optional the asset symbol
        """

        super().__init__(action_space_shape=(1, 1), continuous_action_space=True)

        self.base_symbol = base_symbol
        self.asset_symbol = asset_symbol

    def get_trade(self, action: Union[int, tuple], exchange: AssetExchange):
        """Suggest a trade to the trading environment

            # Arguments
                action: optional number of timesteps used to calculate the trade amount
                exchange: the AssetExchange

            # Returns
                the asset symbol, the type of trade, amount and price

        """
        action_type, trade_amount = action
        trade_type = TradeType(int(action_type * 3))

        current_price = exchange.current_price(
            symbol=self.asset_symbol, output_symbol=self.base_symbol)
        commission_percent = exchange.commission_percent
        base_precision = exchange.base_precision
        asset_precision = exchange.asset_precision

        price = current_price
        amount_to_trade = 0

        if trade_type == TradeType.BUY:
            balance = exchange.balance(self.base_symbol)
            price_adjustment = 1 + (commission_percent / 100)
            price = round(current_price * price_adjustment, base_precision)
            amount_to_trade = round(balance * trade_amount / price, asset_precision)

        elif trade_type == TradeType.SELL:
            portfolio = exchange.portfolio()
            price_adjustment = 1 + (commission_percent / 100)
            price = round(current_price * price_adjustment, base_precision)
            amount_to_trade = round(portfolio.get(self.asset_symbol, 0)
                                    * trade_amount, asset_precision)

        return self.asset_symbol, trade_type, amount_to_trade, price
