from typing import Union

from tensortrade.environments.actions import ActionStrategy, TradeType
from tensortrade.exchanges import AssetExchange


class SimpleDiscreteStrategy(ActionStrategy):
    """Simple discrete strategy, which uses fixed intervals to calculate the trade amount."""

    def __init__(self, n_bins: int = 24, base_symbol: str = 'USD', asset_symbol: str = 'BTC'):
        """
            # Arguments
                n_bins: optional number of timesteps used to calculate the trade amount
                base_symbol: optional the symbol that defines the notional value
                asset_symbol: optional the asset symbol
        """
        super().__init__(action_space_shape=n_bins, continuous_action_space=False)
        self.n_bins = n_bins
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
        trade_type = TradeType(action % len(TradeType))
        trade_amount = float(1 / (action % self.n_bins + 1))

        current_price = exchange.current_price(
            symbol=self.asset_symbol, output_symbol=self.base_symbol)
        commission_percent = exchange.commission_percent
        base_precision = exchange.base_precision

        amount_to_trade = 0
        price = current_price

        if trade_type == TradeType.BUY:
            balance = exchange.balance(self.base_symbol)
            price_adjustment = 1 + (commission_percent / 100)
            price = round(current_price * price_adjustment, base_precision)
            amount_to_trade = round(balance * trade_amount / price, exchange.asset_precision)

        elif trade_type == TradeType.SELL:
            portfolio = exchange.portfolio()
            price_adjustment = 1 + (commission_percent / 100)
            price = round(current_price * price_adjustment, base_precision)
            amount_to_trade = round(portfolio.get(self.asset_symbol, 0) *
                                    trade_amount, exchange.asset_precision)

        return self.asset_symbol, trade_type, amount_to_trade, price
