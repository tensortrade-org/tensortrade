class Trade(object):
    """A trade object for use within trading environments."""

    def __init__(self, symbol: str, trade_type: str, amount: float, price: float):
        """
        Args:
            symbol: The exchange symbol of the asset in the trade (AAPL, ETH/USD, NQ1!, etc).
            trade_type: The type of trade ("limit-buy", "market-sell", "hold", etc).
            amount: The amount of the asset in the trade (shares, satoshis, contracts, etc).
            price: The price paid per asset in terms of the base asset (e.g. 10000 represents $10,000.00 if the `base_asset` is "USD").
        """
        self._symbol = symbol
        self._trade_type = trade_type
        self._amount = amount
        self._price = price

    @property
    def symbol(self):
        """The exchange symbol of the asset in the trade (AAPL, ETH/USD, NQ1!, etc)."""
        return self._symbol

    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol

    @property
    def trade_type(self):
        """The type of trade ("buy", "sell", "hold", etc)."""
        return self._trade_type

    @trade_type.setter
    def trade_type(self, trade_type):
        self._trade_type = trade_type

    @property
    def amount(self):
        """The amount of the asset in the trade (shares, satoshis, contracts, etc)."""
        return self._amount

    @amount.setter
    def amount(self, amount):
        self._amount = amount

    @property
    def price(self):
        """The price paid per asset in terms of the base asset (e.g. 10000 represents $10,000.00 if the `base_asset` is "USD")."""
        return self._price

    @price.setter
    def price(self, price):
        self._price = price

    @property
    def is_buy(self) -> bool:
        return "buy" in self._trade_type

    @property
    def is_sell(self) -> bool:
        return "sell" in self._trade_type
