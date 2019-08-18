from enum import Enum


class TradeType(Enum):
    """A trade type for use within trading environments."""

    HOLD = "hold"
    LIMIT_BUY = "limit_buy"
    MARKET_BUY = "market_buy"
    LIMIT_SELL = "limit_sell"
    MARKET_SELL = "market_sell"

    def is_buy(self) -> bool:
        return self == TradeType.MARKET_BUY or self == TradeType.LIMIT_BUY

    def is_sell(self) -> bool:
        return self == TradeType.MARKET_SELL or self == TradeType.LIMIT_SELL
