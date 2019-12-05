from tensortrade.slippage import RandomUniformSlippageModel
from tensortrade.trades import TradeType, Trade


def test_slippage_in_bounds():
    slippage = RandomUniformSlippageModel()
    assert slippage.max_price_slippage_percent == 3.0
    assert slippage.max_amount_slippage_percent == 0.0

def test_slippage_trade():
    """ Make sure the slippage is not zero. """
    trade_2 = Trade(2, "BTC", TradeType.LIMIT_SELL, 100, 1300)
    slippage = RandomUniformSlippageModel(max_amount_slippage_percent=3.0)
    slipped_trade = slippage.fill_order(trade_2, 1300)

    # print(slipped_trade._price, slipped_trade._amount)
    assert slipped_trade.symbol == "BTC"
    assert slipped_trade.trade_type == TradeType.LIMIT_SELL
    assert slipped_trade.amount != trade_2.amount