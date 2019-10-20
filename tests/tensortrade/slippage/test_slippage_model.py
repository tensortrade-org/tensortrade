

from tensortrade import TradingContext
from tensortrade.slippage import SlippageModel
from tensortrade.trades import Trade


class ConcreteSlippageModel(SlippageModel):

    def fill_order(self, trade: Trade, **kwargs) -> Trade:
        pass


config = {
    'base_instrument': 'USD',
    'products': 'ETH',
    'credentials': {
        'api_key': '48hg34wydghi7ef',
        'api_secret_key': '0984hgoe8d7htg'
    }
}


def test_injects_context_into_slippage_model():

    with TradingContext() as tc:
        model = ConcreteSlippageModel()

        assert model.context == tc