
from tensortrade import TradingContext
from tensortrade.slippage import SlippageModel
from tensortrade.trades import Trade


class ConcreteSlippageModel(SlippageModel):

    def fill_order(self, trade: Trade, **kwargs) -> Trade:
        pass


config = {
    'base_instrument': 'USD',
    'instruments': 'ETH',
    'slippage': {
        'minimum': 0,
        'maximum': 100
    }
}


def test_injects_context_into_slippage_model():

    with TradingContext(**config):
        model = ConcreteSlippageModel()

        assert hasattr(model.context, 'minimum')
        assert hasattr(model.context, 'maximum')
        assert model.context.minimum == 0
        assert model.context.maximum == 100
        assert model.context['minimum'] == 0
        assert model.context['maximum'] == 100
