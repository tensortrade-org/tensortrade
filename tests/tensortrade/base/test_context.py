
import pytest
import tensortrade as td

from tensortrade.base.context import TradingContext


def get_context():
    return TradingContext.get_context()


# =============================================================================
# TradingContext tests
# =============================================================================
def test_is_trading_context_class_there():
    assert td.TradingContext


def test_has_config_attribute():
    c = TradingContext()

    assert hasattr(c, 'base_instrument')
    assert hasattr(c, 'products')
    assert hasattr(c, 'credentials')


config = {
    'base_instrument': 'EURO',
    'products': ['BTC', 'ETH'],
    'credentials': {
        'api_key': '48hg34wydghi7ef',
        'api_secret_key': '0984hgoe8d7htg'
    }
}


def test_init():

    c = TradingContext(base_instrument=config['base_instrument'],
                       products=config['products'],
                       credentials=config['credentials'])
    assert c.base_instrument == 'EURO'
    assert c.products == ['BTC', 'ETH']
    assert c.credentials == {'api_key': '48hg34wydghi7ef',
                             'api_secret_key': '0984hgoe8d7htg'}


def test_init_with_kwargs():
    c = TradingContext(**config)
    assert c.base_instrument == 'EURO'
    assert c.products == ['BTC', 'ETH']
    assert c.credentials == {'api_key': '48hg34wydghi7ef',
                             'api_secret_key': '0984hgoe8d7htg'}


def test_init_instance_attr():

    d = {
        **config,
        'apple': 1,
        'banana': 2,
        'carrot': 3
    }

    c = TradingContext(**d)
    assert c.base_instrument == 'EURO'
    assert c.products == ['BTC', 'ETH']
    assert c.credentials == {'api_key': '48hg34wydghi7ef',
                             'api_secret_key': '0984hgoe8d7htg'}

    assert c['apple'] == 1
    assert hasattr(c, 'apple')
    assert c['apple'] == c.apple
    assert c['banana'] == 2
    assert hasattr(c, 'banana')
    assert c['banana'] == c.banana
    assert c['carrot'] == 3
    assert hasattr(c, 'carrot')
    assert c['carrot'] == c.carrot


def test_context_creation():

    with td.TradingContext(**config) as tc1:
        assert tc1.data == config

        with td.TradingContext(**config) as tc2:
            assert TradingContext.get_context() == tc2

        assert TradingContext.get_context() == tc1


def test_get_context_from_tensor_trade_level():
    with td.TradingContext(**config) as tc:
        assert get_context() == tc


def test_context_within_context():

    with td.TradingContext(**config) as tc1:
        assert get_context() == tc1

        with td.TradingContext(**config) as tc2:
            assert get_context() == tc2

        assert get_context() == tc1


def test_context_retains_data_outside_with():

    with td.TradingContext(**config) as tc:
        assert tc.data == config

    assert tc.data == config


def test_no_context_outside_with_statement():
    with pytest.raises(TypeError, match=r"No base on base stack"):
        get_context()
