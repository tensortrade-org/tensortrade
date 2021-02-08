

from tensortrade.core import TradingContext


def get_context():
    return TradingContext.get_context()


# =============================================================================
# TradingContext tests
# =============================================================================
def test_is_trading_context_class_there():
    assert TradingContext


def test_has_config_attribute():
    c = TradingContext({
        "test": True,
        "exchanges": {"test": True},
        "actions": {"test": True},
        "rewards": {"test": True},
    })

    assert hasattr(c, 'shared')


config = {
    'base_instrument': 'EURO',
    'instruments': ['BTC', 'ETH'],
    'credentials': {
        'api_key': '48hg34wydghi7ef',
        'api_secret_key': '0984hgoe8d7htg'
    }
}


def test_init():
    c = TradingContext({"base_instrument": config['base_instrument'],
                        "instruments": config['instruments']})
    assert c.shared.get('base_instrument') == 'EURO'
    assert c.shared.get('instruments') == ['BTC', 'ETH']


def test_init_with_kwargs():
    c = TradingContext(config)
    assert c.shared.get('base_instrument') == 'EURO'
    assert c.shared.get('instruments') == ['BTC', 'ETH']


def test_context_creation():

    with TradingContext(config) as tc1:
        assert tc1.data == config

        with TradingContext(config) as tc2:
            assert TradingContext.get_context() == tc2

        assert TradingContext.get_context() == tc1


def test_get_context_from_tensor_trade_level():
    with TradingContext(config) as tc:
        assert get_context() == tc


def test_context_within_context():

    with TradingContext(config) as tc1:
        assert get_context() == tc1

        with TradingContext(config) as tc2:
            assert get_context() == tc2

        assert get_context() == tc1


def test_context_retains_data_outside_with():

    with TradingContext(config) as tc:
        assert tc.data == config

    assert tc.data == config


def test_create_trading_context_from_json():
    path = "tests/data/config/configuration.json"

    actions = {"n_actions": 24, "action_type": "discrete"}
    exchanges = {
        "credentials": {
            "api_key": "487r63835t4323",
            "api_secret_key": "do8u43hgiurwfnlveio"
        },
        "name": "bitfinex"
    }

    with TradingContext.from_json(path) as tc:
        assert tc.shared['base_instrument'] == "EURO"
        assert tc.shared['instruments'] == ["BTC", "ETH"]
        assert tc._config['actions'] == actions
        assert tc._config['exchanges'] == exchanges


def test_create_trading_context_from_yaml():
    path = "tests/data/config/configuration.yaml"

    actions = {"n_actions": 24, "action_type": "discrete"}
    exchanges = {
        "credentials": {
            "api_key": "487r63835t4323",
            "api_secret_key": "do8u43hgiurwfnlveio"
        },
        "name": "bitfinex"
    }

    with TradingContext.from_yaml(path) as tc:

        assert tc.shared['base_instrument'] == "EURO"
        assert tc.shared['instruments'] == ["BTC", "ETH"]
        assert tc._config['actions'] == actions
        assert tc._config['exchanges'] == exchanges
