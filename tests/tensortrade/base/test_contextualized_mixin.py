
import tensortrade as td

from abc import abstractmethod, ABC
from typing import Any
from tensortrade.base import InitContextMeta, ContextualizedMixin


class ConcreteContextual(ContextualizedMixin, metaclass=InitContextMeta):

    def __init__(self, data: dict):
        self.data = data


class AbstractContextualData(ContextualizedMixin, metaclass=InitContextMeta):

    @abstractmethod
    def __init__(self, data: dict):
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: dict):
        self._data = data


class ConcreteContextualData(AbstractContextualData):

    def __init__(self, name: str, value: Any):
        super(ConcreteContextualData, self).__init__({name: value})
        self.name = name
        self.value = value


def test_meta_executes_without_issue():
    data = {
        'apple': 1,
        'banana': 2,
        'carrot': 3
    }
    strategy = ConcreteContextual(data)

    assert strategy


def test_meta_injects_no_context_outside_with():
    data = {
        'apple': 1,
        'banana': 2,
        'carrot': 3
    }
    instance = ConcreteContextual(data)

    assert instance.context is None


config = {
    'base_instrument': 'EURO',
    'products': ['BTC', 'ETH'],
    'credentials': {
        'api_key': '48hg34wydghi7ef',
        'api_secret_key': '0984hgoe8d7htg'
    }
}


def test_meta_injects_context_into_instance():

    with td.TradingContext(**config):
        data = {
            'apple': 1,
            'banana': 2,
            'carrot': 3
        }
        instance = ConcreteContextual(data)

        assert instance.context


def test_no_injection_subclassed_abc_contextualized_mixin():

    name = 'Matt'
    value = 'not for sale!'

    instance = ConcreteContextualData(name=name, value=value)

    assert instance.context is None
    assert instance.data == {name: value}


def test_injects_subclassed_abc_contextualized_mixin_in_context():

    config = {
        'height': 'none of your business',
        'weight': 'how dare you ask such a question'
    }

    with td.TradingContext(**config):
        name = 'Matt'
        value = 'not for sale!'
        instance = ConcreteContextualData(name=name, value=value)

        assert instance.context == config

        assert instance.data == {name: value}

    assert instance.context == config
