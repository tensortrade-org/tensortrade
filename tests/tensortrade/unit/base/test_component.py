
from abc import abstractmethod

from tensortrade.core import Component, TradingContext


class DataMessageComponent(Component):

    registered_name = "messages"

    @abstractmethod
    def __init__(self, data: dict):
        self._data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: dict):
        self._data = data

    @abstractmethod
    def message(self):
        raise NotImplementedError()


class WorthMessageComponent(DataMessageComponent):

    def __init__(self, name, value):
        super(WorthMessageComponent, self).__init__(data={name: value})
        self.name = name
        self.value = value

    def message(self):
        return '{} is worth {}'.format(self.name, self.value)


class PlanComponent(Component):

    registered_name = "plans"

    @abstractmethod
    def __init__(self, plan: str):
        self._plan = plan

    @property
    def plan(self):
        return self._plan

    @plan.setter
    def plan(self, plan: str):
        self._plan = plan

    @abstractmethod
    def is_good_plan(self):
        raise NotImplementedError()


class WinPlanComponent(PlanComponent):

    def __init__(self):
        super(WinPlanComponent, self).__init__(plan="Win money!")

    def is_good_plan(self):
        return True


class LosePlanComponent(PlanComponent):

    def __init__(self):
        super(LosePlanComponent, self).__init__(plan="Lose money!")

    def is_good_plan(self):
        return True


def test_no_context_injected_outside_with():

    name = 'TensorTrade'
    value = 'the time and effort.'
    instance = WorthMessageComponent(name=name, value=value)

    assert instance.name == name
    assert instance.value == value


config = {
    'base_instrument': 'EURO',
    'instruments': ['BTC', 'ETH']
}


def test_injects_concrete_tensor_trade_component_with_context():

    with TradingContext(config):

        name = 'TensorTrade'
        value = 'the time and effort.'
        instance = WorthMessageComponent(name=name, value=value)
        assert instance.context == config
        assert instance.message() == 'TensorTrade is worth the time and effort.'


def test_inject_multiple_components_with_context():

    with TradingContext(config):
        name = 'TensorTrade'
        value = 'the time and effort.'
        instance = WorthMessageComponent(name=name, value=value)
        win = WinPlanComponent()
        lose = LosePlanComponent()

        assert instance.context == win.context == lose.context

    assert instance.context == win.context == lose.context


def test_injects_component_space():
    c = {
        'messages': {
            'msg_var': 0,
            'valid': ['Hello', 'World']
        },
        **config
    }

    with TradingContext(c) as c:
        name = 'TensorTrade'
        value = 'the time and effort.'
        instance = WorthMessageComponent(name=name, value=value)

        assert hasattr(instance.context, 'msg_var')
        assert hasattr(instance.context, 'valid')

        assert instance.context.msg_var == 0
        assert instance.context.valid == ['Hello', 'World']


def test_only_name_registered_component_space():
    c = {
        'actions': {
            'n_actions': 20
        },
        'messages': {
            'msg_var': 0,
            'valid': ['Hello', 'World']
        },
        'plans': {
            'future': 'looking good'
        },
        **config
    }

    with TradingContext(c) as c:
        name = 'TensorTrade'
        value = 'the time and effort.'
        instance = WorthMessageComponent(name=name, value=value)

        assert 'msg_var' in instance.context.keys()
        assert 'valid' in instance.context.keys()
        assert instance.context['msg_var'] == 0
        assert instance.context == {'msg_var': 0, 'valid': ['Hello', 'World'], **config}


def test_inject_contexts_at_different_levels():
    c1 = {
        'messages': {
            'msg_var': 0
        },
        'plans': {
            'plans_var': 24
        },
        **config
    }
    c2 = {
        'messages': {
            'level': 1
        },
        **config
    }

    with TradingContext(c1):
        name = 'TensorTrade'
        value = 'the time and effort.'
        instance1 = WorthMessageComponent(name=name, value=value)
        win1 = WinPlanComponent()
        lose1 = LosePlanComponent()
        assert hasattr(instance1.context, 'msg_var')
        assert hasattr(win1.context, 'plans_var')
        assert hasattr(lose1.context, 'plans_var')

        with TradingContext(c2):
            name = 'TensorTrade'
            value = 'the time and effort.'
            instance2 = WorthMessageComponent(name=name, value=value)

            assert hasattr(instance2.context, 'level')
            assert instance2.context == {'level': 1, **config}
