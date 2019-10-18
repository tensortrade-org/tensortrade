
import tensortrade as td


from abc import abstractmethod
from tensortrade.base import Component


class DataMessageComponent(Component):

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
        super(WorthMessageComponent, self).__init__({name: value})
        self.name = name
        self.value = value

    def message(self):
        return '{} is worth {}'.format(self.name, self.value)


class PlanComponent(Component):

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

    assert instance.context is None
    assert instance.name == name
    assert instance.value == value


config = {
        'base_instrument': 'EURO',
        'products': ['BTC', 'ETH'],
        'credentials': {
            'api_key': '48hg34wydghi7ef',
            'api_secret_key': '0984hgoe8d7htg'
        }
}


def test_injects_concrete_tensor_trade_component_with_context():

    with td.TradingContext(**config):

        name = 'TensorTrade'
        value = 'the time and effort.'
        instance = WorthMessageComponent(name=name, value=value)
        assert instance.context == config
        assert instance.message() == 'TensorTrade is worth the time and effort.'


def test_inject_multiple_components_with_context():

    with td.TradingContext(**config):
        name = 'TensorTrade'
        value = 'the time and effort.'
        instance = WorthMessageComponent(name=name, value=value)
        win = WinPlanComponent()
        lose = LosePlanComponent()

        assert instance.context == win.context == lose.context

    assert instance.context == win.context == lose.context


def test_inject_contexts_at_different_levels():
    c1 = {'level': 0, **config}
    c2 = {'level': 1, **config}

    with td.TradingContext(**c1) as tc1:
        name = 'TensorTrade'
        value = 'the time and effort.'
        instance1 = WorthMessageComponent(name=name, value=value)
        win1 = WinPlanComponent()
        lose1 = LosePlanComponent()
        assert instance1.context is tc1
        assert win1.context is tc1
        assert lose1.context is tc1

        assert instance1.context == win1.context == lose1.context

        with td.TradingContext(**c2) as tc2:
            name = 'TensorTrade'
            value = 'the time and effort.'
            instance2 = WorthMessageComponent(name=name, value=value)
            win2 = WinPlanComponent()
            lose2 = LosePlanComponent()
            assert instance2.context is tc2
            assert win2.context is tc2
            assert lose2.context is tc2

            assert instance2.context == win2.context == lose2.context

        assert instance2.context == win2.context == lose2.context

    assert instance1.context is tc1
    assert win1.context is tc1
    assert lose1.context is tc1

    assert instance2.context is tc2
    assert win2.context is tc2
    assert lose2.context is tc2
