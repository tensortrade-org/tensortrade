from tensortrade.orders.criteria import Criteria

class TimedSteps(Criteria):

    def __init__(self, duration: int):
        self.duration = duration

    def check(self, order: 'Order', exchange: 'Exchange'):
        return (exchange.clock.step - order.step) >= self.duration

    def __str__(self):
        return "<Timed: duration= {}steps>".format(self.duration)
