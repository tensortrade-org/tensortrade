

from tensortrade.orders.criteria import Criteria


class Timed(Criteria):

    def __init__(self, duration):
        self.duration = duration

    def call(self, order: 'Order', exchange: 'Exchange'):
        return (order.clock.step - order.created_at) <= self.duration

    def __str__(self):
        return "<Timed: duration={}>".format(self.duration)

