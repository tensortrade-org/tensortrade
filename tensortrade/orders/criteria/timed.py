

from .criteria import Criteria


class Timed(Criteria):

    def __init__(self, wait):
        self.wait = wait

    def call(self, order: 'Order', exchange: 'Exchange'):
        return (order.clock.step - order.created_at) < self.wait

    def __str__(self):
        return "<Timed: wait={}>".format(self.wait)

