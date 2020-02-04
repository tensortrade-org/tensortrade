from tensortrade.orders.criteria import Criteria
from typing import Union
from datetime import datetime

class TimedSeconds(Criteria):

    def __init__(self, duration: Union[int, float]):
        self.duration = duration

    def check(self, order: 'Order', exchange: 'Exchange'):
        return (datetime.now() - order.created_at).total_seconds() >= self.duration

    def __str__(self):
        return "<Timed: duration= {}seconds>".format(self.duration)
