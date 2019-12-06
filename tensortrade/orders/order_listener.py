from abc import ABCMeta, abstractmethod


class OrderListener(object, metaclass=ABCMeta):
    def order_executed(self, order: 'Order', exchange: 'Exchange'):
        pass

    def order_cancelled(self, order: 'Order'):
        pass

    def order_filled(self, order: 'Order', exchange: 'Exchange', amount: float):
        pass

    def order_completed(self, order: 'Order', exchange: 'Exchange'):
        pass
