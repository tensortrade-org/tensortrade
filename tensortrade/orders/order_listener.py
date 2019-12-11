from abc import ABCMeta, abstractmethod


class OrderListener(object, metaclass=ABCMeta):

    def on_execute(self, order: 'Order', exchange: 'Exchange'):
        pass

    def on_cancel(self, order: 'Order', exchange: 'Exchange'):
        pass

    def on_fill(self, order: 'Order', exchange: 'Exchange', amount: float):
        pass

    def on_complete(self, order: 'Order', exchange: 'Exchange'):
        pass
