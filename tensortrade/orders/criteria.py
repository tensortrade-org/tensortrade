
from abc import abstractmethod, ABC


class Criteria(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, exchange: 'Exchange') -> bool:
        pass


class Limit(Criteria):

    name = "limit"

    def __init__(self, side, price):
        super().__init__(self)
        self.side = side
        self.price = price

    def __call__(self, exchange: 'Exchange') -> bool:
        current_price = exchange.current_price()
        c1 = (self.side == "buy" and current_price <= self.price)
        c2 = (self.side == "sell" and current_price >= self.price)
        if c1 or c2:
            return True
        return False
