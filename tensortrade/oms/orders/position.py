
from tensortrade.core import TimedIdentifiable
from tensortrade.oms.orders import Trade, TradeSide, TradeType

class Position(TimedIdentifiable):
    """A class to represent opened positions in Broker 
       to keep track of opened Positions

    Parameters
    ----------
    id : int, optional
        Position-ID created by the Broker upon Order Execution
    is_open : bool, optional
        Indicates the Position is Open in the Broker

    """
    def __init__(self, 
                id: str,
                quantity: 'Quantity',
                order: 'Order',
                ):
        """
        Init function 
        :param symbol: symbol of the stock to be invested
        """
        super().__init__()
        # Position-ID
        self.id = id
        # Flag to indicate that Buy/Sell Position
        self.is_open = True
        self.quantity = quantity
        self.order = order
        self.is_buy = True if order.side == TradeSide.BUY else False
        self.is_sell = True if order.side == TradeSide.SELL else False

    def get_worth_value(self) -> float:
        """Calculate the Worth Value based on Position-Type
        Parameters
        ----------
        - buy_price : float
            the current buying price indicated by the Broker
        - sell_price:             
            the current selling price indicated by the Broker
        Returns
        -------
        `float`: worth value of position
        """
        #TODO: support buy/sell Price difference
        sell_price = self.order.exchange_pair.price
        buy_price  = self.order.exchange_pair.price
        if self.is_open:
            if self.is_buy:
                # calculate worth-bought value based on amounts
                return self.quantity.size * sell_price
            else:
                # calculate worth-bought value based on amounts
                # note: sell-position is also know as short-position - 
                # Broker lends stocks to the trader then immediately sells the stocks, and, after their price goes down, buys them back for a lower price. He then returns the assets to the broker and keeps the difference.
                diff = (self.quantity.size * self.order.price) - (self.quantity.size * buy_price)
                return (self.quantity.size * self.order.price) + diff
        else:
            return 0

    def get_profit_value(self) -> float:
        """Calculate profit value of position, based on Position-Type
        Parameters
        ----------
        - buy_price : float
            the current buying price indicated by the Broker
        - sell_price:             
            the current selling price indicated by the Broker
        Returns
        -------
        `float`: profit-value for this position
        """ 
        if self.is_open:
            return self.get_worth_value() - self.order.price
        else:
            return 0

    def get_profit_percentage(self, buy_price, sell_price) -> float:
        """Calculate profit percentage of position, based on Position-Type
        Parameters
        ----------
        - buy_price : float
            the current buying price indicated by the Broker
        - sell_price:             
            the current selling price indicated by the Broker
        Returns
        -------
        `float`: profit-percentage in range [-1,1]
        """        
        if self.is_open:
            return self.get_profit_value()/self.order.price
        else:
            return 0

    def close(self):
        """close position by the broker """
        self.is_open = False        