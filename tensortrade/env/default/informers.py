
from tensortrade.env.generic import Informer, TradingEnv


class TensorTradeInformer(Informer):

    def __init__(self) -> None:
        super().__init__()

    def info(self, env: 'TradingEnv') -> dict:
        return {
            'step': self.clock.step,
            'portfolio': env.action_scheme.portfolio,
            'broker': env.action_scheme.broker,
            'net_worth': env.action_scheme.portfolio.net_worth
        }
