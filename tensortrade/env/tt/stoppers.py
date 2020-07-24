
from tensortrade.env.generic import Stopper


class MaxLossStopper(Stopper):

    def __init__(self, max_allowed_loss):
        self.max_allowed_loss = max_allowed_loss

    def stop(self, env):
        c1 = env.action_scheme.portfolio.profit_loss < self.max_allowed_loss
        c2 = not env.observer.has_next()
        return c1 or c2
