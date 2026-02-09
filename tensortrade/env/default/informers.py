from tensortrade.env.generic import Informer, TradingEnv


class TensorTradeInformer(Informer):
    """An informer for the TensorTrade environment."""

    def info(self, env: TradingEnv) -> dict:
        return {
            "step": self.clock.step,
            "net_worth": env.action_scheme.portfolio.net_worth,
        }
