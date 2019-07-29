import pandas as pd

from typing import List, Callable

from trader.rewards import RewardStrategy


class IncrementalProfitStrategy(RewardStrategy):
    def __init__(self):
        self.last_bought = 0
        self.last_sold_price = -1

    def reward(self,
               current_step: int,
               current_price: float,
               balance: float,
               net_worth: float,
               assets_held: Dict[str, float],
               trades: pd.DataFrame,
               performance: pd.DataFrame) -> float:
        reward = 0

        curr_balance = balance
        prev_balance = performance['balance'].values[-1]

        if self.last_sold_price is -1:
            self.last_sold_price = current_price

        if curr_balance > prev_balance:
            reward = net_worth - performance['net_worth'].values[self.last_bought]
            self.last_sold_price = current_price

        elif curr_balance < prev_balance:
            reward = self.last_sold_price - current_price
            self.last_bought = current_step

        return reward
