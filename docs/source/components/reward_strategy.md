## Reward Strategy

Reward strategies receive the `Trade` taken at each time step and return a `float`, corresponding to the benefit of that specific action. For example, if the action taken this step was a sell that resulted in positive profits, our `RewardStrategy` could return a positive number to encourage more trades like this. On the other hand, if the action was a sell that resulted in a loss, the strategy could return a negative reward to teach the agent not to make similar actions in the future.

A version of this example algorithm is implemented in `SimpleProfitStrategy`, however more complex strategies can obviously be used instead.

Each reward strategy has a `get_reward` method, which takes in the trade executed at each time step and returns a `float` corresponding to the value of that action. As with action strategies, it is often necessary to store additional state within a reward strategy for various reasons. This state should be reset each time the reward strategy's reset method is called, which is done automatically when the parent `TradingEnvironment` is reset.

```python
from tensortrade.rewards import SimpleProfitStrategy

reward_strategy = SimpleProfitStrategy()
```

_The simple profit strategy returns a reward of -1 for not holding a trade, 1 for holding a trade, 2 for purchasing an instrument, and a value corresponding to the (positive/negative) profit earned by a trade if an instrument was sold._
