# Reward Scheme

Reward schemes receive the `TradingEnv` at each time step and return a `float`, corresponding to the benefit of that specific action. For example, if the action taken this step was a sell that resulted in positive profits, our `RewardScheme` could return a positive number to encourage more trades like this. On the other hand, if the action was a sell that resulted in a loss, the scheme could return a negative reward to teach the agent not to make similar actions in the future.

A version of this example algorithm is implemented in `SimpleProfit`, however more complex schemes can obviously be used instead.

Each reward scheme has a `reward` method, which takes in the `TradingEnv` at each time step and returns a `float` corresponding to the value of that action. As with action schemes, it is often necessary to store additional state within a reward scheme for various reasons. This state should be reset each time the reward scheme's reset method is called, which is done automatically when the environment is reset.

Ultimately the agent creates a sequence of actions to maximize its total reward over a given time. The `RewardScheme` is an abstract class that encapsulates how to tell the trading bot in `tensortrade` if it's trading positively or negatively over time. The same methods will be called each time for each step, and we can directly swap out compatible schemes.

```python
from tensortrade.env.default.rewards import SimpleProfit

reward_scheme = SimpleProfit()
```

_The simple profit scheme returns a reward of -1 for not holding a trade, 1 for holding a trade, 2 for purchasing an instrument, and a value corresponding to the (positive/negative) profit earned by a trade if an instrument was sold._

## Default
These are the default reward schemes.
<hr>

### Simple Profit
A reward scheme that rewards the agent for profitable trades and prioritizes trading over not trading.

<br>**Overview**<br>
The simple profit scheme needs to keep a history of profit over time. The way it does this is through looking at the portfolio as a means of keeping track of how the portfolio moves. This is seen inside of the `get_reward` function.

<br>**Computing Reward**<br>
<br>**Compatibility**<br>
<hr>

### Risk Adjusted Returns
A reward scheme that rewards the agent for increasing its net worth, while penalizing more volatile strategies.

<br>**Overview**<br>
When trading you often are not just looking at the overall returns of your model. You're also looking at the overall volatility of your trading strategy over time compared to other metrics. The two major strategies here are the sharpe and sortino ratio.

The **sharpe ratio** looks at the overall movements of the portfolio and generates a penalty for massive movements through a lower score. This includes major movements towards the upside and downside.

![Sharpe Ratio](../_static/images/sharpe.png)

The **sortino ratio** takes the same idea, though it focuses more on penalizing only the upside. That means it'll give a huge score for moments when the price moves upward, and will only give a negative score when the price drops heavily. This is a great direction for the RL algorithm. Seeing that we don't want to incur heavy downsides, yet want to take on large upsides, using this metric alone gives us lots of progress to mititgate downsides and increase upsides.

![Sortino Ratio](../_static/images/sortino.png)

<br>**Computing Reward**<br>
Given the choice of `return_algorithm` the reward is computed using the `risk_free_rate` and the `target_returns` parameters.

<br>**Compatibility**<br>
<hr>
