# Reward Strategy

Reward strategies receive the `Trade` taken at each time step and return a `float`, corresponding to the benefit of that specific action. For example, if the action taken this step was a sell that resulted in positive profits, our `RewardStrategy` could return a positive number to encourage more trades like this. On the other hand, if the action was a sell that resulted in a loss, the strategy could return a negative reward to teach the agent not to make similar actions in the future.

A version of this example algorithm is implemented in `SimpleProfitStrategy`, however more complex strategies can obviously be used instead.

Each reward strategy has a `get_reward` method, which takes in the trade executed at each time step and returns a `float` corresponding to the value of that action. As with action strategies, it is often necessary to store additional state within a reward strategy for various reasons. This state should be reset each time the reward strategy's reset method is called, which is done automatically when the parent `TradingEnvironment` is reset.

Ultimately the agent creates a sequence of actions to maximize its total reward over a given time. The `RewardStrategy` is an abstract class that encapsulates how to tell the trading bot in `tensortrade` if it's trading positively or negatively over time. The same methods will be called each time for each step, and we can directly swap out strategies. 


## Properties and Setters

* `exchange`
  * The central exchange for the strategy.
  * The exchange being used by the current trading environment. Setting the exchange causes the strategy to reset.

## Methods

* `get_reward`
  * Gets the reward for the RL agent.
  * Returns a float corresponding to the benefit earned by the action taken this timestep.
* `reset`
  * Resets the current state if the reward has a state.
  * Optionally implementable method for resetting stateful strategies.
