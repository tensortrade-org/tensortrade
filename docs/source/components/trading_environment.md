## Trading Environment

A trading environment is a reinforcement learning environment that follows OpenAI's `gym.Env` specification. This allows us to leverage many of the existing reinforcement learning models in our trading agent, if we'd like.

Trading environments are fully configurable gym environments with highly composable `InstrumentExchange`, `FeaturePipeline`, `ActionStrategy`, and `RewardStrategy` components.

- The `InstrumentExchange` provides observations to the environment and executes the agent's trades.
- The `FeaturePipeline` optionally transforms the exchange output into a more meaningful set of features before it is passed to the agent.
- The `ActionStrategy` converts the agent's actions into executable trades.
- The `RewardStrategy` calculates the reward for each time step based on the agent's performance.

That's all there is to it, now it's just a matter of composing each of these components into a complete environment.

When the reset method of a `TradingEnvironment` is called, all of the child components will also be reset. The internal state of each instrument exchange, feature pipeline, transformer, action strategy, and reward strategy will be set back to their default values, ready for the next episode.

Let's begin with an example environment. As mentioned before, initializing a `TradingEnvironment` requires an exchange, an action strategy, and a reward strategy, the feature pipeline is optional.

```python
from tensortrade.environments import TradingEnvironment

environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy,
                                 feature_pipeline=feature_pipeline)
```
