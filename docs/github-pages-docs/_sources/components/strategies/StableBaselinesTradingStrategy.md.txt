# StableBaselinesTradingStrategy

A trading strategy capable of self tuning, training, and evaluating with stable-baselines.

## Class Parameters

* `environment`
  * A `TradingEnvironment` instance for the agent to trade within.
* `agent`
  * A `Tensorforce` agent or agent specification.
* `model`
  * The runner will automatically save the best agent
* `policy`
  * The RL policy to train the agent's model with. Defaults to 'MlpPolicy'.
* `_model_kwargs`
  * ...


## Properties and Setters
* `agent`
  * A `TradingEnvironment` instance for the agent to trade within.
* `max_episode_timesteps`
  * The maximum timesteps per episode.


## Functions
* `restore_agent`
  * Deserialize the strategy's learning agent from a file.
  * **parameters**:
    * `path`
      * The `str` path of the file the agent specification is stored in.
* `save_agent`
  * Serialize the learning agent to a file for restoring later.
  * **parameters**:
    * `path`
      * The `str` path of the file to store the agent specification in.
* `tune`
  * Function `NotImplemented`
* `run`
  * Runs all of the episodes specified. 
  * **parameters**:
    * steps
    * episodes
    * episode_callback
    * evaluation

## Use Cases


**Use Case #1: Run a Strategy**

```py
from stable_baselines import PPO2
from tensortrade.strategies import StableBaselinesTradingStrategy
from tensortrade.environments import TradingEnvironment

environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy)
b_strategy = StableBaselinesTradingStrategy(model=PPO2)
strategy.environment = environment
stategy.run(episodes=10)
```

**Use Case #2: Run a Live Strategy**


```py
import ccxt
from tensortrade.environments import TradingEnvironment
from tensortrade.strategies import StableBaselinesTradingStrategy
from tensortrade.exchanges.live import CCXTExchange
coinbase = ccxt.coinbasepro(...) # your credentials go here in dictionary form
exchange = CCXTExchange(exchange=coinbase,
                        timeframe='1h',
                        base_instrument='USD', 
                        feature_pipeline=feature_pipeline)
environment = TradingEnvironment(exchange=exchange,
                                 action_strategy=action_strategy,
                                 reward_strategy=reward_strategy)
strategy.environment = environment
strategy.restore_agent(path="../agents/ppo_btc/1h")
live_performance = strategy.run(steps=0, trade_callback=episode_cb)
```