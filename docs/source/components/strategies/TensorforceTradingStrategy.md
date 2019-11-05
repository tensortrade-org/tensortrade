# TensorforceTradingStrategy

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

## Properties and Setters
* `environment`
  * A `TradingEnvironment` instance for the agent to trade within.


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

## Use Cases


**Use Case #1: Run a Strategy**

```py
from stable_baselines import PPO2
from tensortrade.strategies import TensorforceTradingStrategy


agent_spec = {
    "type": "ppo_agent",
    "step_optimizer": {
        "type": "adam",
        "learning_rate": 1e-4
    },
    "discount": 0.99,
    "likelihood_ratio_clipping": 0.2,
}

network_spec = [
    dict(type='dense', size=64, activation="tanh"),
    dict(type='dense', size=32, activation="tanh")
]

a_strategy = TensorforceTradingStrategy(environment=environment,
                                        agent_spec=agent_spec,
                                        network_spec=network_spec)
a_strategy.run()
```
<!-- 
**Use Case #2**

```py

``` -->
<!-- ## Use Cases

**Use Case #1**

```py
print("Hello World")
```

**Use Case #2**

```py
print("Hello World")
``` -->