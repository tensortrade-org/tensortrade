## Trading Strategy

A `TradingStrategy` consists of a learning agent and one or more trading environments to tune, train, and evaluate on. If only one environment is provided, it will be used for tuning, training, and evaluating. Otherwise, a separate environment may be provided at each step.

```python
from stable_baselines import PPO2

from tensortrade.strategies import TensorforceTradingStrategy,
                                   StableBaselinesTradingStrategy

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

b_strategy = StableBaselinesTradingStrategy(environment=environment,
                                            model=PPO2,
                                            policy='MlpLnLSTMPolicy')
```