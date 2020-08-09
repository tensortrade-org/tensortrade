## [Tensorforce](https://tensorforce.readthedocs.io/en/0.4.4)

I will also quickly cover the Tensorforce library to show how simple it is to switch between reinforcement learning frameworks.

```python
from tensorforce.agents import Agent

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

agent = Agent.from_spec(spec=agent_spec,
                        kwargs=dict(network=network_spec,
                                    states=environment.states,
                                    actions=environment.actions))
```

_If you would like to know more about Tensorforce agents, you can view the_ [Documentation.](https://tensorforce.readthedocs.io/en/0.4.4)
