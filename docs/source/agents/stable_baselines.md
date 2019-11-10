## [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/)

In this example, we will be using the Stable Baselines library to provide learning agents to our trading scheme, however, the TensorTrade framework is compatible with many reinforcement learning libraries such as Tensorforce, Ray's RLLib, OpenAI's Baselines, Intel's Coach, or anything from the TensorFlow line such as TF Agents.

It is possible that custom TensorTrade learning agents will be added to this framework in the future, though it will always be a goal of the framework to be interoperable with as many existing reinforcement learning libraries as possible, since there is so much concurrent growth in the space.
But for now, Stable Baselines is simple and powerful enough for our needs.

```python
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import PPO2

model = PPO2
policy = MlpLnLstmPolicy
params = { "learning_rate": 1e-5 }

agent = model(policy, environment, model_kwargs=params)
```

_Note: Stable Baselines is not required to use TensorTrade though it is required for this tutorial. This example uses a GPU-enabled Proximal Policy Optimization model with a layer-normalized LSTM perceptron network. If you would like to know more about Stable Baselines, you can view the_ [Documentation.](https://stable-baselines.readthedocs.io/en/master/)
