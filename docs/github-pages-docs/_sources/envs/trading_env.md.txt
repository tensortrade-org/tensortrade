# TradingEnvironment

A trading environment is a reinforcement learning environment that follows OpenAI's `gym.Env` specification. This allows us to leverage many of the existing reinforcement learning models in our trading agent, if we'd like.

`TradingEnv` steps through the various interfaces from the `tensortrade` library in a consistent way, and will likely not change too often as all other parts of `tensortrade` changes. We're going to go through an overview of the Trading environment below.

Trading environments are fully configurable gym environments with highly composable components:
* `ActionScheme`
* `RewardScheme`
* `Observer`
* `Stopper`
* `Informer`
* `Renderer`

- The `ActionScheme` interprets and applies the agent's actions to the environment.
- The `RewardScheme` computes the reward for each time step based on the agent's performance.
- The `Observer` generates the next observation for the agent.
- The `Stopper` determines whether or not the episode is over.
- The `Informer` generates useful monitoring information at each time step.
- The `Renderer` renders a view of the environment and interactions.

That's all there is to it, now it's just a matter of composing each of these components into a complete environment.

When the reset method of a `TradingEnv` is called, all of the child components will also be reset. The internal state of each action scheme, reward scheme, observer, stopper, and informer will be set back to their default values, ready for the next episode.

Let's begin with an example environment. As mentioned before, initializing a `TradingEnv` requires each component in order to function.

## OpenAI Gym Primer

Usually the OpenAI gym runs in the following way:

```python
# Declare the environment
env = TrainingEnvironment()
# Declare and agent with an action_space, usually declared inside of the environment itself
agent = RandomAgent(env.action_space)
reward = 0
done = False

# Reset all of the variables
ob = env.reset() # Gets an observation as a response to resetting the variables
while True:
    # Get an observation, and input the previous reward, and indicator if the episode is complete or not (done).
    action = agent.act(ob, reward, done)
    ob, reward, done, _ = env.step(action)
    if done:
        break
```

As such, the TradingEnvironment runs largely like this as well.

```python
from tensortrade.environments import TradingEnvironment
from tensortrade.strategies import StableBaselinesTradingStrategy


environment = TradingEnv(
    exchange=exchange,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    feature_pipeline=feature_pipeline
)

strategy.environment = environment
test_performance = strategy.run(episodes=1, testing=True)
```

Here you may notice that we don't have the same training code we saw above:

```python
while True:
    # Get an observation, and input the previous reward, and indicator if the episode is complete or not (done).
    action = agent.act(ob, reward, done)
    ob, reward, done, _ = env.step(action)
    if done:
        break
```

That's because the code to run that exist directly inside of the `TradingStrategy` codebase. The command `run`, has abstractions of that code. Please refer to the Strategies codebase.

## Functions:
To better understand what's inside of the `TradingEnv`, you should understand the way each component fits in.

### Public

- `step`
  - Run one timestep within the environment based on the specified action.
- `reset`
  - Resets the state of the environment and returns an initial observation.
- `render`
  - This sends an output of what's occuring in the gym enviornment for the user to keep track of.
