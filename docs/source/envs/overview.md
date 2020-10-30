# Overview

A trading environment is a reinforcement learning environment that follows OpenAI's `gym.Env` specification. This allows us to leverage many of the existing reinforcement learning models in our trading agent, if we'd like.

`TradingEnv` steps through the various interfaces from the `tensortrade` library in a consistent way, and will likely not change too often as all other parts of `tensortrade` changes. We're going to go through an overview of the Trading environment below.

Trading environments are fully configurable gym environments with highly composable components:
* The `ActionScheme` interprets and applies the agent's actions to the environment.
* The `RewardScheme` computes the reward for each time step based on the agent's performance.
* The `Observer` generates the next observation for the agent.
* The `Stopper` determines whether or not the episode is over.
* The `Informer` generates useful monitoring information at each time step.
* The `Renderer` renders a view of the environment and interactions.

That's all there is to it, now it's just a matter of composing each of these components into a complete environment.

When the reset method of a `TradingEnv` is called, all of the child components will also be reset. The internal state of each action scheme, reward scheme, observer, stopper, and informer will be set back to their default values, ready for the next episode.


# What if I can't make a particular environment?

If none of the environments available in codebase serve your needs let us know! We would love to hear about so we can keep improving the quality of our framework as well as keeping up with the needs of the people using it.
