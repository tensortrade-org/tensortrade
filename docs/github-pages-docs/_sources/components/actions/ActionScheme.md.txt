# ActionScheme

Action schemes define the action space of the environment and convert an agent's actions into executable trades.

For example, if we were using a discrete action space of 3 actions (0 = hold, 1 = buy 100 %, 2 = sell 100%), our learning agent does not need to know that returning an action of 1 is equivalent to buying an instrument. Rather, our agent needs to know the reward for returning an action of 1 in specific circumstances, and can leave the implementation details of converting actions to trades to the `ActionScheme`.

Each action scheme has a get_trade method, which will transform the agent's specified action into an executable `Trade`. It is often necessary to store additional state within the scheme, for example to keep track of the currently traded position. This state should be reset each time the action scheme's reset method is called, which is done automatically when the parent `TradingEnvironment` is reset.

## What is an Action?

This is a review of what was mentioned inside of the overview section. It explains how a RL operates. You'll better understand what an action is in context of an observation space and reward. At the same time, hopefully this will be a proper refresher.

An action is a predefined value of how the machine should move inside of the world. To better summarize, its a _command that a player would give inside of a video game in respose to a stimuli_. The commands usually come in the form of an `action_space`. An `action_space` is something that represents how to make the user move inside of an environment. While it might not be easily interpretable by humans, it can easily be interpreted by a machine.

Let's look at a good example. Lets say we're trying to balance a cart with a pole on it (cartpole). We can choose to move the cart left and right. This is a `Discrete(2)` action type.

- 0 - Push cart to the left
- 1 - Push cart to the right

When we get the action from the RL agent, the environment will see that number instead of a name. We can create lists, tuples, and a box.

![Watch Link Run Around In Circles](../../_static/images/cartpole.gif)

## Setters & Properties

Each property and property setter.

- `dtype`
  - A type or str corresponding to the dtype of the `action_space`.
- `exchange`
  - The exchange being used by the current trading environment.
  - This will be set by the trading environment upon initialization. Setting the exchange causes the scheme to reset.
- `action_space`
  - The shape of the actions produced by the scheme. This takes in a `gym.space` and is different for each given scheme.

## Functions

- reset
  - Optionally implementable method for resetting stateful schemes.
- get_trade
  - Get the trade to be executed on the exchange based on the action provided.
  - Usually this is the way we distill the information generated from the `action_space`.
