## Learning Agents

This is where the "deep" part of the deep reinforcement learning framework come in. Learning agents are where the math (read: magic) happens.

At each time step, the agent takes the observation from the environment as input, runs it through its underlying model (a neural network most of the time), and outputs the action to take. For example, the observation might be the previous `open`, `high`, `low`, and `close` price from the exchange. The learning model would take these values as input and output a value corresponding to the action to take, such as `buy`, `sell`, or `hold`.

It is important to remember the learning model has no intuition of the prices or trades being represented by these values. Rather, the model is simply learning which values to output for specific input values or sequences of input values, to earn the highest reward.
