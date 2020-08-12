## Action Scheme

Action schemes define the action space of the environment and convert an agent's actions into executable trades.

For example, if we were using a discrete action space of 3 actions (0 = hold, 1 = buy 100 %, 2 = sell 100%), our learning agent does not need to know that returning an action of 1 is equivalent to buying an instrument. Rather, our agent needs to know the reward for returning an action of 1 in specific circumstances, and can leave the implementation details of converting actions to trades to the `ActionScheme`.

Each action scheme has a get_trade method, which will transform the agent's specified action into an executable `Trade`. It is often necessary to store additional state within the scheme, for example to keep track of the currently traded position. This state should be reset each time the action scheme's reset method is called, which is done automatically when the parent `TradingEnvironment` is reset.

```python
from tensortrade.actions import DiscreteActions

action_scheme = DiscreteActions(n_actions=20, instrument='BTC')
```

_This discrete action scheme uses 20 discrete actions, which equates to 4 discrete sizes for each of the 5 trade types (market buy/sell, limit buy/sell, and hold). E.g. [0,5,10,15]=hold, 1=market buy 25%, 2=market sell 25%, 3=limit buy 25%, 4=limit sell 25%, 6=market buy 50%, 7=market sell 50%, etc…_
