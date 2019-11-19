# DiscreteActions

Simple discrete scheme, which calculates the trade amount as a fraction of the total balance.

## Key Variables

- `instruments`
  - The exchange symbols of the instruments being traded.
- `actions_per_instrument`
  - The number of bins to divide the total balance by. Defaults to 20 (i.e. 1/20, 2/20, ..., 20/20).
- `max_allowed_slippage_percent`
  - The maximum amount above the current price the scheme will pay for an instrument. Defaults to 1.0 (i.e. 1%).

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

- `reset`
  - Optionally implementable method for resetting stateful schemes.
- `get_trade`

  - Get the trade to be executed on the exchange based on the action provided.
  - Usually this is the way we distill the information generated from the `action_space`.

## Use Cases

```py
from tensortrade.actions import DiscreteActions

action_scheme = DiscreteActions(n_actions=20, instrument='BTC')
```

_This discrete action scheme uses 20 discrete actions, which equates to 4 discrete amounts for each of the 5 trade types (market buy/sell, limit buy/sell, and hold). E.g. [0,5,10,15]=hold, 1=market buy 25%, 2=market sell 25%, 3=limit buy 25%, 4=limit sell 25%, 6=market buy 50%, 7=market sell 50%, etc…_
