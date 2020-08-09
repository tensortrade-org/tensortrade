# ContinuousActions

Simple continuous scheme, which calculates the trade size as a fraction of the total balance.

## Key Variables

- `max_allowed_slippage`
  - The exchange symbols of the instruments being traded.
- `instrument`
  - The number of bins to divide the total balance by. Defaults to 20 (i.e. 1/20, 2/20, ..., 20/20).
- `instrument`
  - The maximum size above the current price the scheme will pay for an instrument. Defaults to 1.0 (i.e. 1%).

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

TODO: Place Use Case Here
