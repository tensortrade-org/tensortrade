# Simple Profit

A reward scheme that rewards the agent for profitable trades and prioritizes trading over not trading.

## Class Parameters

None

## Functions

Below are the functions that the `SimpleProfit` uses to effectively operate.

## Private

None

## Public

- `reset` - Reset variables
  - Necessary to reset the last purchase price and state of open positions
  - Variables it resets
    - `_purchase_price`
    - The price the bot purchased the asset
    - `_is_holding_instrument` - A boolean that shares with the get_reward function if we're currently holding onto a trade.
- `get_reward`
  - Returns the reward for the given action
  - The `5^(log_10(profit))` function simply slows the growth of the reward as trades get large.

## Use Cases

The simple profit scheme needs to keep a history of profit over time. The way it does this is through looking at the portfolio as a means of keeping track of how the portfolio moves. It also keeps track to see if it's holding onto a trade as well. This is seen inside of the get_reward function.

**Use Case #1: Buying**

When the bot says buy, it sets the variable `_is_holding_instrument` to `True`, and sets the current price to the price of the trade.

We see that inside of this line of code here. This allows us to check to see if we've made a profit later.

```py
elif trade.is_buy and trade.size > 0:
    self._purchase_price = trade.price
    self._is_holding_instrument = True

```

**Use Case #2: Selling**

We then sell afterward using the original trade price as a reference. Which is suggested in the lines below:

```py
if trade.is_sell and trade.size > 0:
    self._is_holding_instrument = False
    profit_per_instrument = trade.price - self._purchase_price
    profit = trade.size * profit_per_instrument
    profit_sign = np.sign(profit)

    return profit_sign * (1 + (5 ** np.log10(abs(profit))))
```
