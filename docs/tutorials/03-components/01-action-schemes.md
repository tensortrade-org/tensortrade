# Action Schemes

Action schemes convert agent outputs into trading orders. This tutorial explains how they work and which to choose.

## Learning Objectives

After this tutorial, you will understand:
- What action schemes do
- How BSH (Buy/Sell/Hold) works in detail
- Alternative action schemes
- How to customize action behavior

---

## What Action Schemes Do

```
Agent Output ──────> ActionScheme ──────> Trading Orders
   (number)           (interprets)         (buy/sell)
```

The RL agent outputs a number (the action). The action scheme converts this number into actual trading orders.

---

## BSH: Buy/Sell/Hold

BSH is TensorTrade's default and simplest action scheme.

### The State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                     BSH State Machine                           │
│                                                                 │
│                                                                 │
│         action=0                           action=1             │
│    ┌──────────────┐                   ┌──────────────┐         │
│    │              │                   │              │         │
│    │    ┌─────────v─────────┐        │    ┌─────────v──────┐  │
│    │    │                   │        │    │                │  │
│    │    │   LONG (BTC)      │        │    │   CASH (USD)   │  │
│    │    │   self.action=0   │        │    │   self.action=1│  │
│    │    │                   │        │    │                │  │
│    │    └─────────┬─────────┘        │    └────────┬───────┘  │
│    │              │                   │             │          │
│    └──────────────┘                   └─────────────┘          │
│         (stay)                             (stay)              │
│                                                                 │
│                     action=1                action=0            │
│    ┌───────────────────────────────────────────────────────┐   │
│    │                                                       │   │
│    │   LONG ────────────────────────────────────> CASH    │   │
│    │   (BTC)           SELL ALL BTC              (USD)    │   │
│    │                                                       │   │
│    │   CASH <──────────────────────────────────── LONG    │   │
│    │   (USD)           BUY ALL BTC               (BTC)    │   │
│    │                                                       │   │
│    └───────────────────────────────────────────────────────┘   │
│         (trade happens)                   (trade happens)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Code Walkthrough

```python
class BSH(TensorTradeActionScheme):
    """Buy/Sell/Hold action scheme."""

    def __init__(self, cash: 'Wallet', asset: 'Wallet'):
        super().__init__()
        self.cash = cash      # USD wallet
        self.asset = asset    # BTC wallet
        self.action = 0       # Current position (0=long, 1=cash)

    @property
    def action_space(self):
        return Discrete(2)    # Only 0 or 1

    def get_orders(self, action: int, portfolio: 'Portfolio') -> 'Order':
        order = None

        # Only trade if position changes
        if abs(action - self.action) > 0:
            # Determine source and target wallets
            src = self.cash if self.action == 0 else self.asset
            tgt = self.asset if self.action == 0 else self.cash

            # Check we have balance to trade
            if src.balance == 0:
                return []

            # Create order to convert 100% of balance
            order = proportion_order(portfolio, src, tgt, 1.0)
            self.action = action  # Update current position

        return [order]
```

### Key Insight: action_space is Discrete(2), not 3

```
Common misconception:
  Action 0 = BUY
  Action 1 = SELL
  Action 2 = HOLD

Actual BSH:
  Action 0 = "I want to be in BTC (long)"
  Action 1 = "I want to be in USD (cash)"
  HOLD = when action matches current position

If currently in BTC (self.action=0):
  Agent outputs 0 → No trade (stay in BTC) = HOLD
  Agent outputs 1 → Trade BTC→USD = SELL

If currently in USD (self.action=1):
  Agent outputs 0 → Trade USD→BTC = BUY
  Agent outputs 1 → No trade (stay in USD) = HOLD
```

### Why BSH Causes Overtrading

```
Problem: Agent oscillates between 0 and 1

Step 1: action=0 (want BTC), currently cash → BUY (trade 1)
Step 2: action=1 (want cash), currently BTC → SELL (trade 2)
Step 3: action=0 (want BTC), currently cash → BUY (trade 3)
Step 4: action=1 (want cash), currently BTC → SELL (trade 4)
...

4 trades in 4 steps = 4 × 0.1% commission = 0.4% loss
```

---

## SimpleOrders: More Flexibility

SimpleOrders allows multiple position sizes and trade types.

```python
from tensortrade.env.default.actions import SimpleOrders

action_scheme = SimpleOrders(
    criteria=[MarketOrder()],    # Order type
    trade_sizes=[0.25, 0.5, 1.0], # 25%, 50%, 100% of balance
    durations=[None],            # Order duration
)

# Action space: Discrete(N)
# Where N = len(criteria) × len(trade_sizes) × len(durations) × 2 sides + 1 (hold)
#         = 1 × 3 × 1 × 2 + 1 = 7 actions

# Actions:
# 0 = HOLD
# 1 = BUY 25%
# 2 = BUY 50%
# 3 = BUY 100%
# 4 = SELL 25%
# 5 = SELL 50%
# 6 = SELL 100%
```

### Advantages

- Partial position sizes (don't have to go all-in)
- Explicit HOLD action (action 0)
- Can specify order criteria (limit orders, etc.)

### Disadvantages

- Larger action space = harder to learn
- More parameters to configure

---

## ManagedRiskOrders: Stop-Loss & Take-Profit

Automatically places stop-loss and take-profit orders.

```python
from tensortrade.env.default.actions import ManagedRiskOrders

action_scheme = ManagedRiskOrders(
    stop=[0.02, 0.05],      # Stop-loss at 2% or 5% loss
    take=[0.02, 0.05, 0.10], # Take-profit at 2%, 5%, or 10% gain
    trade_sizes=[0.5, 1.0],  # 50% or 100% of balance
)

# Each trade automatically includes:
# - Entry order (market order)
# - Stop-loss order (closes position if loss exceeds threshold)
# - Take-profit order (closes position if gain exceeds threshold)
```

### When to Use

- When you want automatic risk management
- When agent shouldn't have to learn when to exit
- For strategies with defined risk/reward ratios

---

## Choosing an Action Scheme

| Scheme | Action Space | Best For |
|--------|--------------|----------|
| **BSH** | Discrete(2) | Simple strategies, learning direction |
| **SimpleOrders** | Discrete(N) | Position sizing strategies |
| **ManagedRiskOrders** | Discrete(N) | Risk-managed strategies |

### Start with BSH

BSH is recommended for beginners because:
1. Smallest action space = faster learning
2. Easiest to understand
3. Works well with PBR reward

### Graduate to SimpleOrders

When you want:
1. Partial positions (25%, 50%)
2. Explicit hold action
3. Multiple order types

---

## Connecting to Reward Schemes

BSH and PBR work together:

```python
# Create components
reward_scheme = PBR(price=price)
action_scheme = BSH(cash=cash, asset=asset)

# Attach reward scheme to action scheme
action_scheme.attach(reward_scheme)

# Now when BSH processes an action:
# 1. BSH calls reward_scheme.on_action(action)
# 2. PBR updates its internal position tracking
# 3. PBR can compute correct reward based on position
```

The `.attach()` pattern allows reward schemes to know what action was taken.

---

## Custom Action Schemes

### Basic Template

```python
from tensortrade.env.generic import ActionScheme
from gymnasium.spaces import Discrete

class MyActionScheme(ActionScheme):
    def __init__(self, portfolio, ...):
        super().__init__()
        self.portfolio = portfolio

    @property
    def action_space(self):
        return Discrete(3)  # 0, 1, 2

    def perform(self, env, action):
        if action == 0:
            # Buy logic
            pass
        elif action == 1:
            # Sell logic
            pass
        # action == 2 = hold (do nothing)

    def reset(self):
        # Reset state between episodes
        pass
```

### Example: Confidence-Based Trading

```python
from gymnasium.spaces import Box
import numpy as np

class ConfidenceActionScheme(ActionScheme):
    """Trade amount based on agent confidence."""

    def __init__(self, cash, asset, min_confidence=0.6):
        super().__init__()
        self.cash = cash
        self.asset = asset
        self.min_confidence = min_confidence

    @property
    def action_space(self):
        # [direction, confidence]
        # direction: -1 (sell) to +1 (buy)
        # confidence: 0 to 1
        return Box(low=np.array([-1, 0]),
                   high=np.array([1, 1]),
                   dtype=np.float32)

    def perform(self, env, action):
        direction, confidence = action

        # Only trade if confident enough
        if confidence < self.min_confidence:
            return  # Hold

        # Trade proportion based on confidence
        proportion = confidence

        if direction > 0:
            # Buy: confidence determines size
            order = proportion_order(
                self.portfolio,
                self.cash, self.asset,
                proportion=proportion
            )
        else:
            # Sell
            order = proportion_order(
                self.portfolio,
                self.asset, self.cash,
                proportion=proportion
            )

        if order:
            self.broker.submit(order)
            self.broker.update()
```

---

## Reducing Overtrading

### Option 1: Training Commission

Train with higher commission to teach trading discipline:

```python
# During training
exchange_options = ExchangeOptions(commission=0.005)  # 0.5%

# Agent learns: "trading is expensive, be selective"
```

### Option 2: Action Masking

Prevent trading for N steps after a trade:

```python
class CooldownBSH(BSH):
    def __init__(self, cash, asset, cooldown=10):
        super().__init__(cash, asset)
        self.cooldown = cooldown
        self.steps_since_trade = cooldown

    def get_orders(self, action, portfolio):
        self.steps_since_trade += 1

        # Can't trade during cooldown
        if self.steps_since_trade < self.cooldown:
            return []

        # Normal BSH logic
        if abs(action - self.action) > 0:
            # Trade
            self.steps_since_trade = 0
            # ... rest of logic
```

### Option 3: Trade Penalty in Reward

See [Reward Schemes](02-reward-schemes.md) for AdvancedPBR with trade penalty.

---

## Key Takeaways

1. **BSH is a state machine** - Actions represent desired position, not trade type
2. **Action 0 = long, Action 1 = cash** - Not BUY/SELL/HOLD
3. **Overtrading is common** - Agent oscillates, each flip costs commission
4. **SimpleOrders adds flexibility** - Position sizing, explicit hold
5. **Custom schemes are possible** - Inherit from ActionScheme

---

## Checkpoint

Before continuing, verify you understand:

- [ ] Why BSH has action_space Discrete(2), not 3
- [ ] When a trade actually happens in BSH (position change)
- [ ] Why the agent might oscillate between 0 and 1
- [ ] How `.attach()` connects action and reward schemes

---

## Next Steps

[02-reward-schemes.md](02-reward-schemes.md) - Learn how rewards shape agent behavior
