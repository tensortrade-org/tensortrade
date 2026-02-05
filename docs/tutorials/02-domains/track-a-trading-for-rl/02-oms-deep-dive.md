# OMS Deep Dive

This tutorial explores TensorTrade's Order Management System in detail.

## Learning Objectives

After this tutorial, you will understand:
- How orders flow through the system
- Wallet and balance mechanics
- Exchange configuration options
- How to customize execution behavior

---

## Order Flow

When your agent takes an action, here's what happens:

```
Agent outputs: action = 0 (BUY)
         │
         v
┌────────────────────────────────────────────────────────────────┐
│  ActionScheme.perform(action)                                  │
│                                                                │
│  1. Interpret action (0 = switch to long position)             │
│  2. Check current position (currently in USD)                  │
│  3. Create order: proportion_order(src=USD, tgt=BTC, pct=1.0) │
│  4. Submit to broker                                           │
└────────────────────────────────────────────────────────────────┘
         │
         v
┌────────────────────────────────────────────────────────────────┐
│  Broker.submit(order)                                          │
│                                                                │
│  1. Validate order (sufficient balance?)                       │
│  2. Add to order queue                                         │
│  3. Call broker.update() to process                            │
└────────────────────────────────────────────────────────────────┘
         │
         v
┌────────────────────────────────────────────────────────────────┐
│  Exchange.execute(order)                                       │
│                                                                │
│  1. Get current price from price stream                        │
│  2. Calculate quantity after commission                        │
│  3. Debit source wallet                                        │
│  4. Credit target wallet                                       │
│  5. Record trade                                               │
└────────────────────────────────────────────────────────────────┘
         │
         v
┌────────────────────────────────────────────────────────────────┐
│  Portfolio updated                                             │
│                                                                │
│  USD Wallet: $10,000 → $0                                     │
│  BTC Wallet: 0 → 0.0999 BTC                                   │
│  Net Worth: $10,000 → $9,990 (commission)                     │
└────────────────────────────────────────────────────────────────┘
```

---

## Instruments

Instruments represent tradeable assets.

```python
from tensortrade.oms.instruments import Instrument

# Built-in instruments
from tensortrade.oms.instruments import USD, BTC, ETH

# Custom instrument
AAPL = Instrument("AAPL", 2, "Apple Inc Stock")
#                  symbol  precision  name
#                          (decimals)
```

### Quantities

Quantities combine an amount with an instrument:

```python
from tensortrade.oms.instruments import Quantity

# Create a quantity
amount = 100 * USD  # $100
btc_amount = 0.5 * BTC  # 0.5 BTC

# Quantity operations
total = (100 * USD) + (50 * USD)  # 150 USD
half = (100 * USD) / 2  # 50 USD

# Get numeric value
amount.size  # 100.0
amount.instrument  # USD
```

---

## Wallets

Wallets hold balances of instruments.

```python
from tensortrade.oms.wallets import Wallet
from tensortrade.oms.instruments import USD, BTC

# Create wallets
usd_wallet = Wallet(exchange, 10000 * USD)
btc_wallet = Wallet(exchange, 0.5 * BTC)

# Check balance
usd_wallet.balance  # Quantity(10000, USD)
usd_wallet.balance.as_float()  # 10000.0

# Wallet is tied to an exchange
usd_wallet.exchange  # The exchange it trades on
```

### Locked vs Free Balance

When you place an order, funds are "locked" until execution:

```python
wallet.balance      # Total balance (free + locked)
wallet.locked       # Funds reserved for pending orders
wallet.total        # Same as balance

# Example:
# Balance: 100 USD
# Order placed: Buy 50 USD of BTC
# Now:
#   Free: 50 USD
#   Locked: 50 USD
#   Total: 100 USD
```

---

## Portfolio

Portfolio manages multiple wallets.

```python
from tensortrade.oms.wallets import Portfolio

# Create portfolio
portfolio = Portfolio(
    base_instrument=USD,  # Denomination currency
    wallets=[usd_wallet, btc_wallet]
)

# Key properties
portfolio.net_worth  # Total value in base currency
portfolio.base_balance  # Balance of base currency
portfolio.profit_loss  # Cumulative P&L

# Get specific wallet
portfolio.get_wallet(exchange_id, instrument=BTC)

# Performance tracking
portfolio.performance  # Dict of net_worth over time
```

### Net Worth Calculation

```python
def net_worth(self):
    total = 0
    for wallet in self.wallets:
        if wallet.instrument == self.base_instrument:
            total += wallet.balance.size
        else:
            # Convert to base currency using current price
            price = self.get_price(wallet.instrument)
            total += wallet.balance.size * price
    return total
```

---

## Exchanges

Exchanges execute orders and manage prices.

```python
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order

# Configuration options
options = ExchangeOptions(
    commission=0.001,     # 0.1% commission
    min_trade_size=0.001, # Minimum trade size
    max_trade_size=1e6,   # Maximum trade size
    min_trade_price=1e-8, # Minimum price
    max_trade_price=1e8,  # Maximum price
)

# Create exchange with price stream
price = Stream.source([100000, 100100, 99900, ...], dtype="float").rename("USD-BTC")
exchange = Exchange("bitfinex", service=execute_order, options=options)(price)

# Access exchange properties
exchange.id  # "bitfinex"
exchange.options.commission  # 0.001
```

### Price Streams

The exchange gets prices from a Stream:

```python
# Single price stream
price = Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")

# The stream must be attached to exchange
exchange = Exchange("sim", execute_order, options)(price)

# Exchange uses stream name to identify trading pair
# "USD-BTC" means: trading pair where you pay USD to get BTC
```

---

## Order Types

### Basic Order

```python
from tensortrade.oms.orders import Order, TradeSide, TradeType

order = Order(
    step=clock.step,           # Current timestep
    side=TradeSide.BUY,        # BUY or SELL
    trade_type=TradeType.MARKET,  # MARKET or LIMIT
    exchange_pair=exchange_pair,   # What to trade
    price=100000,              # Current price
    quantity=0.1 * BTC,        # How much
    portfolio=portfolio,       # Portfolio to update
)
```

### Proportion Order

Easier way to trade a percentage of balance:

```python
from tensortrade.oms.orders import proportion_order

# Convert 100% of USD to BTC
order = proportion_order(
    portfolio=portfolio,
    source=usd_wallet,      # From USD
    target=btc_wallet,      # To BTC
    proportion=1.0          # 100% of balance
)

# Convert 50% of BTC to USD
order = proportion_order(
    portfolio=portfolio,
    source=btc_wallet,
    target=usd_wallet,
    proportion=0.5
)
```

### Risk-Managed Order

Includes stop-loss and take-profit:

```python
from tensortrade.oms.orders import risk_managed_order

order = risk_managed_order(
    side=TradeSide.BUY,
    exchange_pair=pair,
    price=100000,
    quantity=0.1 * BTC,
    down_percent=0.02,   # 2% stop loss
    up_percent=0.05,     # 5% take profit
    portfolio=portfolio,
)
```

---

## BSH Mechanics

BSH (Buy/Sell/Hold) is TensorTrade's default action scheme.

```python
from tensortrade.env.default.actions import BSH

action_scheme = BSH(cash=usd_wallet, asset=btc_wallet)
```

### State Machine

```
                    action = 0
       ┌────────────────────────────────────┐
       │                                    │
       v                                    │
   ┌───────┐                           ┌───────┐
   │  BTC  │                           │  USD  │
   │ (Long)│                           │(Cash) │
   └───────┘                           └───────┘
       │                                    ^
       │                                    │
       └────────────────────────────────────┘
                    action = 1

Internal state: self.action
  - 0: Currently in BTC (long)
  - 1: Currently in USD (cash)

On action input:
  - If action == self.action: No trade (HOLD)
  - If action != self.action: Trade to switch position
```

### Why Action Space is Discrete(2)

```python
@property
def action_space(self):
    return Discrete(2)  # Only 0 or 1

# The agent outputs:
#   0 = "I want to be in BTC"
#   1 = "I want to be in USD"
#
# Note: This is NOT "BUY" and "SELL"
# It's "desired position"
#
# If already in desired position, no trade happens
```

### The Overtrading Problem

```python
def get_orders(self, action: int, portfolio: 'Portfolio') -> 'Order':
    order = None

    if abs(action - self.action) > 0:  # Position changed
        # Create order to switch positions
        src = self.cash if self.action == 0 else self.asset
        tgt = self.asset if self.action == 0 else self.cash
        order = proportion_order(portfolio, src, tgt, 1.0)
        self.action = action

    return [order]

# Problem: If agent oscillates between 0 and 1 frequently,
# each oscillation creates a trade (and pays commission)
```

---

## Customizing Execution

### Custom Commission

```python
# Percentage commission
options = ExchangeOptions(commission=0.002)  # 0.2%

# The commission is applied to every trade:
# Trade $10,000 → Pay $20 commission → Net $9,980
```

### Custom Execution Service

```python
def my_execute_order(order, exchange, portfolio):
    """Custom execution logic."""
    # Add slippage
    slippage = random.uniform(0, 0.001)  # Up to 0.1%
    adjusted_price = order.price * (1 + slippage)

    # Execute with adjusted price
    # ...

# Use custom execution
exchange = Exchange("sim", service=my_execute_order, options=options)
```

### Slippage Models

```python
from tensortrade.oms.services.slippage import RandomSlippageModel

slippage = RandomSlippageModel(max_slippage=0.01)  # Up to 1%

# Slippage affects execution price:
# Order: BUY at $100,000
# Slippage: +0.5%
# Actual execution: $100,500
```

---

## Tracking Trades

### Trade History

```python
# Trades are recorded in the broker
broker = action_scheme.broker

for trade in broker.trades:
    print(f"Step {trade.step}: {trade.side} {trade.quantity} at {trade.price}")

# Output:
# Step 5: BUY 0.0999 BTC at $100,000
# Step 23: SELL 0.0999 BTC at $101,500
# Step 45: BUY 0.0985 BTC at $101,200
```

### Performance Tracking

```python
# Portfolio tracks net worth over time
for step, perf in portfolio.performance.items():
    print(f"Step {step}: ${perf['net_worth']:.2f}")

# Output:
# Step 0: $10,000.00
# Step 1: $9,990.00  (after commission)
# Step 2: $10,050.00 (price moved up)
# ...
```

---

## Common Patterns

### Full Position Switching (BSH)

```python
# Always trade 100% of balance
proportion_order(portfolio, src, tgt, proportion=1.0)
```

### Partial Position

```python
# Trade 25% of balance
proportion_order(portfolio, src, tgt, proportion=0.25)

# Allows gradual position building
# Reduces per-trade commission impact
```

### Position Sizing Based on Confidence

```python
# If agent outputs confidence 0-1
confidence = agent.predict(state)
proportion = confidence * max_position_size

proportion_order(portfolio, src, tgt, proportion=proportion)
```

---

## Key Takeaways

1. **Orders flow through ActionScheme → Broker → Exchange → Portfolio**
2. **BSH is a state machine** - trades only when desired position changes
3. **Commission is applied per trade** - configure via ExchangeOptions
4. **Wallets track balances** - locked during pending orders
5. **Portfolio aggregates wallets** - calculates net worth in base currency

---

## Checkpoint

Before continuing, verify you understand:

- [ ] How BSH decides when to trade (position change, not action)
- [ ] Where commission is applied (Exchange execution)
- [ ] How proportion_order creates trades
- [ ] Why overtrading happens with BSH (oscillation between 0 and 1)

---

## Next Steps

Now that you understand the OMS, learn about reward schemes:
- [Reward Schemes](../../03-components/02-reward-schemes.md)
- [First Training](../../04-training/01-first-training.md)
