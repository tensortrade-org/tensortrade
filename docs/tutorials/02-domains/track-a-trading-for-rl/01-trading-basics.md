# Trading Basics for RL Practitioners

You know agents, rewards, and policies. This tutorial explains the trading domain you're stepping into.

## Learning Objectives

After this tutorial, you will understand:
- How orders and execution work
- What commission and slippage are
- How portfolios and positions work
- Why P&L is different from reward

---

## Markets: Where Trading Happens

### The Order Book

When you trade, you're interacting with an order book:

```
             ORDER BOOK (BTC/USD)
────────────────────────────────────────
        ASKS (Sellers)
        $100,150 | 0.5 BTC
        $100,100 | 1.2 BTC
        $100,050 | 0.8 BTC   ← Best ask
────────── SPREAD ($50) ──────────────
        $100,000 | 2.0 BTC   ← Best bid
        $99,950  | 1.5 BTC
        $99,900  | 3.0 BTC
        BIDS (Buyers)
────────────────────────────────────────
```

**Key concepts**:
- **Bid**: Highest price buyers will pay
- **Ask**: Lowest price sellers will accept
- **Spread**: Gap between bid and ask
- **Liquidity**: How much volume at each price

In TensorTrade, we simulate this with a single price stream (simplified).

---

## Orders: How You Trade

### Market Order

"Buy/sell NOW at whatever price is available"

```python
# You want to buy 0.1 BTC immediately
order = MarketOrder(side=BUY, quantity=0.1 * BTC)

# Execution:
# - Fills at best ask price ($100,050)
# - You pay $10,005 + commission
```

**Pros**: Guaranteed execution
**Cons**: Might get bad price if market is thin

### Limit Order

"Buy/sell only if price reaches my target"

```python
# You want to buy 0.1 BTC but only at $99,900 or better
order = LimitOrder(side=BUY, price=99900, quantity=0.1 * BTC)

# Execution:
# - Only fills if price drops to $99,900
# - Might never fill if price goes up
```

**Pros**: Control your price
**Cons**: Might not execute

TensorTrade primarily uses market orders (immediate execution).

---

## Commission: The Cost of Trading

### What It Is

Every trade costs a fee. This is how exchanges make money.

```
Trade: Buy $10,000 of BTC
Commission rate: 0.1%
Commission paid: $10,000 × 0.001 = $10

You wanted $10,000 of BTC
You actually get: $9,990 worth of BTC
```

### Why It Matters for RL

```
Scenario: Agent makes many small profitable trades

Trade 1: +$5 profit, -$10 commission = -$5 net
Trade 2: +$8 profit, -$10 commission = -$2 net
Trade 3: +$12 profit, -$10 commission = +$2 net
Trade 4: +$3 profit, -$10 commission = -$7 net

Total profit from predictions: +$28
Total commission paid: -$40
Net P&L: -$12 LOSS

The agent was RIGHT on direction but LOST money!
```

**TensorTrade default**: 0.1% commission (configurable)

```python
exchange_options = ExchangeOptions(commission=0.001)  # 0.1%
```

---

## Slippage: Price Movement During Execution

### What It Is

Between deciding to trade and execution, price can move.

```
You see: BTC at $100,000
You decide: BUY
Time passes: 100ms
Execution: BTC now at $100,050
You paid: $100,050 (not $100,000)

Slippage: $50 (0.05%)
```

### In TensorTrade

TensorTrade can simulate slippage:

```python
from tensortrade.oms.services.slippage import RandomSlippageModel

slippage_model = RandomSlippageModel(max_slippage=0.01)  # Up to 1%
```

For training, we often ignore slippage to focus on strategy learning.

---

## Positions: What You Own

### Long Position

You own the asset. You profit when price goes UP.

```
Buy 0.1 BTC at $100,000 (cost: $10,000)
Price rises to $110,000
Your 0.1 BTC now worth: $11,000
Profit: +$1,000 (10%)
```

### Cash Position (Flat)

You hold only cash. You don't profit or lose from price movement.

```
Hold $10,000 USD
BTC price rises from $100,000 to $110,000
Your USD still worth: $10,000
Profit: $0
```

### Short Position

You borrow asset and sell it. You profit when price goes DOWN.

```
Borrow 0.1 BTC, sell at $100,000 (receive: $10,000)
Price drops to $90,000
Buy back 0.1 BTC for $9,000
Return borrowed BTC
Profit: +$1,000
```

**Note**: TensorTrade's BSH doesn't support true shorting. Action 1 = cash, not short.

---

## Portfolio: Your Account

### Components

```
┌─────────────────────────────────────────┐
│               Portfolio                  │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │         USD Wallet              │   │
│  │         Balance: $5,000         │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │         BTC Wallet              │   │
│  │         Balance: 0.05 BTC       │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Current BTC Price: $100,000           │
│  BTC Value: 0.05 × $100,000 = $5,000   │
│                                         │
│  Net Worth = $5,000 + $5,000 = $10,000 │
└─────────────────────────────────────────┘
```

### Net Worth Calculation

```python
net_worth = usd_balance + (btc_balance × btc_price)
```

Net worth changes when:
1. Price of held assets changes
2. You trade (commission reduces net worth)
3. You receive deposits/make withdrawals

---

## P&L vs Reward

### They're Different!

**P&L (Profit & Loss)**: Real money gained/lost
```
P&L = Final Net Worth - Initial Net Worth
```

**Reward**: Learning signal for the agent
```
Reward = f(state, action, next_state)  # Could be anything
```

### Example

```
Initial: $10,000
Final: $9,500
P&L: -$500

But during the episode:
- Agent correctly predicted 5 price moves: reward +500
- Agent incorrectly predicted 3 price moves: reward -300
- Agent held during crash (avoided loss): reward +200
Total Reward: +400

P&L: -$500 (lost money)
Reward: +400 (learned useful behaviors)
```

### Why This Matters

You can have:
- Positive reward + negative P&L (learning but not profitable yet)
- Positive P&L + negative reward (got lucky, bad strategy)
- High reward + commission destroying profit

Always track BOTH during training.

---

## Trading Metrics

### Return

```
Return = (Final - Initial) / Initial × 100%

Example: $10,000 → $10,500
Return = 500 / 10000 × 100 = 5%
```

### Drawdown

Maximum drop from a peak.

```
Peak: $12,000
Trough: $9,000
Drawdown = (12000 - 9000) / 12000 = 25%

┌──────────────────────────────────┐
│   $12,000 ──── PEAK            │
│         \                       │
│          \                      │
│           \  ← 25% Drawdown    │
│            \                    │
│   $9,000 ─── TROUGH            │
└──────────────────────────────────┘
```

### Sharpe Ratio

Risk-adjusted return.

```
Sharpe = (Average Return - Risk Free Rate) / Std(Returns)

Sharpe > 1: Good
Sharpe > 2: Excellent
Sharpe < 0: Losing money
```

**Warning**: Sharpe ratio doesn't work well as RL reward (see pitfalls).

---

## TensorTrade's Order Management System (OMS)

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                      OMS Architecture                        │
│                                                             │
│   Agent Decision                                            │
│         │                                                   │
│         v                                                   │
│   ┌───────────────┐                                        │
│   │ ActionScheme  │ Converts action to Order               │
│   └───────┬───────┘                                        │
│           │                                                 │
│           v                                                 │
│   ┌───────────────┐                                        │
│   │    Broker     │ Manages order lifecycle                │
│   └───────┬───────┘                                        │
│           │                                                 │
│           v                                                 │
│   ┌───────────────┐                                        │
│   │   Exchange    │ Executes order, applies commission     │
│   └───────┬───────┘                                        │
│           │                                                 │
│           v                                                 │
│   ┌───────────────┐                                        │
│   │   Portfolio   │ Updates wallet balances                │
│   └───────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Code Example

```python
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.services.execution.simulated import execute_order

# Create exchange with 0.1% commission
exchange_options = ExchangeOptions(commission=0.001)
exchange = Exchange("sim", service=execute_order, options=exchange_options)

# Create wallets
usd_wallet = Wallet(exchange, 10000 * USD)  # $10,000
btc_wallet = Wallet(exchange, 0 * BTC)      # 0 BTC

# Create portfolio
portfolio = Portfolio(USD, [usd_wallet, btc_wallet])

# Check status
print(portfolio.net_worth)  # $10,000
```

---

## Practical Exercise

Calculate the outcome:

```
Initial: $10,000 USD, 0 BTC
BTC Price: $100,000
Commission: 0.1%

Step 1: BUY (convert all USD to BTC)
  - USD spent: $10,000
  - Commission: $10 (0.1%)
  - BTC received: $9,990 / $100,000 = 0.0999 BTC
  - New balance: $0 USD, 0.0999 BTC
  - Net worth: 0.0999 × $100,000 = $9,990

Step 2: Price rises to $105,000
  - Balance: $0 USD, 0.0999 BTC
  - Net worth: 0.0999 × $105,000 = $10,489.50
  - Unrealized P&L: +$489.50

Step 3: SELL (convert all BTC to USD)
  - BTC sold: 0.0999 × $105,000 = $10,489.50
  - Commission: $10.49 (0.1%)
  - USD received: $10,479.01
  - New balance: $10,479.01 USD, 0 BTC
  - Net worth: $10,479.01

Final P&L: $10,479.01 - $10,000 = +$479.01
(Not $500 due to commission!)
```

---

## Key Takeaways

1. **Commission reduces every trade** - 0.1% adds up fast with frequent trading
2. **P&L ≠ Reward** - Track both, optimize for P&L
3. **Net worth fluctuates with prices** - Even without trading
4. **Long = profit when up**, Cash = no exposure to price movement
5. **Order execution is immediate** in TensorTrade (market orders)

---

## Checkpoint

Before continuing, make sure you understand:

- [ ] How commission affects each trade
- [ ] Why frequent trading is expensive
- [ ] What net worth calculation includes
- [ ] The difference between P&L and reward

---

## Next Steps

[02-oms-deep-dive.md](02-oms-deep-dive.md) - Deep dive into TensorTrade's Order Management System
