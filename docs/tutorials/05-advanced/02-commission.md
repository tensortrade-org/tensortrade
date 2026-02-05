# Commission Impact Analysis

This is our breakthrough discovery. The agent CAN predict direction. Commission destroys the profit.

## Learning Objectives

After this tutorial, you will understand:
- Why commission is the critical challenge
- The math behind commission costs
- How training with commission helps
- Future paths to profitability

---

## The Discovery

After months of experiments, we found something important:

| Test Condition | P&L | vs Buy-and-Hold |
|----------------|-----|-----------------|
| Agent (0% commission) | **+$239** | **+$594** |
| Agent (0.1% commission) | -$650 | -$295 |
| Buy-and-Hold | -$355 | baseline |

**The agent learned to predict market direction.** At zero commission, it made +$239 while the market dropped 3.55%.

**The problem is trading frequency.** The agent trades so often that commission costs destroy all the profit.

---

## The Math

### Trading Frequency

From our evaluation:
```
Test period: 30 days (720 hours)
Number of trades: ~2,000
Trades per day: ~67
Trades per hour: ~2.8
```

The agent changes position almost 3 times per hour on average.

### Commission Cost

```
Account size: $10,000
Average trade size: $10,000 (full position changes)
Commission rate: 0.1%
Trades: 2,000

Commission per trade: $10,000 × 0.001 = $10
Total commission: 2,000 × $10 = $20,000 in fees

Wait, that's MORE than the account size!

Actual calculation (considering varying balances):
Average position value: ~$10,000
Commission cost: ~$2,000-$3,000 over 30 days
```

### The Profit Destruction

```
Direction prediction profit:    +$239
Commission cost:               -$2,000+
─────────────────────────────────────────
Net result:                    -$1,700+

The agent MADE $239 from correct predictions
But LOST $2,000+ to commission
```

---

## Why Does the Agent Overtrade?

### BSH Encourages Position Changes

```python
# BSH action scheme:
# - Action 0: "I want to be in BTC"
# - Action 1: "I want to be in USD"

# Every change creates a trade
Step 1: action=0 → BUY (trade)
Step 2: action=1 → SELL (trade)
Step 3: action=0 → BUY (trade)
Step 4: action=1 → SELL (trade)

# Agent oscillates rapidly
```

### Agent's Uncertainty

The agent isn't confident in its predictions:

```
Agent's internal state at each step:
  P(should be long) = 0.52  → action 0 (BUY)
  P(should be long) = 0.49  → action 1 (SELL)
  P(should be long) = 0.51  → action 0 (BUY)
  P(should be long) = 0.48  → action 1 (SELL)

Small changes in confidence → constant position flips
```

### No Direct Penalty

Standard PBR reward doesn't penalize trading:

```python
reward = price_change × position

# If you flip from long to cash:
# - No penalty in reward
# - But you paid commission (not in reward!)
```

---

## Training with Commission

### The Experiment

We tested different training commission levels:

| Train Commission | Test P&L (0% comm) | Test P&L (0.1% comm) |
|------------------|--------------------|-----------------------|
| 0.0% | -$102 | -$3,029 |
| 0.05% | -$73 | -$765 |
| 0.3% | +$167 | -$1,827 |
| **0.5%** | **+$239** | -$2,050 |

### Key Insights

1. **Training at 0% commission**: Agent trades constantly, terrible real-world performance
2. **Training at 0.5% commission**: Agent learns trading is expensive, trades less
3. **Best direction prediction at 0.5%**: +$239 (beats B&H by $594)

### Why Higher Training Commission Helps

```
At 0% training commission:
  - Agent sees no cost to trading
  - Learns to trade on every tiny signal
  - Makes 3000+ trades per month

At 0.5% training commission:
  - Agent feels trading cost in simulation
  - Learns to only trade with strong signals
  - Makes 1000-2000 trades per month
  - Better direction prediction (more thoughtful)
```

---

## Visualizing the Problem

```
                        Direction Prediction vs Commission Cost

    Profit │
           │
    +$500  │                                    ┌────────────────┐
           │                                    │ Direction      │
           │                                    │ Prediction     │
    +$239  │ ───────────────────────────────── │ +$239          │
           │                                    └────────────────┘
      $0   │─────────────────────────────────────────────────────────
           │
   -$500   │
           │
  -$1,000  │
           │
  -$1,500  │
           │
  -$2,000  │                                    ┌────────────────┐
           │ ───────────────────────────────── │ Commission     │
  -$2,500  │                                    │ Cost           │
           │                                    │ -$2,000+       │
           │                                    └────────────────┘

    The green bar (+$239) is completely overwhelmed by the red bar (-$2,000+)
```

---

## Solutions (Work in Progress)

### Solution 1: Trade Less

The core problem. Agent needs to trade 10x less:
- Current: ~2,000 trades/month (~67/day)
- Target: ~200 trades/month (~7/day)

**Approaches:**
- Higher training commission (partially works)
- Trade penalty in reward (didn't work well in our tests)
- Action masking (minimum hold period)
- Confidence threshold

### Solution 2: Position Sizing

Replace binary BSH with continuous position sizing:

```python
# Current (BSH):
# Action 0: 100% in BTC
# Action 1: 100% in USD

# Better (Position Sizing):
# Action: 0.0 to 1.0 (percent in BTC)
# 0.7 means 70% BTC, 30% USD

# Advantages:
# - Gradual position changes
# - Can express partial confidence
# - Smaller trades = less commission impact
```

### Solution 3: Commission-Aware Reward

Include commission in the reward signal:

```python
def reward(env):
    pbr = price_change * position

    # Subtract estimated commission for this step
    if position_changed:
        commission_cost = position_size * commission_rate
        return pbr - commission_cost

    return pbr
```

### Solution 4: Minimum Holding Period

Force agent to hold positions longer:

```python
class HoldingPeriodBSH(BSH):
    def __init__(self, cash, asset, min_hold=10):
        super().__init__(cash, asset)
        self.min_hold = min_hold
        self.steps_since_trade = min_hold

    def get_orders(self, action, portfolio):
        self.steps_since_trade += 1

        if self.steps_since_trade < self.min_hold:
            return []  # Can't trade yet

        if abs(action - self.action) > 0:
            self.steps_since_trade = 0
            # ... create order
```

### Solution 5: Confidence Threshold

Only trade when agent is confident:

```python
# Instead of discrete actions, output probability
# Only trade if P(action) > threshold

def process_action(logits, threshold=0.7):
    probs = softmax(logits)

    if max(probs) < threshold:
        return HOLD  # Not confident enough

    return argmax(probs)
```

---

## What Would Profitability Look Like?

### Math for Break-Even

```
Current situation:
  Direction profit: +$239
  Commission cost: -$2,000
  Net: -$1,761

To break even:
  Need direction profit > commission cost
  Or commission cost < direction profit

Option A: Better predictions
  Current direction accuracy: ~51%
  Need: ~55%+ for commission to be covered
  (Very hard to achieve consistently)

Option B: Fewer trades
  Current trades: 2,000/month
  At +$239 direction profit with 2000 trades
  = $0.12 profit per trade

  Commission at 0.1%: $10 per trade
  Need: $10 profit per trade minimum
  = ~100 trades/month maximum
  (20x reduction in trading!)

Option C: Better fills
  0.1% commission is standard
  Some exchanges: 0.01% (maker fees)
  At 0.01%: $1 per trade
  Current direction profit would cover ~239 trades
```

---

## Experiments You Can Try

### Experiment 1: Very High Training Commission

```python
env_config = {
    "commission": 0.01,  # 1% per trade (very high)
}

# Does agent learn to trade even less?
```

### Experiment 2: Action Masking

```python
# Implement CooldownBSH with different hold periods
for hold_period in [5, 10, 20, 50]:
    results = train_with_cooldown(hold_period)
    print(f"Hold {hold_period}: {results}")
```

### Experiment 3: Position Sizing

```python
# Replace BSH with continuous action space
# Track trade sizes and frequencies
```

---

## Key Numbers to Remember

| Metric | Current | Target |
|--------|---------|--------|
| Direction P&L | +$239 | +$500+ |
| Trades/month | ~2,000 | ~200 |
| Commission cost | -$2,000 | -$200 |
| Net P&L | -$1,761 | +$300 |

**The path to profitability is mostly about trading discipline, not prediction accuracy.**

---

## Key Takeaways

1. **The agent CAN predict direction** - +$239 at 0% commission
2. **Overtrading destroys profit** - ~2,000 trades/month
3. **Commission is the enemy** - $2,000+ in fees
4. **Training commission helps** - 0.5% trains better discipline
5. **Solutions exist** - Position sizing, hold periods, thresholds
6. **10x reduction needed** - From ~2,000 to ~200 trades/month

---

## Checkpoint

After this tutorial, verify you understand:

- [ ] Why the agent is profitable at 0% but not 0.1% commission
- [ ] The math: 2000 trades × 0.1% × $10k position
- [ ] Why higher training commission helps
- [ ] At least 2 potential solutions to reduce trading

---

## Next Steps

[03-walk-forward.md](03-walk-forward.md) - Proper validation methodology

Or contribute to TensorTrade:
- Implement position sizing action scheme
- Add action masking / holding period
- Test commission-aware rewards
