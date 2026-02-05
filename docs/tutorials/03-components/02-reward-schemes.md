# Reward Schemes

Reward schemes compute the learning signal for your agent. This is arguably the most important component - a bad reward function can make learning impossible.

## Learning Objectives

After this tutorial, you will understand:
- Why reward design matters so much
- How PBR (Position-Based Returns) works
- Alternative reward schemes and their problems
- How to avoid reward hacking

---

## Why Rewards Matter

The reward is all the agent optimizes for. If your reward doesn't align with your goal, the agent will find unexpected ways to maximize it.

```
Your Goal: Maximize profit
Reward: Number of winning trades

Agent learns: Make many tiny winning trades
Reality: Commission destroys profit
Result: Agent maximizes reward but loses money
```

---

## PBR: The Recommended Reward Scheme

### The Formula

```
R_t = (P_t - P_{t-1}) × position_t

Where:
  P_t = price at time t
  position_t = +1 if long, -1 if cash
```

### Why It Works

```
Scenario 1: Long position, price goes up
  Price change: $100,050 - $100,000 = +$50
  Position: +1 (long)
  Reward: +$50 × 1 = +$50
  Correct! Agent is rewarded for being long during up move.

Scenario 2: Long position, price goes down
  Price change: $99,950 - $100,000 = -$50
  Position: +1 (long)
  Reward: -$50 × 1 = -$50
  Correct! Agent is penalized for being long during down move.

Scenario 3: Cash position, price goes up
  Price change: $100,050 - $100,000 = +$50
  Position: -1 (cash)
  Reward: +$50 × (-1) = -$50
  Correct! Agent missed the up move, penalized for not being long.

Scenario 4: Cash position, price goes down
  Price change: $99,950 - $100,000 = -$50
  Position: -1 (cash)
  Reward: -$50 × (-1) = +$50
  Correct! Agent avoided the loss, rewarded for being in cash.
```

### Code Implementation

```python
class PBR(TensorTradeRewardScheme):
    """Position-Based Returns reward scheme."""

    def __init__(self, price: 'Stream') -> None:
        super().__init__()
        self.position = -1  # Start in cash (position = -1)

        # Create reward stream
        # r = price difference from step to step
        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()

        # position stream (reads from self.position)
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")

        # reward = position × price_change
        reward = (position * r).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int) -> None:
        # BSH calls this when action is taken
        # action 0 = long → position = +1
        # action 1 = cash → position = -1
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio') -> float:
        return self.feed.next()["reward"]
```

### The Key Insight

PBR gives a signal EVERY step, not just when you trade.

```
Compare to "reward on trade only":

Step 1: HOLD                    │ PBR: -$20 (missed down move in cash)
Step 2: HOLD                    │ PBR: +$30 (avoided up move... wait)
Step 3: BUY (finally trade)    │ Old: +$0 (trade executed)
Step 4: HOLD                    │ PBR: +$15 (price went up while long)
Step 5: HOLD                    │ PBR: -$10 (price went down while long)

Old reward: Gets signal only at step 3
PBR: Gets signal EVERY step
→ PBR learns 5x faster!
```

---

## SimpleProfit: The Naive Approach

### The Formula

```python
reward = (net_worth_t - net_worth_{t-1}) / net_worth_{t-1}
```

### The Problem

```
Step 1: Hold (no trade) → net_worth unchanged → reward = 0
Step 2: Hold (no trade) → net_worth unchanged → reward = 0
Step 3: Price moves but position unchanged → reward = 0
...
Agent gets no signal! Can't learn anything.

Only when trade happens:
Step 10: Trade → net_worth changes → reward ≠ 0
But this is very sparse signal.
```

### When SimpleProfit Works

- If agent trades every step (not realistic)
- As a validation metric (not training reward)

---

## RiskAdjustedReturns: Sounds Good, Doesn't Work

### The Idea

Reward risk-adjusted performance (Sharpe ratio):

```python
sharpe = (mean_return - risk_free_rate) / std_return
```

### Why It Fails for RL

```
Problem 1: Needs multiple samples
  Single step Sharpe ratio is meaningless
  Need window of returns → reward is delayed

Problem 2: Noisy signal
  Small window → very noisy Sharpe
  Large window → very delayed signal

Problem 3: Gaming
  Agent learns: "Make one trade, then do nothing"
  One winning trade = infinite Sharpe (zero std)
```

### Experimental Results

```
PBR Reward:            Test P&L -$650
Sharpe Ratio Reward:   Test P&L -$3,174  (much worse!)
```

---

## AdvancedPBR: Addressing Overtrading

### The Idea

Combine PBR with penalties for trading:

```python
R_t = pbr_weight × PBR + trade_penalty × |position_change| + hold_bonus × is_holding

Where:
  pbr_weight = 1.0 (standard PBR)
  trade_penalty = -0.001 (penalty for changing position)
  hold_bonus = 0.0001 (reward for not trading)
```

### Implementation

```python
class AdvancedPBR(TensorTradeRewardScheme):
    def __init__(
        self,
        price: 'Stream',
        pbr_weight: float = 1.0,
        trade_penalty: float = -0.001,
        hold_bonus: float = 0.0001,
        volatility_threshold: float = 0.001
    ):
        super().__init__()
        self.position = -1
        self.prev_action = 0
        self.pbr_weight = pbr_weight
        self.trade_penalty = trade_penalty
        self.hold_bonus = hold_bonus
        # ... setup streams

    def on_action(self, action: int) -> None:
        self.action_changed = (action != self.prev_action)
        self.prev_action = action
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio') -> float:
        data = self.feed.next()
        pbr_reward = data["pbr_reward"]

        # 1. PBR component
        reward = self.pbr_weight * pbr_reward

        # 2. Trading penalty
        if self.action_changed:
            reward += self.trade_penalty  # Negative value

        # 3. Hold bonus in flat markets
        if not self.action_changed:
            reward += self.hold_bonus

        return reward
```

### Results

```
Standard PBR:     ~2000 trades/month
AdvancedPBR:      ~1800 trades/month (slight reduction)

Conclusion: Direct penalty didn't reduce trading much.
Better approach: Train with higher commission.
```

---

## Comparing Reward Schemes

| Scheme | Signal Frequency | Learning Speed | Best Use |
|--------|-----------------|----------------|----------|
| **PBR** | Every step | Fast | Direction prediction |
| **SimpleProfit** | On trade only | Slow | Validation metric |
| **RiskAdjustedReturns** | Delayed | Very slow | Don't use for training |
| **AdvancedPBR** | Every step | Fast | Experimental |

---

## Common Reward Pitfalls

### Pitfall 1: Sparse Rewards

```python
# BAD: Only reward at end of episode
def reward(env):
    if env.done:
        return env.portfolio.profit
    return 0

# Agent gets one signal per 720 steps
# Learning is impossibly slow
```

### Pitfall 2: Reward Doesn't Match Goal

```python
# Goal: Maximize profit
# BAD: Reward number of trades
def reward(env):
    return env.trade_count

# Agent learns: Trade as much as possible
# Reality: Commission destroys profit
```

### Pitfall 3: Reward Hacking

```python
# BAD: Reward Sharpe ratio
def reward(env):
    if len(env.returns) < 2:
        return 0
    return np.mean(env.returns) / (np.std(env.returns) + 1e-8)

# Agent learns: Make one tiny profitable trade, then stop
# Sharpe = small_profit / 0 = infinity (until regularized)
```

### Pitfall 4: Ignoring Commission

```python
# BAD: Reward gross profit
def reward(env):
    return env.position_profit  # Before commission

# Agent ignores trading costs
# Learns strategy that's only profitable with 0 commission
```

---

## Best Practices

### 1. Use PBR for Training

```python
reward_scheme = PBR(price=price)
```

### 2. Track Multiple Metrics

```python
# Even if training with PBR, track:
# - Actual P&L (net of commission)
# - Number of trades
# - Max drawdown
# - Sharpe ratio (for validation, not reward)
```

### 3. Test Reward Logic Before Training

```python
# Verify reward makes sense
for action in [0, 1]:
    for price_change in [-100, 0, 100]:
        reward = compute_reward(action, price_change)
        print(f"Action {action}, price Δ{price_change}: reward {reward}")

# Should see:
# Action 0 (long), price +100: reward +100 (correct!)
# Action 0 (long), price -100: reward -100 (correct!)
# Action 1 (cash), price +100: reward -100 (missed up, correct!)
# Action 1 (cash), price -100: reward +100 (avoided down, correct!)
```

### 4. Include Commission in Training

```python
# During training, use realistic commission
exchange_options = ExchangeOptions(commission=0.001)

# Or higher to encourage less trading
exchange_options = ExchangeOptions(commission=0.005)
```

---

## Custom Reward Schemes

### Template

```python
from tensortrade.env.generic import RewardScheme

class MyRewardScheme(RewardScheme):
    def __init__(self, price_stream, ...):
        super().__init__()
        # Store parameters
        # Setup any streams needed

    def on_action(self, action: int) -> None:
        # Called by ActionScheme when action taken
        # Update internal state
        pass

    def reward(self, env: 'TradingEnv') -> float:
        # Compute and return reward
        # Can access env.portfolio, env.observer, etc.
        pass

    def reset(self) -> None:
        # Reset state between episodes
        pass
```

### Example: Commission-Aware PBR

```python
class CommissionAwarePBR(TensorTradeRewardScheme):
    """PBR that subtracts commission from reward."""

    def __init__(self, price, commission=0.001):
        super().__init__()
        self.commission = commission
        self.position = -1
        self.prev_action = 0

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")
        pbr = (position * r).fillna(0).rename("pbr")

        self.feed = DataFeed([pbr])
        self.feed.compile()

    def on_action(self, action: int) -> None:
        self.position_changed = (action != self.prev_action)
        self.prev_action = action
        self.position = 1 if action == 0 else -1

    def get_reward(self, portfolio: 'Portfolio') -> float:
        pbr = self.feed.next()["pbr"]

        # Subtract estimated commission cost on trade
        if self.position_changed:
            trade_cost = portfolio.net_worth * self.commission
            return pbr - trade_cost

        return pbr
```

---

## Key Takeaways

1. **PBR is the best reward for trading** - Dense signal, fast learning
2. **Reward every step** - Sparse rewards = slow learning
3. **Sharpe ratio fails for RL** - Don't use for training reward
4. **Test reward logic manually** - Verify it makes sense before training
5. **Commission matters** - Include realistic costs in training

---

## Checkpoint

Before continuing, verify you understand:

- [ ] Why PBR gives reward every step (not just on trades)
- [ ] What position values +1 and -1 mean in PBR
- [ ] Why Sharpe ratio reward doesn't work for RL
- [ ] How to verify your reward logic makes sense

---

## Next Steps

[03-observers-feeds.md](03-observers-feeds.md) - Learn about feature engineering and data pipelines
