# Track C: Full Introduction

New to both RL and trading? Start here.

## What You'll Learn

This track covers:
1. Basic trading concepts (what is buying/selling)
2. Basic RL concepts (what is an agent)
3. How TensorTrade combines them
4. Your first training run

## Prerequisites

- Python programming experience
- Basic math (percentages, basic algebra)
- No prior trading or ML experience required

## The Journey

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   START HERE                                                │
│       │                                                     │
│       v                                                     │
│   ┌───────────────────────────────────────────────────────┐│
│   │ What is Trading?                                      ││
│   │ - Buying and selling assets                           ││
│   │ - Making money from price changes                     ││
│   │ - Markets and exchanges                               ││
│   └───────────────────────────────────────────────────────┘│
│       │                                                     │
│       v                                                     │
│   ┌───────────────────────────────────────────────────────┐│
│   │ What is RL?                                           ││
│   │ - Learning from trial and error                       ││
│   │ - Agent, environment, reward                          ││
│   │ - How policies are learned                            ││
│   └───────────────────────────────────────────────────────┘│
│       │                                                     │
│       v                                                     │
│   ┌───────────────────────────────────────────────────────┐│
│   │ TensorTrade Overview                                  ││
│   │ - How the pieces fit together                         ││
│   │ - Running your first script                           ││
│   │ - Understanding the output                            ││
│   └───────────────────────────────────────────────────────┘│
│       │                                                     │
│       v                                                     │
│   CONTINUE TO CORE TUTORIALS                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## What is Trading?

### The Basic Idea

Trading is buying something hoping it will be worth more later.

```
You buy: 1 Bitcoin at $100,000
Time passes...
Bitcoin is now: $110,000
You sell: 1 Bitcoin at $110,000

Profit: $110,000 - $100,000 = $10,000
```

### The Catch: You Can Also Lose

```
You buy: 1 Bitcoin at $100,000
Time passes...
Bitcoin drops to: $90,000
You sell: 1 Bitcoin at $90,000

Loss: $90,000 - $100,000 = -$10,000
```

### The Goal

Predict which direction prices will move:
- If you think price goes UP → BUY first, sell later
- If you think price goes DOWN → Don't buy (or sell if you have it)

---

## What is Reinforcement Learning?

### The Basic Idea

Teach a computer to make decisions by rewarding good outcomes.

```
Like training a dog:
- Dog sits → Give treat (positive reward)
- Dog jumps on table → No treat (no reward)
- Over time, dog learns: sitting = treats

Like training a trader:
- Agent buys → Price goes up → Positive reward
- Agent buys → Price goes down → Negative reward
- Over time, agent learns: when to buy
```

### The Key Terms

| Term | Meaning | In TensorTrade |
|------|---------|----------------|
| **Agent** | The learner/decision maker | A neural network |
| **Environment** | Where the agent acts | Simulated market |
| **State** | What the agent sees | Market data (prices, indicators) |
| **Action** | What the agent does | Buy, sell, or hold |
| **Reward** | Feedback signal | Profit or loss |

### The Loop

```
1. Agent sees market state (prices, indicators)
2. Agent decides: BUY, SELL, or HOLD
3. Environment shows result (price moved up/down)
4. Agent receives reward (profit or loss)
5. Agent adjusts strategy based on reward
6. Repeat thousands of times

After many iterations, agent learns patterns:
"When RSI is low AND trend is up, buying usually works"
```

---

## How TensorTrade Works

### The Setup

```python
# 1. Get market data
data = fetch_btc_data()  # Historical prices

# 2. Create simulated market
exchange = Exchange(commission=0.1%)  # 0.1% fee per trade
portfolio = Portfolio(starting_cash=$10,000)

# 3. Create the RL environment
env = TradingEnv(
    data=data,
    portfolio=portfolio,
    reward_scheme=PBR,  # Reward based on position returns
)

# 4. Train the agent
agent = PPO()  # The learning algorithm
agent.train(env, iterations=100)

# 5. Test the agent
results = agent.evaluate(test_data)
print(f"Profit: ${results.profit}")
```

### What Happens During Training

```
Episode 1 (random behavior):
  Agent: BUY, SELL, BUY, SELL, HOLD, BUY...
  Result: Lost $500 (random trading loses money)
  Reward: Negative

Episode 10 (starting to learn):
  Agent: BUY when RSI low, HOLD when uncertain...
  Result: Lost $200 (still learning)
  Reward: Less negative

Episode 100 (learned patterns):
  Agent: BUY at support, SELL at resistance...
  Result: Gained $100 (patterns are working!)
  Reward: Positive

Episode 1000 (refined strategy):
  Agent: Trades only with high confidence...
  Result: Gained $239 (direction prediction works!)
  Reward: Consistently positive
```

---

## Your First Run

### Step 1: Install

```bash
# Create Python environment
python3.12 -m venv tensortrade-env
source tensortrade-env/bin/activate

# Install TensorTrade
pip install -r requirements.txt
pip install -e .
```

### Step 2: Run Demo

```bash
python examples/training/train_simple.py
```

### Step 3: Understand Output

```
======================================================================
Episode 1/5
======================================================================
 Step | Action |    USD Balance |    BTC Balance |    Net Worth |     Reward
----------------------------------------------------------------------
    1 | BUY  |         $0.00 |      0.099800 |   $10,000.00 |      +0.00
```

What this means:
- **Step 1**: First hour of trading
- **Action BUY**: Agent decided to buy Bitcoin
- **USD Balance $0**: All cash was used to buy BTC
- **BTC Balance 0.099800**: How much Bitcoin we now have
- **Net Worth $10,000**: Total value (BTC value + cash)
- **Reward +0.00**: No reward yet (price hasn't changed)

---

## The Big Discovery

After all our experiments, we found:

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   The agent CAN predict market direction!           │
│                                                     │
│   At 0% commission:  +$239 PROFIT                   │
│   At 0.1% commission: -$650 LOSS                    │
│                                                     │
│   The problem: Too many trades                      │
│   2000 trades × 0.1% commission = $2000 in fees    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

The challenge isn't prediction - it's **trading discipline**.

---

## What's Next?

You have two options:

### Option A: Learn More Theory First

1. [Trading for RL People](../track-a-trading-for-rl/01-trading-basics.md) - Deeper trading concepts
2. [RL for Traders](../track-b-rl-for-traders/01-rl-fundamentals.md) - Deeper RL concepts

### Option B: Jump Into Practice

1. [Your First Run](../../01-foundations/03-your-first-run.md) - Detailed walkthrough
2. [First Training](../../04-training/01-first-training.md) - Train a real agent

---

## Quick Glossary

| Term | Simple Definition |
|------|-------------------|
| **Agent** | The AI that makes trading decisions |
| **Episode** | One complete trading simulation |
| **Commission** | Fee paid for each trade |
| **P&L** | Profit and Loss (money made/lost) |
| **Reward** | Learning signal given to agent |
| **BSH** | Buy/Sell/Hold action scheme |
| **PBR** | Position-Based Returns (reward method) |
| **PPO** | The RL algorithm we use |
| **Overfitting** | Memorizing training data, failing on new data |
| **Overtrading** | Trading too often, paying too much in fees |

---

## Key Takeaways

1. **Trading** = buying low, selling high (hopefully)
2. **RL** = learning from rewards
3. **TensorTrade** = simulated market + RL agent
4. **The challenge** = not prediction, but trading frequency
5. **Commission** = the profit killer

Welcome to TensorTrade! Start with the foundations tutorials.
