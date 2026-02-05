# Your First Run

This tutorial walks you through running `train_simple.py` and understanding every line of output.

## Learning Objectives

After this tutorial, you will:
- Successfully run your first TensorTrade script
- Understand what each output line means
- Know how to interpret wallet balances and P&L
- Be ready to customize your first experiment

---

## Prerequisites

Make sure TensorTrade is installed:

```bash
# From the tensortrade directory
pip install -e .

# Verify
python -c "import tensortrade; print('OK')"
```

---

## Run the Script

```bash
python examples/training/train_simple.py
```

You should see output like this:

```
======================================================================
TensorTrade Training Demo - Showing Wallet Balances
======================================================================

Fetching BTC/USD data...
Using 200 rows | Price range: $98,234 - $101,456

======================================================================
Creating Environment...
======================================================================
Initial Cash:  $10,000.00 USD
Initial Asset: 0.00000000 BTC
```

Let's break down what's happening.

---

## Understanding the Output

### 1. Data Fetching

```
Fetching BTC/USD data...
Using 200 rows | Price range: $98,234 - $101,456
```

**What's happening**:
- TensorTrade fetches historical BTC/USD data from CryptoDataDownload
- We use the last 200 hourly candles (~8 days of data)
- Price range shows the min/max close prices in that period

**Code responsible**:
```python
cdd = CryptoDataDownload()
data = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
data = data.tail(200).reset_index(drop=True)
```

---

### 2. Environment Creation

```
======================================================================
Creating Environment...
======================================================================
Initial Cash:  $10,000.00 USD
Initial Asset: 0.00000000 BTC
```

**What's happening**:
- Creating wallets: one for USD, one for BTC
- Starting with $10,000 cash and 0 BTC
- Net worth = $10,000 (all in cash)

**Code responsible**:
```python
initial_cash = 10000
cash = Wallet(exchange, initial_cash * USD)
asset = Wallet(exchange, 0 * BTC)
portfolio = Portfolio(USD, [cash, asset])
```

---

### 3. Episode Output

```
======================================================================
Episode 1/5
======================================================================
 Step | Action |    USD Balance |    BTC Balance |    Net Worth |     Reward
----------------------------------------------------------------------
    1 | BUY  |         $0.00 |      0.099800 |   $10,000.00 |      +0.00
   30 | HOLD |         $0.00 |      0.099800 |    $9,856.21 |     -12.45
   45 | SELL |     $9,812.34 |      0.000000 |    $9,812.34 |     +43.87
```

Let's decode each column:

| Column | Meaning |
|--------|---------|
| **Step** | Current time step (hour) |
| **Action** | What the agent decided: BUY, SELL, or HOLD |
| **USD Balance** | How much cash in the USD wallet |
| **BTC Balance** | How much BTC in the asset wallet |
| **Net Worth** | USD + (BTC × current price) |
| **Reward** | PBR reward for this step |

---

### 4. First Trade (BUY)

```
    1 | BUY  |         $0.00 |      0.099800 |   $10,000.00 |      +0.00
```

**What happened**:
1. Agent output action `0` (BUY in BSH scheme)
2. ActionScheme converted all USD to BTC
3. $10,000 → ~0.0998 BTC (after 0.1% commission)
4. Net worth stays ~$10,000 (BTC value = original USD)
5. Reward = 0 (no price movement yet)

**The math**:
```
BTC price: ~$100,200
Commission: 0.1%

$10,000 - (0.1% commission) = $9,990 to convert
$9,990 / $100,200 = 0.0998 BTC
```

---

### 5. Holding Period (HOLD)

```
   30 | HOLD |         $0.00 |      0.099800 |    $9,856.21 |     -12.45
```

**What happened**:
1. Agent output action `0` again (stay in BTC)
2. Since position didn't change, no trade executed
3. BTC price dropped, so net worth decreased
4. Negative reward because price fell while holding BTC

**Understanding PBR reward**:
```
PBR = (price_change) × position

Price fell ~$1.44 per BTC hour-over-hour
Position = +1 (long BTC)
Reward = -$1.44 × 0.0998 BTC ≈ -$0.14 (scaled up for learning)
```

---

### 6. Exit Trade (SELL)

```
   45 | SELL |     $9,812.34 |      0.000000 |    $9,812.34 |     +43.87
```

**What happened**:
1. Agent output action `1` (SELL/go to cash)
2. ActionScheme sold all BTC for USD
3. 0.0998 BTC × price - commission = $9,812.34
4. Now fully in cash (protected from further price drops)

---

### 7. Episode Summary

```
----------------------------------------------------------------------
Episode Summary:
  Steps: 150 | Trades: 12 | Total Reward: 156.23
  Initial: $10,000.00 -> Final: $9,743.21
  P&L: $-256.79 (-2.57%)
```

| Metric | Meaning |
|--------|---------|
| **Steps** | Total time steps in episode |
| **Trades** | Number of position changes (BUY or SELL) |
| **Total Reward** | Sum of all PBR rewards |
| **Initial/Final** | Starting and ending net worth |
| **P&L** | Profit & Loss (Final - Initial) |

**Key insight**: 12 trades in 150 steps = trading every ~12.5 hours on average. This is the overtrading problem.

---

## Why Did We Lose Money?

This demo uses a **random policy** (not a trained agent):

```python
# From train_simple.py
rand = np.random.random()
if rand < 0.15:
    action = 0  # Buy (15% chance)
elif rand < 0.30:
    action = 1  # Sell (15% chance)
else:
    action = 2  # Hold (70% chance)
```

Random trading + commission = guaranteed loss over time.

A trained RL agent would learn to:
1. Only trade when confident about direction
2. Hold positions longer to reduce commission costs

---

## The Episode Loop Code

Here's the core loop annotated:

```python
while not done and not truncated and step < 150:
    # 1. Agent picks action (random in this demo)
    rand = np.random.random()
    if rand < 0.15:
        action = 0  # Buy
    elif rand < 0.30:
        action = 1  # Sell
    else:
        action = 2  # Hold

    # 2. Environment executes action and returns new state
    obs, reward, done, truncated, info = env.step(action)

    # 3. Track cumulative reward
    total_reward += reward
    step += 1

    # 4. Get current wallet states
    usd_bal = cash.balance.as_float()
    btc_bal = asset.balance.as_float()
    worth = portfolio.net_worth
```

---

## Hands-On Exercise

Try modifying the script:

### Exercise 1: Change Initial Capital

Edit `train_simple.py`:
```python
initial_cash = 50000  # Was 10000
```

Run again. Does P&L percentage change?

### Exercise 2: Change Commission

```python
exchange_options = ExchangeOptions(commission=0.005)  # 0.5% instead of 0.1%
```

Run again. How does higher commission affect results?

### Exercise 3: Change Action Probabilities

```python
if rand < 0.05:       # Less buying (was 0.15)
    action = 0
elif rand < 0.10:     # Less selling (was 0.30)
    action = 1
else:
    action = 2        # More holding
```

Run again. Does fewer trading improve results?

---

## Expected Observations

After running multiple times, you should notice:

1. **P&L varies** - random actions mean random results
2. **More trades = more commission** - each trade costs ~0.1%
3. **Holding more reduces losses** - from commission alone
4. **Net worth ≠ total reward** - reward is learning signal, not actual money

---

## What's Next?

This demo showed the mechanics. To actually **learn** to trade profitably, we need:

1. **Real RL training** - Use Ray RLlib or other RL libraries
2. **Better features** - Not just OHLCV, but derived indicators
3. **Hyperparameter tuning** - Find the right learning rate, etc.
4. **Evaluation** - Test on data the agent hasn't seen

---

## Checkpoint

Before continuing, verify you can:

- [ ] Run `train_simple.py` without errors
- [ ] Explain what BUY, SELL, HOLD mean in BSH
- [ ] Calculate why commission reduces net worth
- [ ] Understand that this demo uses random actions (not trained)

---

## Key Takeaways

1. **train_simple.py** is a demo of the environment, not real training
2. **BSH actions** are binary: be in BTC (0) or USD (1)
3. **PBR rewards** encourage correct position, not just trading
4. **Commission** is a major cost - 12 trades at 0.1% = ~1.2% loss
5. **Random policy loses money** - real training is needed

---

## Next Steps

You've completed the foundations. Choose your path:

- **Track A**: [Trading for RL People](../02-domains/track-a-trading-for-rl/01-trading-basics.md) - If you know RL, learn trading
- **Track B**: [RL for Traders](../02-domains/track-b-rl-for-traders/01-rl-fundamentals.md) - If you know trading, learn RL
- **Skip to Training**: [First Training](../04-training/01-first-training.md) - Ready to train a real agent
