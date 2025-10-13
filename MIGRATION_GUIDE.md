# TensorTrade Migration Guide

This guide helps you migrate from TensorTrade 1.0.3 to 1.0.4-dev1.

## Overview

Version 1.0.4-dev1 includes several breaking changes, primarily due to:
- Ray 1.9.2 → Ray 2.37.0 upgrade
- Python 3.7+ → Python 3.11.9+ requirement
- Dependency updates for compatibility and security

## Quick Migration Checklist

- [ ] Upgrade Python to 3.11.9 or higher
- [ ] Update all dependencies
- [ ] Update Ray Tune API calls
- [ ] Test stream selector functionality
- [ ] Verify GPU compatibility (if using GPU)
- [ ] Update example notebooks
- [ ] Run test suite

## Step-by-Step Migration

### Step 1: Upgrade Python

**Current Requirement:** Python >= 3.11.9

**Check your version:**
```bash
python --version
```

**If you need to upgrade:**
1. Download Python 3.11+ from [python.org](https://www.python.org/downloads/)
2. Install Python
3. Create new virtual environment:
   ```bash
   python -m venv tensortrade-env
   source tensortrade-env/bin/activate  # On Windows: tensortrade-env\Scripts\activate
   ```

### Step 2: Update Dependencies

**Update requirements.txt:**
```bash
cd tensortrade
pip install -r requirements.txt --upgrade
pip install -r examples/requirements.txt --upgrade
pip install -e . --upgrade
```

**Key dependency changes:**
- Ray: 1.9.2 → 2.37.0
- TensorFlow: >=2.7.0 → >=2.15.1
- NumPy: >=1.17.0 → >=1.26.4,<2.0
- Pandas: >=0.25.0 → >=2.2.3

### Step 3: Update Ray Tune API

**Old API (Ray 1.x):**
```python
from ray import tune

analysis = tune.run(
    "PPO",
    stop={"training_iteration": 100},
    config={
        "env": "TradingEnv",
        "lr": tune.uniform(0.001, 0.01),
        # ... other config
    },
    num_samples=10,
    checkpoint_freq=10
)

best_config = analysis.get_best_config(metric="episode_reward_mean", mode="max")
```

**New API (Ray 2.x):**
```python
from ray import tune

tuner = tune.Tuner(
    "PPO",
    param_space={
        "env": "TradingEnv",
        "lr": tune.uniform(0.001, 0.01),
        # ... other config
    },
    tune_config=tune.TuneConfig(
        num_samples=10,
        metric="episode_reward_mean",
        mode="max",
    ),
    run_config=tune.RunConfig(
        stop={"training_iteration": 100},
        checkpoint_config=tune.CheckpointConfig(
            checkpoint_frequency=10,
            num_to_keep=3,
        ),
    ),
)

results = tuner.fit()
best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
best_config = best_result.config
```

**Key Changes:**
1. `tune.run()` → `tune.Tuner()`
2. `config` → `param_space`
3. Separate `tune_config` and `run_config`
4. `checkpoint_freq` → `checkpoint_config.checkpoint_frequency`
5. `analysis.get_best_config()` → `results.get_best_result().config`

### Step 4: Update Stream Selector Usage

**No code changes required!** The stream selector has been fixed to handle multiple naming conventions automatically.

**What was fixed:**
- Now handles `:symbol`, `-symbol`, and plain `symbol` naming
- No more "No stream satisfies selector condition" errors

**If you had workarounds:**
You can remove any custom stream naming workarounds you implemented.

### Step 5: Update GPU Usage

**New device parameter:**
```python
env = default.create(
    portfolio=portfolio,
    action_scheme="simple",
    reward_scheme="simple",
    feed=feed,
    window_size=10,
    device="cpu"  # or "cuda" for GPU
)
```

**What changed:**
- Observations are now guaranteed to be numpy arrays
- No more tensor/device mismatch errors
- Explicit device control

### Step 6: Update Technical Analysis Code

**Old code (broken):**
```python
df.ta.study(strategy, exclude=['kvo'])
```

**New code (fixed):**
```python
# Use the fixed function provided in examples
data = generate_features_fixed(data)
```

**For quantstats:**
```python
# Use the fixed function that skips treynor_ratio
data = generate_all_default_quantstats_features_fixed(data)
```

### Step 7: Update Example Notebooks

**If you copied example notebooks:**
1. Update to latest versions from repository
2. Or apply fixes manually (see examples/README.md)

**Key updates:**
- `use_lstm_rllib.ipynb`: Ray 2.x API
- `use_attentionnet_rllib.ipynb`: Ray 2.x API
- `train_and_evaluate.ipynb`: Fixed TA functions

### Step 8: Test Your Code

**Run tests:**
```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/tensortrade/unit/env/default/test_stream_selector.py -v
pytest tests/tensortrade/unit/oms/exchanges/test_exchange_streams.py -v
pytest tests/tensortrade/unit/env/test_gpu_compatibility.py -v
pytest tests/tensortrade/integration/test_end_to_end.py -v
```

**Manual testing:**
1. Create simple environment
2. Run reset() and step()
3. Verify no errors
4. Check observation shapes

## Common Migration Issues

### Issue 1: Python Version Error

**Error:** `TensorTrade requires Python >= 3.11.9`

**Solution:**
1. Upgrade Python
2. Create new virtual environment
3. Reinstall dependencies

### Issue 2: Ray Import Error

**Error:** `ModuleNotFoundError: No module named 'ray'`

**Solution:**
```bash
pip install ray[default,tune,rllib,serve]==2.37.0
```

### Issue 3: NumPy Version Conflict

**Error:** `numpy 2.0 is not compatible with tensorflow`

**Solution:**
```bash
pip install "numpy>=1.26.4,<2.0" --force-reinstall
```

### Issue 4: Ray Tune API Error

**Error:** `TypeError: run() got an unexpected keyword argument 'config'`

**Solution:**
Update to Ray 2.x API (see Step 3 above)

### Issue 5: Stream Selector Error

**Error:** `Exception: No stream satisfies selector condition`

**Solution:**
This should be fixed automatically. If you still see it:
1. Verify you're using latest code
2. Check stream naming conventions
3. See [Environment Setup Guide](docs/ENVIRONMENT_SETUP.md#troubleshooting)

## Breaking Changes Summary

### API Changes

| Old API | New API | Notes |
|---------|---------|-------|
| `tune.run()` | `tune.Tuner()` | Ray 2.x migration |
| `config={}` | `param_space={}` | Parameter naming |
| `checkpoint_freq=N` | `checkpoint_config.checkpoint_frequency=N` | Checkpoint config |
| `analysis.get_best_config()` | `results.get_best_result().config` | Results API |

### Dependency Changes

| Package | Old Version | New Version |
|---------|-------------|-------------|
| Python | >=3.7 | >=3.11.9 |
| Ray | 1.9.2 | 2.37.0 |
| TensorFlow | >=2.7.0 | >=2.15.1 |
| NumPy | >=1.17.0 | >=1.26.4,<2.0 |
| Pandas | >=0.25.0 | >=2.2.3 |

### Configuration Changes

| Old Config | New Config | Notes |
|------------|------------|-------|
| N/A | `device="cpu"` | New GPU control |
| N/A | `_ensure_numpy()` | New method |

## Backward Compatibility

### What's NOT Backward Compatible

- Ray 1.x API code will not work
- Python < 3.11.9 will not work
- Old dependency versions may cause conflicts

### What IS Backward Compatible

- Core TensorTrade API (environments, portfolios, etc.)
- Data feed system
- OMS (Order Management System)
- Agent framework
- Most custom components

## Testing Your Migration

### Minimal Test

```python
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
import tensortrade.env.default as default

# Create environment
exchange = Exchange("simulated", service=execute_order)(
    Stream.source([100, 101, 102], dtype="float").rename("USD-BTC")
)

portfolio = Portfolio(USD, [
    Wallet(exchange, 10000 * USD),
    Wallet(exchange, 0 * BTC)
])

feed = DataFeed([
    Stream.source([100, 101, 102], dtype="float").rename("price")
])

env = default.create(
    portfolio=portfolio,
    action_scheme="simple",
    reward_scheme="simple",
    feed=feed,
    window_size=1
)

# Test
obs, info = env.reset()
print(f"✓ Environment created successfully!")
print(f"✓ Observation shape: {obs.shape}")

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"✓ Environment step successful!")
print(f"✓ Reward: {reward}")
```

### Full Test

Run the complete test suite:
```bash
pytest tests/ -v
```

## Getting Help

If you encounter migration issues:

1. Check this migration guide
2. Review [CHANGES.md](CHANGES.md)
3. See [Environment Setup Guide](docs/ENVIRONMENT_SETUP.md)
4. Check [Compatibility Matrix](COMPATIBILITY.md)
5. Search [GitHub Issues](https://github.com/tensortrade-org/tensortrade/issues)
6. Ask on [Discord](https://discord.gg/ZZ7BGWh)
7. Open a new issue with:
   - Your current version
   - Error message
   - Steps to reproduce
   - Environment details

## Rollback Plan

If migration fails and you need to rollback:

```bash
# Uninstall current version
pip uninstall tensortrade

# Install old version
pip install tensortrade==1.0.3

# Downgrade dependencies
pip install ray==1.9.2
pip install tensorflow==2.7.0
pip install "numpy>=1.17.0"
pip install "pandas>=0.25.0"
```

**Note:** Consider using a separate virtual environment for testing migration before committing.

## Timeline

- **1.0.3**: Previous stable release
- **1.0.4-dev1**: Current development release (this version)
- **1.0.4**: Planned stable release (after testing)

## Feedback

Found issues during migration? Please:
1. Open an issue on GitHub
2. Include migration details
3. Suggest improvements to this guide

## Next Steps

After successful migration:
1. Test your custom code thoroughly
2. Update your documentation
3. Train new models with updated environment
4. Monitor for any issues
5. Provide feedback to the community

