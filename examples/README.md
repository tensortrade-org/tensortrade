# TensorTrade Examples

This directory contains Jupyter notebook examples demonstrating various features and use cases of TensorTrade.

## Prerequisites

Before running the examples, make sure you have:

1. **Python 3.11.9+** installed
2. **TensorTrade** installed (see [Installation Guide](../docs/ENVIRONMENT_SETUP.md))
3. **Example dependencies** installed:
   ```bash
   pip install -r requirements.txt
   ```

## Examples Overview

### 1. setup_environment_tutorial.ipynb
**Purpose**: Learn how to set up a basic trading environment

**What you'll learn:**
- How to fetch historical data
- How to create exchanges and portfolios
- How to set up data feeds
- How to create and configure a trading environment
- How to run basic trading simulations

**Prerequisites:**
- Basic Python knowledge
- Understanding of trading concepts

**Runtime:** ~5 minutes

### 2. train_and_evaluate.ipynb
**Purpose**: Train and evaluate trading agents using reinforcement learning

**What you'll learn:**
- Feature engineering for trading data
- Using technical indicators (pandas_ta)
- Training RL agents
- Evaluating agent performance
- Using quantstats for performance metrics

**Prerequisites:**
- Completed setup_environment_tutorial.ipynb
- Basic understanding of machine learning

**Runtime:** ~30-60 minutes (depending on training iterations)

**Note:** This example has been updated to fix pandas_ta and quantstats compatibility issues.

### 3. use_lstm_rllib.ipynb
**Purpose**: Train LSTM-based trading agents using Ray RLlib

**What you'll learn:**
- Using Ray Tune for hyperparameter optimization
- Training LSTM models for trading
- Distributed training with Ray
- Checkpoint management
- Model evaluation

**Prerequisites:**
- Completed train_and_evaluate.ipynb
- Understanding of LSTM networks
- Ray 2.37.0 installed

**Runtime:** ~2-4 hours (for full hyperparameter search)

**Note:** This example has been updated to use Ray 2.x API (tune.Tuner instead of tune.run).

### 4. use_attentionnet_rllib.ipynb
**Purpose**: Train attention-based trading agents using Ray RLlib

**What you'll learn:**
- Using attention mechanisms in trading
- Advanced neural network architectures
- Comparing attention vs LSTM performance

**Prerequisites:**
- Completed use_lstm_rllib.ipynb
- Understanding of attention mechanisms

**Runtime:** ~2-4 hours

**Note:** This example has been updated to use Ray 2.x API.

### 5. ledger_example.ipynb
**Purpose**: Understand the order management system and ledger

**What you'll learn:**
- How orders are executed
- Portfolio balance tracking
- Transaction history
- Ledger functionality

**Prerequisites:**
- Completed setup_environment_tutorial.ipynb

**Runtime:** ~10 minutes

### 6. renderers_and_plotly_chart.ipynb
**Purpose**: Visualize trading performance with interactive charts

**What you'll learn:**
- Using Plotly for visualization
- Creating custom renderers
- Real-time chart updates
- Performance visualization

**Prerequisites:**
- Completed setup_environment_tutorial.ipynb
- Plotly installed

**Runtime:** ~15 minutes

### 7. use_stochastic_data.ipynb
**Purpose**: Generate and use stochastic price data for testing

**What you'll learn:**
- Generating synthetic price data
- Using stochastic processes (GBM, Ornstein-Uhlenbeck, etc.)
- Testing strategies on synthetic data
- Backtesting without historical data

**Prerequisites:**
- Basic understanding of stochastic processes

**Runtime:** ~10 minutes

## Running the Examples

### Option 1: Jupyter Notebook

```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to examples directory and open any notebook
```

### Option 2: JupyterLab

```bash
# Start JupyterLab
jupyter lab

# Navigate to examples directory and open any notebook
```

### Option 3: Google Colab

1. Upload the notebook to Google Colab
2. Install dependencies in the first cell:
   ```python
   !pip install tensortrade
   !pip install -r requirements.txt
   ```
3. Run the notebook

## Common Issues

### Issue: Ray version mismatch

**Error:** `TypeError: register() missing 1 required positional argument: 'entry_point'`

**Solution:**
```bash
pip install ray[default,tune,rllib,serve]==2.37.0
```

### Issue: pandas_ta study() not found

**Error:** `AttributeError: 'AnalysisIndicators' object has no attribute 'study'`

**Solution:** Use the fixed `generate_features_fixed()` function provided in the updated notebook.

### Issue: quantstats treynor_ratio error

**Error:** `treynor_ratio() missing 1 required positional argument: 'benchmark'`

**Solution:** Use the fixed `generate_all_default_quantstats_features_fixed()` function provided in the updated notebook.

### Issue: Stream selector error

**Error:** `Exception: No stream satisfies selector condition`

**Solution:** This has been fixed in the latest version. Make sure you're using TensorTrade v1.0.4-dev1 or later.

## Tips for Success

1. **Start with setup_environment_tutorial.ipynb** - It provides the foundation for all other examples
2. **Run cells sequentially** - Don't skip cells or run them out of order
3. **Read the comments** - Each notebook has detailed explanations
4. **Experiment** - Modify parameters and see how they affect results
5. **Save your work** - Make copies of notebooks before modifying them
6. **Check dependencies** - Make sure all required packages are installed

## Data Files

The `data/` directory contains sample datasets:

- `Coinbase_BTCUSD_1h.csv` - Bitcoin hourly data from Coinbase
- `Coinbase_BTCUSD_d.csv` - Bitcoin daily data from Coinbase
- `configuration.json` - Sample JSON configuration
- `configuration.yaml` - Sample YAML configuration

## Performance Notes

- **Training time varies** based on:
  - Number of training iterations
  - Hyperparameter search space
  - Hardware (CPU vs GPU)
  - Number of parallel workers

- **Memory usage** can be high for:
  - Large datasets
  - Complex neural networks
  - Multiple parallel workers

- **Recommendations:**
  - Start with small datasets
  - Use fewer training iterations initially
  - Reduce parallel workers if memory is limited
  - Use GPU if available

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](../docs/ENVIRONMENT_SETUP.md#troubleshooting)
2. Review the [Compatibility Matrix](../COMPATIBILITY.md)
3. Search [GitHub Issues](https://github.com/tensortrade-org/tensortrade/issues)
4. Ask on [Discord](https://discord.gg/ZZ7BGWh)

## Contributing

Found a bug or have an improvement? Please:

1. Open an issue on GitHub
2. Submit a pull request with fixes
3. Share your custom examples

## Next Steps

After completing the examples:

1. Read the [API Documentation](https://www.tensortrade.org/)
2. Explore custom components
3. Build your own trading strategies
4. Deploy to production (with caution - still in Beta)

## License

All examples are licensed under Apache 2.0. See [LICENSE](../LICENSE) for details.

