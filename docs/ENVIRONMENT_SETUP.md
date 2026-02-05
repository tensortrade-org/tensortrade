# TensorTrade Environment Setup Guide

This guide provides step-by-step instructions for setting up a working TensorTrade environment.

## Prerequisites

- **Python 3.12+** (required - Python 3.11 and below are NOT supported)
- **pip** (latest version recommended)
- **Virtual environment tool** (venv, conda, or virtualenv)

## Quick Start (Recommended)

```bash
# Requires Python 3.12+
python3.12 -m venv tensortrade-env
source tensortrade-env/bin/activate  # Windows: tensortrade-env\Scripts\activate

# Install core library
pip install --upgrade pip
pip install -e .

# Verify (232 unit tests should pass, 2 skipped)
pytest tests/tensortrade/unit -v
```

## Training with Ray/RLlib

For reinforcement learning training, install additional dependencies:

```bash
# Install Ray with RLlib and other training dependencies
pip install -r examples/requirements.txt

# Verify full test suite including RLlib integration (251 passed, 2 skipped)
pytest tests/ -v
```

## Alternative Installation Methods

### Option 1: Using venv

```bash
# Create virtual environment
python3.12 -m venv tensortrade-env

# Activate virtual environment
# On Windows:
tensortrade-env\Scripts\activate
# On Linux/macOS:
source tensortrade-env/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install TensorTrade
pip install -r requirements.txt
pip install -e .

# Install example dependencies (optional)
pip install -r examples/requirements.txt
```

### Option 2: Using Conda

```bash
# Create conda environment (use 3.11 or 3.12)
conda create -n tensortrade python=3.12

# Activate environment
conda activate tensortrade

# Install TensorTrade
pip install -r requirements.txt
pip install -e .

# Install example dependencies (optional)
pip install -r examples/requirements.txt
```

### Option 3: Google Colab

```python
# Run in Colab cell
!pip install --upgrade pip
!git clone https://github.com/tensortrade-org/tensortrade.git
%cd tensortrade
!pip install -r requirements.txt
!pip install -r examples/requirements.txt
!pip install -e .
```

## Detailed Installation Steps

### Step 1: Verify Python Version

```bash
python --version
# Should output: Python 3.11.x or 3.12.x
# Note: Python 3.14 is NOT supported (TensorFlow incompatibility)
```

If you have an older version, download Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/).

### Step 2: Create Virtual Environment

**Why use a virtual environment?**
- Isolates dependencies
- Prevents version conflicts
- Makes it easy to reproduce the environment

```bash
# Using venv (use python3.11 or python3.12)
python3.12 -m venv tensortrade-env

# Using conda
conda create -n tensortrade python=3.12
```

### Step 3: Activate Virtual Environment

```bash
# Windows (venv)
tensortrade-env\Scripts\activate

# Linux/macOS (venv)
source tensortrade-env/bin/activate

# Conda (all platforms)
conda activate tensortrade
```

### Step 4: Install Core Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install TensorTrade core dependencies
pip install -r requirements.txt
```

**Core dependencies installed:**
- numpy>=1.26.4,<2.0
- pandas>=2.2.3,<3.0
- gymnasium>=0.28.1,<1.0
- tensorflow>=2.15.1
- ta>=0.4.7
- And more...

### Step 5: Install TensorTrade

```bash
# Install in editable mode (for development)
pip install -e .

# Or install normally
pip install .
```

### Step 6: Install Example Dependencies (Optional)

```bash
pip install -r examples/requirements.txt
```

**Example dependencies include:**
- ray[default,tune,rllib,serve]==2.37.0
- ccxt>=1.72.37
- jupyterlab>=1.1.4
- scikit-learn
- optuna
- feature_engine

### Step 7: Verify Installation

```python
# Test import
python -c "import tensortrade; print(tensortrade.__version__)"

# Should output: 1.0.4.dev1
```

## Platform-Specific Instructions

### Windows 10/11

**Additional Requirements:**
- Visual Studio Build Tools (for some packages)
- Download from: https://visualstudio.microsoft.com/downloads/

**Common Issues:**
1. **Long path names**: Enable long path support in Windows
   ```
   Registry: HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
   Set LongPathsEnabled to 1
   ```

2. **Permission errors**: Run terminal as Administrator

### Linux (Ubuntu 20.04+)

**Additional Requirements:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential
```

**For CUDA support:**
```bash
# Install CUDA toolkit (for GPU support)
# Follow: https://developer.nvidia.com/cuda-downloads

# Install TensorFlow with CUDA
pip install tensorflow[and-cuda]==2.15.1
```

### macOS (12.0+)

**Additional Requirements:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Troubleshooting

### Issue: TensorFlow Not Found (Python 3.14)

**Problem:** `No matching distribution found for tensorflow>=2.15.1`

**Solution:** TensorFlow does not yet support Python 3.14. Use Python 3.11 or 3.12 instead:
```bash
# Check your Python version
python --version

# If using 3.14, switch to 3.12
python3.12 -m venv tensortrade-env
source tensortrade-env/bin/activate
```

### Issue: Ray Installation Fails

**Solution:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install Ray separately
pip install ray[default,tune,rllib,serve]==2.37.0

# If still fails, try without extras
pip install ray==2.37.0
pip install ray[tune]
pip install ray[rllib]
```

### Issue: TensorFlow CUDA Not Working

**Solution:**
```bash
# Uninstall existing TensorFlow
pip uninstall tensorflow

# Install TensorFlow with CUDA support
pip install tensorflow[and-cuda]==2.15.1

# Verify CUDA is available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Issue: NumPy Version Conflict

**Solution:**
```bash
# Ensure NumPy < 2.0 for TensorFlow compatibility
pip install "numpy>=1.26.4,<2.0" --force-reinstall
```

### Issue: Pandas API Errors (pct_change, ewm)

**Problem:** Tests fail with pandas deprecation warnings or API errors.

**Solution:** TensorTrade requires pandas < 3.0:
```bash
pip install "pandas>=2.2.3,<3.0" --force-reinstall
```

### Issue: importlib-metadata Error

**Solution:**
```bash
# Install compatible version
pip install "importlib-metadata>=4.13.0,<6.0"
```

### Issue: "No stream satisfies selector condition"

**Solution:**
This has been fixed in the latest version. Make sure you're using the updated code with the stream selector fix in `tensortrade/env/default/observers.py`.

### Issue: Examples Don't Run

**Solution:**
```bash
# Make sure example dependencies are installed
pip install -r examples/requirements.txt

# Verify Ray version
python -c "import ray; print(ray.__version__)"
# Should output: 2.37.0
```

## Testing Your Installation

### Basic Test

```python
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
import tensortrade.env.default as default

# Create simple environment
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

# Test environment
obs, info = env.reset()
print(f"Environment created successfully! Observation shape: {obs.shape}")
```

### Run Tests

```bash
# Run unit tests
pytest tests/tensortrade/unit -v

# Run integration tests
pytest tests/tensortrade/integration -v

# Run all tests
pytest tests/ -v
```

## Getting Help

If you encounter issues not covered in this guide:

1. Search [GitHub Issues](https://github.com/tensortrade-org/tensortrade/issues)
3. Ask on [Discord](https://discord.gg/ZZ7BGWh)
4. Open a new issue with:
   - Your Python version
   - Your OS
   - Complete error message
   - Steps to reproduce

## Next Steps

- Read the [Quick Start Guide](../README.md#quick-start)
- Try the [Setup Environment Tutorial](../examples/setup_environment_tutorial.ipynb)
- Explore [Example Notebooks](../examples/)
- Read the [API Documentation](https://www.tensortrade.org/)

## Maintenance

### Updating TensorTrade

```bash
# Pull latest changes
git pull origin master

# Reinstall
pip install -e . --upgrade

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Cleaning Up

```bash
# Deactivate virtual environment
deactivate  # or: conda deactivate

# Remove virtual environment
rm -rf tensortrade-env  # or: conda env remove -n tensortrade
```

## Best Practices

1. **Always use a virtual environment**
2. **Keep dependencies up to date**
3. **Test after updates**
4. **Document your environment** (use `pip freeze > my-requirements.txt`)
5. **Use version control** for your code

