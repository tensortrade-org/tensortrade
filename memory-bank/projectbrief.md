# TensorTrade Project Brief

## Project Overview
TensorTrade is an open-source Python framework for building, training, evaluating, and deploying robust trading algorithms using reinforcement learning. The framework is designed to be highly composable and extensible, scaling from simple trading strategies on a single CPU to complex investment strategies on distributed HPC machines.

## Core Purpose
- **Primary Goal**: Enable fast experimentation with algorithmic trading strategies using reinforcement learning
- **Target Users**: Researchers, quantitative analysts, and developers working on algorithmic trading
- **Key Value**: Simplifies the process of testing and deploying robust trading agents using deep reinforcement learning

## Key Features
1. **Modular Architecture**: Every component is reusable and can be mixed and matched
2. **Gymnasium Integration**: Compatible with OpenAI Gym/Gymnasium for RL algorithms
3. **Comprehensive OMS**: Full Order Management System with exchanges, wallets, orders, and portfolios
4. **Flexible Data Pipeline**: Support for various data sources and real-time feeds
5. **Built-in Components**: Pre-built action schemes, reward schemes, observers, and renderers
6. **Stochastic Processes**: Built-in support for various financial stochastic processes

## Technical Stack
- **Core**: Python 3.11+, NumPy, Pandas
- **RL Framework**: Gymnasium (OpenAI Gym successor)
- **ML**: TensorFlow 2.7+
- **Data**: Stochastic processes, technical analysis
- **Visualization**: Matplotlib, Plotly
- **Documentation**: Sphinx

## Current Status
- **Version**: 1.0.4-dev1 (Beta)
- **License**: Apache 2.0
- **Development Status**: Production/Stable (but marked as Beta)
- **Repository**: https://github.com/tensortrade-org/tensortrade

## Architecture Philosophy
The framework follows three guiding principles inspired by Keras:
1. **User Friendliness**: Consistent & simple APIs, minimal cognitive load
2. **Modularity**: Fully configurable modules that can be plugged together
3. **Easy Extensibility**: Simple to add new modules, total expressiveness for research and production

## Target Use Cases
- Research in algorithmic trading and reinforcement learning
- Backtesting trading strategies
- Live trading system development
- Educational purposes for learning RL in finance
- Production trading system deployment (with caution due to Beta status)
