# [TensorTrade: Trade Efficiently with Reinforcement Learning](https://towardsdatascience.com/trade-smarter-w-reinforcement-learning-a5e91163f315?source=friends_link&sk=ea3afd0a305141eb9147be4718826dfb)

[![Build Status](https://travis-ci.com/tensortrade-org/tensortrade.svg?branch=master)](https://travis-ci.org/tensortrade-org/tensortrade)
[![Documentation Status](https://readthedocs.org/projects/tensortrade/badge/?version=latest)](https://tensortrade.org)
[![Apache License](https://img.shields.io/github/license/tensortrade-org/tensortrade.svg?color=brightgreen)](http://www.apache.org/licenses/LICENSE-2.0)
[![Discord](https://img.shields.io/discord/592446624882491402.svg?color=brightgreen)](https://discord.gg/ZZ7BGWh)
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-360/)

---

<div align="center">
  <img src="https://github.com/notadamking/tensortrade/blob/master/docs/source/_static/logo.jpg">
</div>

---

**TensorTrade is still in Beta, meaning it should be used very cautiously if used in production, as it may contain bugs.**

TensorTrade is an open source Python framework for building, training, evaluating, and deploying robust trading algorithms using reinforcement learning. The framework focuses on being highly composable and extensible, to allow the system to scale from simple trading strategies on a single CPU, to complex investment strategies run on a distribution of HPC machines.

Under the hood, the framework uses many of the APIs from existing machine learning libraries to maintain high quality data pipelines and learning models. One of the main goals of TensorTrade is to enable fast experimentation with algorithmic trading strategies, by leveraging the existing tools and pipelines provided by `numpy`, `pandas`, `gym`, `keras`, and `tensorflow`.

Every piece of the framework is split up into re-usable components, allowing you to take advantage of the general use components built by the community, while keeping your proprietary features private. The aim is to simplify the process of testing and deploying robust trading agents using deep reinforcement learning, to allow you and I to focus on creating profitable strategies.

_The goal of this framework is to enable fast experimentation, while maintaining production-quality data pipelines._

Read [the documentation](https://www.tensortrade.org/en/latest/).

## Guiding principles

_Inspired by [Keras' guiding principles](https://github.com/keras-team/keras)._

- **User friendliness.** TensorTrade is an API designed for human beings, not machines. It puts user experience front and center. TensorTrade follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear and actionable feedback upon user error.

- **Modularity.** A trading environment is a conglomeration of fully configurable modules that can be plugged together with as few restrictions as possible. In particular, exchanges, feature pipelines, action schemes, reward schemes, trading agents, and performance reports are all standalone modules that you can combine to create new trading environments.

- **Easy extensibility.** New modules are simple to add (as new classes and functions), and existing modules provide ample examples. To be able to easily create new modules allows for total expressiveness, making TensorTrade suitable for advanced research and production use.

## Getting Started

You can get started testing on Google Colab or your local machine, by viewing our [many examples](https://github.com/tensortrade-org/tensortrade/tree/master/examples)

## Installation

TensorTrade requires Python >= 3.7 for all functionality to work as expected.
You can install TensorTrade both as a pre-packaged solution by running the default setup command.
```bash
pip install tensortrade
```
You can then alternatively install TensorTrade directly from the master code repository, pulling directly from the latest commits. This will give you the latest features\fixes, but it is highly untested code, so proceed at your own risk.
```bash
pip install git+https://github.com/tensortrade-org/tensortrade.git
```
Alternatively you can clone\download the repository in your local environment an manually install the requirements, either the "base" ones, or the ones that also include requirements to run the examples in the documentation.
```bash
pip install -r requirements.txt
pip install -r examples/requirements.txt
```

## Docker

To run the commands below, ensure Docker is installed. Visit https://docs.docker.com/install/ for more information.

### Run Jupyter Notebooks

To run a jupyter notebook in your browser, execute the following command and visit the `http://127.0.0.1:8888/?token=...` link printed to the command line.

```bash
make run-notebook
```

### Build Documentation

To build the HTML documentation, execute the following command.

```bash
make run-docs
```

### Run Test Suite

To run the test suite, execute the following command.

```bash
make run-tests
```

## Support

You can ask questions and join the development discussion:

- On the [TensorTrade Discord server](https://discord.gg/ZZ7BGWh).
- On the [TensorTrade Gitter](https://gitter.im/tensortrade-framework/community).

You can also post **bug reports and feature requests** in [GitHub issues](https://github.com/notadamking/tensortrade/issues). Make sure to read [our guidelines](https://github.com/notadamking/tensortrade/blob/master/CONTRIBUTING.md) first.


## Contributors

Contributions are encouraged and welcomed. This project is meant to grow as the community around it grows. Let me know on Discord in the #suggestions channel if there is anything that you would like to see in the future, or if there is anything you feel is missing.

**Working on your first Pull Request?** You can learn how from this _free_ series [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github)

![https://github.com/notadamking/tensortrade/graphs/contributors](https://contributors-img.firebaseapp.com/image?repo=notadamking/tensortrade)
