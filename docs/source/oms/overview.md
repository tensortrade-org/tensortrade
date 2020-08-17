# Overview
An `order management system (OMS)` is a system that controls how an order for a
specific financial instrument is filled. In building an OMS, you have to make clear
what the lifecycle of an order is. The OMS we use in the default environment for
TensorTrade is a first attempt at building such a system. The goal of our system,
however, was meant to serve the purpose simulating a real order management system.
We created it with the user in mind, hoping to give maximum customization the types
of order placed.

# Portfolio
A [portfolio](https://www.investopedia.com/terms/p/portfolio.asp) is a collection of financial investments like stocks, bonds, commodities, cash, and cash equivalents, including closed-end funds and exchange-traded funds (ETFs). Now a portfolio can include more than just these types of assets. For example, real-estate is an investment that can be considered to be part of a portfolio. For the purposes of algorithmic trading, however, a more formal definition of a portfolio is needed. In order for that to be done we first need to define what a financial instrument is and how it fits in with the idea of a portfolio.

<br> **Instruments** <br>
The core idea surrounding a financial instrument is the idea of tradability. In our universe, an object is said to be tradable if and only if can be traded on an exchange. This definition makes the idea more concrete and solves some ambiguity that can arise when multiple exchanges are involved. For example, Bitcoin (BTC) is an asset that you can hold an amount of, however, its value depends on the exchange that you are trading it on. Therefore, given an `Exchange` we need to be able hold a `Quantity` of an `Instrument`. Now enters the idea for a `Wallet`.

A `Quantity` of an instrument can be created with the following code
```python
from tensortrade.oms.instruments import Quantity, USD, BTC

q = 10000 * USD

q = Quantity(BTC, 10000)
```

<br> **Wallets** <br>
A `Wallet` is specified by an `Exchange` and a `Quantity`. The financial instrument that the wallet holds is implicitly defined in the `Quantity` we are holding in the wallet. The wallet also gives us the ability to transfer funds from a given wallet to another.

```python
# Suppose we have already defined an exchange that supports the financial instrument
# we are creating.
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet

w1 = Wallet(exchange, 10000 * USD)
w2 = Wallet(exchange, 0 * BTC)
```

<br> **Creating a Portfolio** <br>
A `Portfolio` in the library is defined to be a set of wallets. This makes building a portfolio rather simple. All we need to do is create the wallets and pass them in to the `Portfolio` for construction.

```python
# Suppose we have already defined two exchange `e1` and `e2`.

from tensortrade.oms.instruments import USD, BTC, ETH, LTC
from tensortrade.oms.wallets import Wallet, Portfolio

portfolio = Portfolio(
    base_instrument=USD,
    wallets=[
        Wallet(e1, 10000 * USD),
        Wallet(e1, 0 * BTC),
        Wallet(e1, 0 * ETH),
        Wallet(e2, 10 * USD),
        Wallet(e2, 0 * ETH),
        Wallet(e2, 0 * LTC),
    ]
)

```
In addition, you also have to specify what the base instrument is so the portfolio knows what every instruments value should be in.

# Orders
An `Order` is the way in which you can move funds from one wallet to another. The supported orders that can be made right now are the following:
* Market
* Limit
* Stop Loss
* Take Profit

Currently all the `default` action schemes use these orders when interpreting agent actions. The stop loss and take profit orders are the most complicated of which and require the use of an `OrderSpec` for them to function properly. An `OrderSpec` is required when an order must be connected with and followed by a successive order. In the case of a stop order, the process is to buy the quantity requested at the current price and then wait until the price hits a particular mark and then sell it. In addition, an `Order` has an optional `criteria` parameter that needs to be satisfied before being able to execute on an exchange.
