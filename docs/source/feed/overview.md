# Overview

The `feed` package provides useful tools when building trading environments. The primary reason for using this package is to help build the mechanisms that generate observations from an environment. Therefore, it is fitting that their primary location of use is in the `Observer` component. The `Stream` API provides the granularity needed to connect specific data sources to the `Observer`.

# What is a `Stream`?
A `Stream` is the basic building block for the `DataFeed`, which is also a stream itself. Each stream has a name and a data type and they can be set after the stream is created.  Streams can be created through the following mechanisms:
* generators
* iterables
* sensors
* direct implementation of `Stream`

For example, if you wanted to make a stream for a simple counter. We will make it such that it will start at 0 and increment by 1 each time it is called and on `reset` will set the count back to 0. The following code accomplishes this functionality through creating a generator function.

```python
from tensortrade.feed import Stream

def counter():
    i = 0
    while True:
        yield i
        i += 1

s = Stream.source(counter)
```

In addition, you can also use the built-in `count` generator from the `itertools` package.

```python
from itertools import count

s = Stream.source(count(start=0, step=1))
```

These will all create infinite streams that will keep incrementing by 1 to infinity. If you wanted to make something that counted until some finite number you can use the built in `range` function.

```python
s = Stream.source(range(5))
```

This can also be done by giving in a `list` directly.

```python
s = Stream.source([1, 2, 3, 4, 5])
```

The direct approach to stream creation is by subclassing `Stream` and implementing the `forward`, `has_next`, and `reset` methods. If the stream does not hold stateful information, then `reset` is not required to be implemented and can be ignored.

```python
class Counter(Stream):

    def __init__(self):
        super().__init__()
        self.count = None

    def forward(self):
        if self.count is None:
            self.count = 0
        else:
            self.count += 1
        return self.count

    def has_next(self):
        return True

    def reset(self):
        self.count = None

s = Counter()
```

There is also a way of creating streams which serves the purpose of watching a particular object and how it changes over time. This can be done through the `sensor` function. For example, we can use this to directly track performance statistics on our `portfolio`. Here is a specific example of how we can use it to track the number of orders the are currently active inside the order management system.

```python
from tensortrade.env.default.actions import SimpleOrders

action_scheme = SimpleOrders()

s = Stream.sensor(action_scheme.broker, lambda b: len(b.unexecuted))
```

As the agent and the environment are interacting with one another, this stream will be able to monitor the number of active orders being handled by the broker. This stream can then be used by either computing performance statistics and supplying them to a `Renderer` or simply by including it within the observation space.

Now that we have seen the different ways we can create streams, we need to understand the ways in which we can aggregate new streams from old. This is where the data type of a stream becomes important.

# Using Data Types
The purpose of the data type of a stream, `dtype`, is to add additional functionality and behavior to a stream such that it can be aggregated with other streams of the same type in an easy and intuitive way. For example, what if the number of executed orders from the `broker` is not important by itself, but is important with respect to the current time of the process. This can be taken into account if we create a stream for keeping count of the active orders and another one for keeping track of the step in the process. Here is what that would look like.

```python
from itertools import count

from tensortrade.feed import Stream
from tensortrade.env.default.actions import SimpleOrders

n = Stream.source(count(0, step=1), dtype="float")
n_active = Stream.sensor(action_scheme.broker, lambda b: len(b.unexecuted), dtype="float")

s = (n_active / (n + 1)).rename("avg_n_active")
```

Suppose we find that this is not a useful statistic and instead would like to know how many of the active order have been filled since the last time step. This can be done by using the `lag` operator on our stream and finding the difference between the current count and the count from the last time step.

```python
n_active = Stream.sensor(action_scheme.broker, lambda b: len(b.unexecuted), dtype="float")

s = (n_active - n_active.lag()).rename("n_filled")
```

As you can see from the code above, we were able to make more complex streams by using simple ones. Take note, however, in the way we use the `rename` function. We only really want to rename a stream if we will be using it somewhere else where its name will be useful (e.g. in our `feed`). We do not want to name all the intermediate streams that are used to build our final statistic because the code will become too cumbersome and annoying. To avoid these complications, streams are created to automatically generate a unique name on instantiation. We leave the naming for the user to decide which streams are useful to name.

Since the most common data type is `float` in these tasks, the following is a list of supported special operations for it:
* Let `s`, `s1`, `s2` be streams.
* Let `c` be a constant.
* Let `n` be a number.
* Unary:
    * `-s`, `s.neg()`
    * `abs(s)`, `s.abs()`
    * `s**2`, `pow(s, n)`
* Binary:
    * `s1 + s2`, `s1.add(s2)`, `s + c`, `c + s`
    * `s1 - s2`, `s1.sub(s2)`, `s - c`, `c - s`
    * `s1 * s2`, `s1.mul(s2)`, `s * c`, `c * s`
    * `s1 / s2`, `s1.div(s2)`, `s / c`, `c / s`


There are many more useful functions that can be utilized, too many to list in fact. You can find all of the. however, in the API reference section of the documentation.

# Advanced Usages
The `Stream` API is very robust and can handle complex streaming operations, particularly for the `float` data type. Some of the more advanced usages include performance tracking and developing reward schemes for the `default` trading environment. In the following example, we will show how to track the net worth of a portfolio. This implementation will be coming directly from the wallets that are defined in the `portfolio`.

```python
# Suppose we have an already constructed portfolio object, `portfolio`.

worth_streams = []

for wallet in portfolio.wallets:

    total_balance = Stream.sensor(
        wallet,
        lambda w: w.total_balance.as_float(),
        dtype="float"
    )

    symbol = w.instrument.symbol

    if symbol == portfolio.base_instrument.symbol
        worth_streams += [total_balance]
    else:
        price = Stream.select(
            w.exchange.streams(),
            lambda s: s.name.endswith(symbol)
        )
        worth_streams += [(price * total_balance)]

net_worth = Stream.reduce(worth_streams).sum().rename("net_worth")
```
