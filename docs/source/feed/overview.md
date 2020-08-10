# Overview

The `feed` package provides useful tools when building trading environments. The primary reason for using this package is to help build the mechanisms that generate observations from of an environment. Therefore, it is fitting that their primary location of use is in the `Observer` component of the `TradingEnv`.

## What is a `Stream`?
A `Stream` is the basic building block for data feeds. The `Stream` API provides the granularity needed to connect specific data sources to the `Observer`. For example, below is how you would turn a counter into a stream.


```python
from tensortrade.feed import Stream, DataFeed

s = Stream.source()
```


## Other Uses
