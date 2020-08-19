# Simple Sine Curve

A simple sine curve can be used to perform a sanity check on your trading algorithm. A trading algorithm should be able to learn to make money on the predictable pattern of a sine curve.

To create a DataFeed that follows a sine curve the following code could be used. Remember that a TensorTrade DataFeed can accept any `Iterable`.

```python
import numpy as np
import tensortrade

#create the list of values
x = np.arange(0, 2*np.pi, (2*np.pi)/ 1000)

#create a tensortrade stream from the inputs
data_range = tensortrade.feed.Stream.source(data_values)

#transform values into a sine curve and rename the stream
price = data_range.apply(lambda v: 50 * np.sin(4 * np.pi * v) + 100).rename('USDT-BTC')

#create a datafeed with multiple feature inputs derived from the sine curve
feed = DataFeed([
    price,
    price.rolling(window=10).mean().rename("fast"),
    price.rolling(window=50).mean().rename("medium"),
    price.rolling(window=100).mean().rename("slow"),
    price.log().diff().fillna(0).rename("lr")
])
```
