# Code Structure

The TensorTrade library is modular. The `tensortrade` library usually has a common setup:

1. An abstract `MetaABC` class that highlights the methods that will generally be called inside of the main `TradingEnvironment`.
2. Specific applications of that abstract class are then specified later to make more detailed specifications.

## Example of Structure:

A good example of this structure is the `Exchange` component. It represents all exchange interactions.

The beginning of the code in [Exchange](https://github.com/notadamking/tensortrade/blob/master/tensortrade/exchanges/exchange.py) is seen here.

```py
class Exchange(object, metaclass=ABCMeta):
    """An abstract exchange for use within a trading environment."""

    def __init__(self, base_instrument: str = 'USD', dtype: Union[type, str] = np.float32, feature_pipeline: FeaturePipeline = None):
        """
        Arguments:
            base_instrument: The exchange symbol of the instrument to store/measure value in.
            dtype: A type or str corresponding to the dtype of the `observation_space`.
            feature_pipeline: A pipeline of feature transformations for transforming observations.
        """
        self._base_instrument = base_instrument
        self._dtype = dtype
        self._feature_pipeline = feature_pipeline
```

As you can see above, the [Exchange](https://github.com/notadamking/tensortrade/blob/master/tensortrade/exchanges/exchange.py) has a large majority of the instantiation details that carries over to all other reprentations of that type of class. `ABCMeta` represents that all classes that inherit it shall be recognizable as an instance of `Exchange`. This is nice when you need to do type checking.

When creating a new exchange type (everything that's an inheritance of the `Exchange`), one needs to add further details for how information should be declared by default. Once you create a new type of exchange, you can have new rules placed in by default. Let's look at the SimulatedExchange and it can have parameters dynamically set via the `**kwargs` arguement in later exchanges.

**SimulatedExchange:**

```py
class SimulatedExchange(Exchange):
    """An exchange, in which the price history is based off the supplied data frame and
    trade execution is largely decided by the designated slippage model.
    If the `data_frame` parameter is not supplied upon initialization, it must be set before
    the exchange can be used within a trading environment.
    """

    def __init__(self, data_frame: pd.DataFrame = None, **kwargs):
        super().__init__(
            dtype=self.default('dtype', np.float32),
            feature_pipeline=self.default('feature_pipeline', None)
        )

        self._commission_percent = self.default('commission_percent', 0.3, kwargs)
        self._base_precision = self.default('base_precision', 2, kwargs)
        self._instrument_precision = self.default('instrument_precision', 8, kwargs)
        self._min_trade_size = self.default('min_trade_size', 1e-6, kwargs)
        self._max_trade_size = self.default('max_trade_size', 1e6, kwargs)

        self._initial_balance = self.default('initial_balance', 1e4, kwargs)
        self._observation_columns = self.default(
            'observation_columns',
            ['open', 'high', 'low', 'close', 'volume'],
            kwargs
        )
        self._price_column = self.default('price_column', 'close', kwargs)
        self._window_size = self.default('window_size', 1, kwargs)
        self._pretransform = self.default('pretransform', True, kwargs)
        self._price_history = None

        self.data_frame = self.default('data_frame', data_frame)

        model = self.default('slippage_model', 'uniform', kwargs)
        self._slippage_model = slippage.get(model) if isinstance(model, str) else model()
```

Everything that inherits `SimulatedExchange` uses the specified kwargs to set the parameters.

Therefore, even when we don't directly see the parameters inside of `FBMExchange`, all of the defaults are being called.

**An example:**

```py
exchange = FBMExchange(base_instrument='BTC', timeframe='1h', base_precision=4) # we're replacing the default base precision.
```
