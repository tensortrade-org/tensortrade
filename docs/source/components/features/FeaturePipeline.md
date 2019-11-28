# FeaturePipeline

Feature pipelines are meant for transforming observations from the environment into meaningful features for an agent to learn from. If a pipeline has been added to a particular exchange, then observations will be passed through the `FeaturePipeline` before being output to the environment.

For example, a feature pipeline could normalize all price values, make a time series stationary, add a moving average column, and remove an unnecessary column, all before the observation is returned to the agent.

Feature pipelines can be initialized with an arbitrary number of comma-separated transformers. Each `FeatureTransformer` needs to be initialized with the set of columns to transform, or if nothing is passed, all input columns will be transformed.

Each feature transformer has a transform method, which will transform a single observation (a `pandas.DataFrame`) from a larger data set, keeping any necessary state in memory to transform the next frame. For this reason, it is often necessary to reset the FeatureTransformer periodically. This is done automatically each time the parent `FeaturePipeline` or `Exchange` is reset.

## How It Operates

The `FeaturePipeline` has a setup that resembles the `keras` library. The concept is simple:

1. We take in an observation of data (price information), usually in the form of a pandas dataframe.
2. We take the observation and effectively run it through all declared ways of transforming that data inside of the FeaturePipeline and turn the result as a `gym.space`.

Just like kera's `Sequential` module, it accepts a list inside of its constructor and iterates through each piece on call. To draw on parallels, look at `keras`:

```py
model = Sequential([
    Dense(32, input_shape=(500,)),
    Dense(32)
])
```

## Class Parameters

- `steps`
  - A list of feature transformations to apply to observations.
- `dtype`
  - The `dtype` elements in the pipeline should be cast to.

## Properties and Setters

- `steps`
  - A list of feature transformations to apply to observations.
- `dtype`
  - The `dtype` that elements in the pipeline should be input and output as.
- `reset`
  - Reset all transformers within the feature pipeline.

## Functions

Below are the functions that the `FeaturePipeline` uses to effectively operate.

### Private

- `_transform`
  - Utility method for transforming observations via a list of _make changes here_ `FeatureTransformer` objects.
  - In other words, it runs through all of the `steps` in a for loop, and casts the response.

**The code from the transform function:**
As you see, it iterates through every step and adds the observation to the dataframe.

```py
for transformer in self._steps:
    observations = transformer.transform(observations)
```

At the end the observations are converted into a ndarray so that they can be interpreted by the agent.

### Public

- `reset`
  - Reset all transformers within the feature pipeline.
- `transform`
  - Apply the pipeline of feature transformations to an observation frame.

## Use Cases

**Use Case #1: Initiate Pipeline**

```py
from tensortrade.features import FeaturePipeline
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.indicators import SimpleMovingAverage

price_columns = ["open", "high", "low", "close"]
normalize_price = MinMaxNormalizer(price_columns)
moving_averages = SimpleMovingAverage(price_columns)
difference_all = FractionalDifference(difference_order=0.6)
feature_pipeline = FeaturePipeline(steps=[normalize_price,
                                          moving_averages,
                                          difference_all])

exchange.feature_pipeline = feature_pipeline
```
