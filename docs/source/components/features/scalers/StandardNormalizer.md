# StandardNormalizer

A transformer for normalizing values within a feature pipeline by removing the mean and scaling to unit variance.

## Class Parameters

- `columns`
  - A list of column names to normalize.
- `feature_min`
  - The minimum value in the range to scale to.
- `feature_max`
  - The maximum value in the range to scale to.
- `inplace`
  - If `False`, a new column will be added to the output for each input column.

## Properties and Setters

- None

## Functions

Below are the functions that the `StandardNormalizer` uses to effectively operate.

### Private

_None_

### Public

- `transform`
  - Apply the pipeline of feature transformations to an observation frame.
- `reset`
  - Resets the history of the standard scaler.

## Use Cases:

**Use Case #1: Different Input Spaces**

This `StandardNormalizer` operates differently depending on if we pretransform the observation to an ndarray or keep it as a pandas dataframe.

```py
from tensortrade.features import FeaturePipeline
from tensortrade.features.scalers import StandardNormalizer
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
