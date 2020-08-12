# MinMaxNormalizer

A transformer for normalizing values within a feature pipeline by the column-wise extrema.

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

None

## Functions

Below are the functions that the `MinMaxNormalizer` uses to effectively operate.

### Private

_None_

### Public

- `transform`
  - Apply the pipeline of feature transformations to an observation frame.

## Use Cases:

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
