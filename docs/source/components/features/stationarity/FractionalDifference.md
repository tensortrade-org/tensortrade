# FractionalDifference

A transformer for differencing values within a feature pipeline by a fractional order. It removes the stationarity of the dataset available in realtime. To learn more about why non-stationarity should be converted to stationary information, please look at the blog [here](https://towardsdatascience.com/preserving-memory-in-stationary-time-series-6842f7581800).

## Class Parameters

- `columns`
  - A list of column names to difference.
- `difference_order`
  - The fractional difference order. Defaults to 0.5.
- `difference_threshold`
  - A type or str corresponding to the dtype of the `observation_space`.
- `inplace`
  - If `False`, a new column will be added to the output for each input column.

## Functions

Below are the functions that the `FractionalDifference` uses to effectively operate.

### Private

- `_difference_weights`
  - Gets the weights for ...
- `_fractional_difference`
  - Computes fractionally differenced series, with an increasing window width.

### Public

- `transform`
  - Apply the pipeline of feature transformations to an observation frame.
- `reset`
  - Resets the history of the standard scaler.

## Use Cases:

**Use Case #1: Different Input Spaces**

This `FeatureTransformer` operates differently depending on if we pretransform the observation to an ndarray or keep it as a pandas dataframe.

```py
from tensortrade.features import FeaturePipeline
from tensortrade.features.stationarity import FractionalDifference
price_columns = ["open", "high", "low", "close"]
difference_all = FractionalDifference(difference_order=0.6) # fractional difference is seen here
feature_pipeline = FeaturePipeline(steps=[difference_all])
exchange.feature_pipeline = feature_pipeline
```
