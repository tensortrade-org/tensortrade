# FeatureTransformer

As stated before in the [overview](../overview.md), We use an `ABCMeta` abstract hierarchy to handle the transformation calls of each asset. The `FeatureTransformer` is an abstract of all other price transformers available inside of the `tensortrade` library. As such, it has a set of common functions that are called on almost every transformer.

## Properties and Setters

- `columns`
  - A list of column names to normalize

## Functions

Below are the functions that the `FeatureTransformer` uses to effectively operate.

### Private

`None`

### Public

- `reset`
  - Optionally implementable method for resetting stateful transformers.
- `transform`
  - Transform the data set and return a new data frame.
