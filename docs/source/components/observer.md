# Observer

The `Observer` component is what the environment uses to represent the way the agent sees the environment. The way it is built in the `default` environment is with the help of the `DataFeed`. There were many reasons for this decision, but the most important ones are that we would like to be able to compute path-dependent observations in a reliable way. We would also like to minimize lookahead biases and we would also like it to translate well into the live setting as well. To leave the opportunities broad, however, all you need to do is fill out the `observe` method to determine what your agent sees next.

# Default
The `default` observer at the moment uses a structure of a look-back window in order to create an observation. After the `feed` is made the environment will use a window of past observations and will have the shape `(window_size, n_features)`.

# IntradayObserver
The `IntradayObserver` divides the DataFeed into episodes of single trading days. It takes the same parameters as the `default` observer and additional parameters of `stop_time` of type `datetime.time` and `randomize` of type `bool`. `stop_time` is the `datetime.time` of the `timestamp` in the `DataFrame` that marks the end of the episode. It is imperative to ensure that each trading day includes the respective `timestamp`. `randomize` determines if the episode, or trading day, should be selected randomly or in the sequence of the `DataFeed`. The `IntradayObserver` requires that the `DataFeed` include a `Stream` of timestamps named `'timestamp'`.
