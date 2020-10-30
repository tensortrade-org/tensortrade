# Observer

The `Observer` component is what the environment uses to represent the way the agent sees the environment. The way it is built in the `default` environment is with the help of the `DataFeed`. There were many reasons for this decision, but the most important ones are that we would like to be able to compute path-dependent observations in a reliable way. We would also like to minimize lookahead biases and we would also like it to translate well into the live setting as well. To leave the opportunities broad, however, all you need to do is fill out the `observe` method to determine what your agent sees next.

# Default
The `default` observer at the moment uses a structure of a look-back window in order to create an observation. After the `feed` is made the environment will use a window of past observations and will have the shape `(window_size, n_features)`.
