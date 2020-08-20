# Stopper

The `Stopper` component just evaluates whether or not the environment is done running after each `step` in the environment. For example, right now the `default` environment just evaluate if the environment is done based on the profit loss of the agent. If the profit loss drops below a certain level the environment will be done. In addition, if the `feed` runs out of data to give the environment will also be done for that episode.
