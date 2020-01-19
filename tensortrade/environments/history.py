
import pandas as pd
import numpy as np


class History(object):

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.observations = pd.DataFrame()

    def push(self, observation: dict):
        """Saves an observation."""
        self.observations = self.observations.append(
            observation,
            ignore_index=True
        )
        if len(self.observations) > self.window_size:
            self.observations = self.observations[-self.window_size:]

    def observe(self) -> np.array:
        """Returns the observation to be observed by the agent."""
        observation = self.observations.copy()

        if len(observation) < self.window_size:
            size = self.window_size - len(observation)
            padding = np.zeros((size, observation.shape[1]))
            padding = pd.DataFrame(padding, columns=self.observations.columns)
            observation = pd.concat([padding, observation], ignore_index=True, sort=False)

        if isinstance(observation, pd.DataFrame):
            observation = observation.fillna(0, axis=1)
            observation = observation.values

        observation = np.nan_to_num(observation)

        return observation
