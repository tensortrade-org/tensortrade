import tensorflow as tf

from gym.spaces import Space
from typing import Dict, List, Tuple
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork


class TensorflowModel(TFModelV2):
    """Implementation of Tensorflow model in RLlib."""

    def __init__(self,
                 obs_space: Space,
                 action_space: Space,
                 num_outputs: int,
                 model_config: dict,
                 name: str,
                 network: TFModelV2 = FullyConnectedNetwork):
        """
        Arguments:
            obs_space: The observation space that will be passed into the model.
            action_space: The action space that will be output by the model.
            num_outputs: The number of outputs from the model.
            model_config: A dictionary of configuration variables.
            name: The name of the model.
            network: The TF network to of the underlying model.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self._model = network(obs_space, action_space, num_outputs, model_config, name)
        self.register_variables(self._model.variables())

    def forward(self, input_dict: Dict[str, tf.Tensor], state: List[tf.Tensor], seq_lens: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Call the model with the given input tensors and state.

        Arguments:
            input_dict: Dictionary of input tensors, including "obs", "obs_flat", "prev_action", "prev_reward", "is_training".
            state(list): list of state tensors with sizes matching those returned by get_initial_state + the batch dimension.
            seq_lens: 1D tensor holding input sequence lengths.

        Returns:
            (outputs, state): The model output tensor of size [BATCH, num_outputs].
        """
        return self._model.forward(input_dict, state, seq_lens)

    def value_function(self) -> List[float]:
        """Return the value function estimate for the most recent forward pass through the model.

        Returns:
            Value estimate tensor of shape [BATCH].
        """
        return self._model.value_function()
