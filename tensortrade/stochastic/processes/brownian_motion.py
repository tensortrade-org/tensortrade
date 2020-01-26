# Copyright 2020 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import numpy as np

from tensortrade.stochastic.utils import ModelParameters, convert_to_prices


def brownian_motion_log_returns(params: ModelParameters):
    """
    Constructs a Wiener process (Brownian Motion).

    References:
        - http://en.wikipedia.org/wiki/Wiener_process

    Arguments:
        params : ModelParameters
            The parameters for the stochastic model.

    Returns:
        brownian motion log returns
    """
    sqrt_delta_sigma = np.sqrt(params.all_delta) * params.all_sigma
    return np.random.normal(loc=0, scale=sqrt_delta_sigma, size=params.all_time)


def brownian_motion_levels(params: ModelParameters):
    """
    Constructs a price sequence whose returns evolve according to brownian
    motion.

    Arguments:
        params : ModelParameters
            The parameters for the stochastic model.

    Returns:
        A price sequence which follows brownian motion
    """
    return convert_to_prices(params, brownian_motion_log_returns(params))
