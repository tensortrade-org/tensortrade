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

import random
import numpy as np
import pandas as pd
import scipy as sp

from tensortrade.stochastic.processes.gbm import geometric_brownian_motion_log_returns
from tensortrade.stochastic.utils import ModelParameters, generate, convert_to_prices


# =============================================================================
# Merton Jump Diffusion Stochastic Process
# =============================================================================
def jump_diffusion_process(params: 'ModelParameters') -> 'np.array':
    """Produces a sequence of Jump Sizes which represent a jump diffusion
    process.

    These jumps are combined with a geometric brownian motion (log returns)
    to produce the Merton model.

    Parameters
    ----------
    params : ModelParameters
        The parameters for the stochastic model.

    Returns
    -------
    `np.array`
        The jump sizes for each point in time (mostly zeroes if jumps are
        infrequent).
    """
    s_n = time = 0
    small_lamda = -(1.0 / params.lamda)
    jump_sizes = []
    for _ in range(params.all_time):
        jump_sizes.append(0.0)
    while s_n < params.all_time:
        s_n += small_lamda * np.log(np.random.uniform(0, 1))
        for j in range(params.all_time):
            if time * params.all_delta <= s_n * params.all_delta <= (j + 1) * params.all_delta:
                jump_sizes[j] += random.normalvariate(params.jumps_mu, params.jumps_sigma)
                break
        time += 1
    return jump_sizes


def geometric_brownian_motion_jump_diffusion_log_returns(params: 'ModelParameters') -> 'np.array':
    """Constructs combines a geometric brownian motion process (log returns)
    with a jump diffusion process (log returns) to produce a sequence of gbm
    jump returns.

    Parameters
    ----------
    params : ModelParameters
        The parameters for the stochastic model.

    Returns
    -------
    `np.array`
        A GBM process with jumps in it
    """
    jump_diffusion = jump_diffusion_process(params)
    geometric_brownian_motion = geometric_brownian_motion_log_returns(params)
    return np.add(jump_diffusion, geometric_brownian_motion)


def geometric_brownian_motion_jump_diffusion_levels(params: 'ModelParameters') -> 'np.array':
    """Converts a sequence of gbm jmp returns into a price sequence which
    evolves according to a geometric brownian motion but can contain jumps
    at any point in time.

    Parameters
    ----------
    params : ModelParameters
        The parameters for the stochastic model.

    Returns
    -------
    `np.array`
        The price levels.
    """
    return convert_to_prices(params, geometric_brownian_motion_jump_diffusion_log_returns(params))


# =============================================================================
# Heston Stochastic Volatility Process
# =============================================================================
def cox_ingersoll_ross_heston(params: 'ModelParameters') -> 'np.array':
    """Constructs the rate levels of a mean-reverting cox ingersoll ross process.

    Used to model interest rates as well as stochastic volatility in the Heston
    model. The returns between the underlying and the stochastic volatility
    should be correlated we pass a correlated Brownian motion process into the
    method from which the interest rate levels are constructed. The other
    correlated process are used in the Heston model.

    Parameters
    ----------
    params : ModelParameters
        The parameters for the stochastic model.

    Returns
    -------
    `np.array`
        The interest rate levels for the CIR process
    """
    # We don't multiply by sigma here because we do that in heston
    sqrt_delta_sigma = np.sqrt(params.all_delta) * params.all_sigma
    brownian_motion_volatility = np.random.normal(loc=0, scale=sqrt_delta_sigma, size=params.all_time)
    a, mu, zero = params.heston_a, params.heston_mu, params.heston_vol0
    volatilities = [zero]
    for i in range(1, params.all_time):
        drift = a * (mu - volatilities[i - 1]) * params.all_delta
        randomness = np.sqrt(volatilities[i - 1]) * brownian_motion_volatility[i - 1]
        volatilities.append(volatilities[i - 1] + drift + randomness)
    return np.array(brownian_motion_volatility), np.array(volatilities)


def heston_construct_correlated_path(params: 'ModelParameters',
                                     brownian_motion_one: 'np.array') -> 'np.array':
    """A simplified version of the Cholesky decomposition method for just two
    assets. It does not make use of matrix algebra and is therefore quite easy
    to implement.

    Parameters
    ----------
    params : ModelParameters
        The parameters for the stochastic model.
    brownian_motion_one : `np.array`
            (Not filled)

    Returns
    -------
    `np.array`
        A correlated brownian motion path.
    """
    # We do not multiply by sigma here, we do that in the Heston model
    sqrt_delta = np.sqrt(params.all_delta)
    # Construct a path correlated to the first path
    brownian_motion_two = []
    for i in range(params.all_time - 1):
        term_one = params.cir_rho * brownian_motion_one[i]
        term_two = np.sqrt(1 - pow(params.cir_rho, 2)) * random.normalvariate(0, sqrt_delta)
        brownian_motion_two.append(term_one + term_two)
    return np.array(brownian_motion_one), np.array(brownian_motion_two)


def heston_model_levels(params: 'ModelParameters') -> 'np.array':
    """Generates price levels corresponding to the Heston model.

    The Heston model is the geometric brownian motion model with stochastic
    volatility. This stochastic volatility is given by the cox ingersoll ross
    process. Step one on this method is to construct two correlated GBM
    processes. One is used for the underlying asset prices and the other is used
    for the stochastic volatility levels.

    Parameters
    ----------
    params : ModelParameters
        The parameters for the stochastic model.

    Returns
    -------
    `np.array`
        The prices for an underlying following a Heston process

    Warnings
    --------
        This method is dodgy! Need to debug!
    """
    # Get two correlated brownian motion sequences for the volatility parameter and the underlying asset
    # brownian_motion_market, brownian_motion_vol = get_correlated_paths_simple(param)
    brownian, cir_process = cox_ingersoll_ross_heston(params)
    brownian, brownian_motion_market = heston_construct_correlated_path(params, brownian)

    heston_market_price_levels = [params.all_s0]
    for i in range(1, params.all_time):
        drift = params.gbm_mu * heston_market_price_levels[i - 1] * params.all_delta
        vol = cir_process[i - 1] * heston_market_price_levels[i - 1] * brownian_motion_market[i - 1]
        heston_market_price_levels.append(heston_market_price_levels[i - 1] + drift + vol)
    return np.array(heston_market_price_levels), np.array(cir_process)


def get_correlated_geometric_brownian_motions(params: 'ModelParameters',
                                              correlation_matrix: 'np.array',
                                              n: int) -> 'np.array':
    """Constructs a basket of correlated asset paths using the Cholesky
    decomposition method.

    Parameters
    ----------
    params : `ModelParameters`
        The parameters for the stochastic model.
    correlation_matrix : `np.array`
        An n x n correlation matrix.
    n : int
        Number of assets (number of paths to return)

    Returns
    -------
    `np.array`
        n correlated log return geometric brownian motion processes.
    """
    decomposition = sp.linalg.cholesky(correlation_matrix, lower=False)
    uncorrelated_paths = []
    sqrt_delta_sigma = np.sqrt(params.all_delta) * params.all_sigma
    # Construct uncorrelated paths to convert into correlated paths
    for i in range(params.all_time):
        uncorrelated_random_numbers = []
        for j in range(n):
            uncorrelated_random_numbers.append(random.normalvariate(0, sqrt_delta_sigma))
        uncorrelated_paths.append(np.array(uncorrelated_random_numbers))
    uncorrelated_matrix = np.asmatrix(uncorrelated_paths)
    correlated_matrix = uncorrelated_matrix * decomposition
    assert isinstance(correlated_matrix, np.matrix)
    # The rest of this method just extracts paths from the matrix
    extracted_paths = []
    for i in range(1, n + 1):
        extracted_paths.append([])
    for j in range(0, len(correlated_matrix) * n - n, n):
        for i in range(n):
            extracted_paths[i].append(correlated_matrix.item(j + i))
    return extracted_paths


def heston(base_price: int = 1,
           base_volume: int = 1,
           start_date: str = '2010-01-01',
           start_date_format: str = '%Y-%m-%d',
           times_to_generate: int = 1000,
           time_frame: str = '1h',
           params: 'ModelParameters' = None) -> 'pd.DataFrame':
    """Generates price data from the Heston model.

    Parameters
    ----------
    base_price : int, default 1
        The base price to use for price generation.
    base_volume : int, default 1
        The base volume to use for volume generation.
    start_date : str, default '2010-01-01'
        The start date of the generated data
    start_date_format : str, default '%Y-%m-%d'
        The format for the start date of the generated data.
    times_to_generate : int, default 1000
        The number of bars to make.
    time_frame : str, default '1h'
        The time frame.
    params : `ModelParameters`, optional
        The model parameters.

    Returns
    -------
    `pd.DataFrame`
        The generated data frame containing the OHLCV bars.
    """

    data_frame = generate(
        price_fn=lambda p: heston_model_levels(p)[0],
        base_price=base_price,
        base_volume=base_volume,
        start_date=start_date,
        start_date_format=start_date_format,
        times_to_generate=times_to_generate,
        time_frame=time_frame,
        params=params
    )
    return data_frame
