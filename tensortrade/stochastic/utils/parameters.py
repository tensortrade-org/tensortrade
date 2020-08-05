# Copyright 2019 The TensorTrade Authors.
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

from random import uniform


class ModelParameters:
    """Defines the model parameters of different stock price models.

    Parameters
    ----------
    all_s0 : float
        The initial asset value.
    all_time : int
        The amount of time to simulate for.
    all_delta : float
        The rate of time.
        (e.g. 1/252 = daily, 1/12 = monthly)
    all_sigma : float
        The volatility of the stochastic processes.
    gbm_mu : float
        The annual drift factor for geometric brownian motion.
    jumps_lambda : float, default 0.0
        The probability of a jump happening at each point in time.
    jumps_sigma : float, default 0.0
        The volatility of the jump size.
    jumps_mu : float, default 0.0
        The average jump size.
    cir_a : float, default 0.0
        The rate of mean reversion for Cox Ingersoll Ross.
    cir_mu : float, default 0.0
        The long run average interest rate for Cox Ingersoll Ross.
    all_r0 : float, default 0.0
        The starting interest rate value.
    cir_rho : float, default 0.0
        The correlation between the wiener processes of the Heston model.
    ou_a : float, default 0.0
        The rate of mean reversion for Ornstein Uhlenbeck.
    ou_mu : float, default 0.0
        The long run average interest rate for Ornstein Uhlenbeck.
    heston_a : float, default 0.0
        The rate of mean reversion for volatility in the Heston model.
    heston_mu : float, default 0.0
        The long run average volatility for the Heston model.
    heston_vol0 : float, default 0.0
        The starting volatility value for the Heston model.
    """

    def __init__(self,
                 all_s0: float,
                 all_time: int,
                 all_delta: float,
                 all_sigma: float,
                 gbm_mu: float,
                 jumps_lambda: float = 0.0,
                 jumps_sigma: float = 0.0,
                 jumps_mu: float = 0.0,
                 cir_a: float = 0.0,
                 cir_mu: float = 0.0,
                 all_r0: float = 0.0,
                 cir_rho: float = 0.0,
                 ou_a: float = 0.0,
                 ou_mu: float = 0.0,
                 heston_a: float = 0.0,
                 heston_mu: float = 0.0,
                 heston_vol0: float = 0.0) -> None:
        self.all_s0 = all_s0
        self.all_time = all_time
        self.all_delta = all_delta
        self.all_sigma = all_sigma
        self.gbm_mu = gbm_mu
        self.lamda = jumps_lambda
        self.jumps_sigma = jumps_sigma
        self.jumps_mu = jumps_mu
        self.cir_a = cir_a
        self.cir_mu = cir_mu
        self.all_r0 = all_r0
        self.cir_rho = cir_rho
        self.ou_a = ou_a
        self.ou_mu = ou_mu
        self.heston_a = heston_a
        self.heston_mu = heston_mu
        self.heston_vol0 = heston_vol0


def default(base_price: float, t_gen: int, delta: float) -> 'ModelParameters':
    """Creates a basic model parameter set with key parameters specified default
    parameters.

    Parameters
    ----------
    base_price : float
        The base price to use for price generation.
    t_gen : int
        The number of bars to generate.
    delta : float
        The time delta to use.

    Returns
    -------
    `ModelParameters`
        The default model parameters to use.
    """
    return ModelParameters(
        all_s0=base_price,
        all_r0=0.5,
        all_time=t_gen,
        all_delta=delta,
        all_sigma=0.125,
        gbm_mu=0.058,
        jumps_lambda=0.00125,
        jumps_sigma=0.001,
        jumps_mu=-0.2,
        cir_a=3.0,
        cir_mu=0.5,
        cir_rho=0.5,
        ou_a=3.0,
        ou_mu=0.5,
        heston_a=0.25,
        heston_mu=0.35,
        heston_vol0=0.06125
    )


def random(base_price: float, t_gen: int, delta: float) -> 'ModelParameters':
    """Creates a random model parameter set with key parameters specified
    default parameters.

    Parameters
    ----------
    base_price : float
        The base price to use for price generation.
    t_gen : int
        The number of bars to generate.
    delta : int
        The time delta to use.

    Returns
    -------
    `ModelParameters`
        The random model parameters to use.
    """
    return ModelParameters(
        all_s0=base_price,
        all_r0=0.5,
        all_time=t_gen,
        all_delta=delta,
        all_sigma=uniform(0.1, 0.8),
        gbm_mu=uniform(-0.3, 0.6),
        jumps_lambda=uniform(0.0071, 0.6),
        jumps_sigma=uniform(-0.03, 0.04),
        jumps_mu=uniform(-0.2, 0.2),
        cir_a=3.0,
        cir_mu=0.5,
        cir_rho=0.5,
        ou_a=3.0,
        ou_mu=0.5,
        heston_a=uniform(1, 5),
        heston_mu=uniform(0.156, 0.693),
        heston_vol0=0.06125
    )
