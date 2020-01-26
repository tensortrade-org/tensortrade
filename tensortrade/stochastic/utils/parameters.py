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


import random


class ModelParameters:
    """
    Encapsulates model parameters
    """

    def __init__(self,
                 all_s0, all_time, all_delta, all_sigma, gbm_mu,
                 jumps_lamda=0.0, jumps_sigma=0.0, jumps_mu=0.0,
                 cir_a=0.0, cir_mu=0.0, all_r0=0.0, cir_rho=0.0,
                 ou_a=0.0, ou_mu=0.0,
                 heston_a=0.0, heston_mu=0.0, heston_vol0=0.0):
        # This is the starting asset value
        self.all_s0 = all_s0
        # This is the amount of time to simulate for
        self.all_time = all_time
        # This is the delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
        self.all_delta = all_delta
        # This is the volatility of the stochastic processes
        self.all_sigma = all_sigma
        # This is the annual drift factor for geometric brownian motion
        self.gbm_mu = gbm_mu
        # This is the probability of a jump happening at each point in time
        self.lamda = jumps_lamda
        # This is the volatility of the jump size
        self.jumps_sigma = jumps_sigma
        # This is the average jump size
        self.jumps_mu = jumps_mu
        # This is the rate of mean reversion for Cox Ingersoll Ross
        self.cir_a = cir_a
        # This is the long run average interest rate for Cox Ingersoll Ross
        self.cir_mu = cir_mu
        # This is the starting interest rate value
        self.all_r0 = all_r0
        # This is the correlation between the wiener processes of the Heston model
        self.cir_rho = cir_rho
        # This is the rate of mean reversion for Ornstein Uhlenbeck
        self.ou_a = ou_a
        # This is the long run average interest rate for Ornstein Uhlenbeck
        self.ou_mu = ou_mu
        # This is the rate of mean reversion for volatility in the Heston model
        self.heston_a = heston_a
        # This is the long run average volatility for the Heston model
        self.heston_mu = heston_mu
        # This is the starting volatility value for the Heston model
        self.heston_vol0 = heston_vol0


def default(base_price, t_gen, delta):
    return ModelParameters(
                    all_s0=base_price,
                    all_r0=0.5,
                    all_time=t_gen,
                    all_delta=delta,
                    all_sigma=0.125,
                    gbm_mu=0.058,
                    jumps_lamda=0.00125,
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


def random(base_price, t_gen, delta):
    return ModelParameters(
        all_s0=base_price,
        all_r0=0.5,
        all_time=t_gen,
        all_delta=delta,
        all_sigma=random.uniform(0.1, 0.8),
        gbm_mu=random.uniform(-0.3, 0.6),
        jumps_lamda=random.uniform(0.0071, 0.6),
        jumps_sigma=random.uniform(-0.03, 0.04),
        jumps_mu=random.uniform(-0.2, 0.2),
        cir_a=3.0,
        cir_mu=0.5,
        cir_rho=0.5,
        ou_a=3.0,
        ou_mu=0.5,
        heston_a=random.uniform(1, 5),
        heston_mu=random.uniform(0.156, 0.693),
        heston_vol0=0.06125
    )
