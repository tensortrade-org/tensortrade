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


import time
import numpy as np
import multiprocessing as mp

from typing import Callable

from tensortrade.agents import Agent
from tensortrade.agents.parallel.parallel_dqn_model import ParallelDQNModel
from tensortrade.agents.parallel.parallel_dqn_optimizer import ParallelDQNOptimizer
from tensortrade.agents.parallel.parallel_dqn_trainer import ParallelDQNTrainer
from tensortrade.agents.parallel.parallel_queue import ParallelQueue


class ParallelDQNAgent(Agent):

    def __init__(self,
                 create_env: Callable[[None], 'TradingEnvironment'],
                 model: ParallelDQNModel = None):
        self.create_env = create_env
        self.model = model or ParallelDQNModel(create_env=self.create_env)

    def restore(self, path: str, **kwargs):
        self.model.restore(path, **kwargs)

    def save(self, path: str, **kwargs):
        self.model.save(path, agent_id=self.id, **kwargs)

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        return self.model.get_action(state, **kwargs)

    def update_networks(self, model: 'ParallelDQNModel'):
        self.model.update_networks(model)

    def update_target_network(self):
        self.model.update_target_network()

    def _start_trainer_process(self,
                               create_env,
                               memory_queue,
                               model_update_queue,
                               done_queue,
                               n_steps,
                               n_episodes,
                               eps_start,
                               eps_end,
                               eps_decay_steps,
                               update_target_every):
        trainer_process = ParallelDQNTrainer(self,
                                             create_env,
                                             memory_queue,
                                             model_update_queue,
                                             done_queue,
                                             n_steps,
                                             n_episodes,
                                             eps_start,
                                             eps_end,
                                             eps_decay_steps,
                                             update_target_every)

        trainer_process.start()

        return trainer_process

    def _start_optimizer_process(self,
                                 model,
                                 n_envs,
                                 memory_queue,
                                 model_update_queue,
                                 done_queue,
                                 discount_factor,
                                 batch_size,
                                 learning_rate,
                                 memory_capacity):
        optimizer_process = ParallelDQNOptimizer(model,
                                                 n_envs,
                                                 memory_queue,
                                                 model_update_queue,
                                                 done_queue,
                                                 discount_factor,
                                                 batch_size,
                                                 learning_rate,
                                                 memory_capacity)

        optimizer_process.daemon = True
        optimizer_process.start()

        return optimizer_process

    def train(self,
              n_steps: int = None,
              n_episodes: int = None,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        n_envs: int = kwargs.get('n_envs', mp.cpu_count())
        batch_size: int = kwargs.get('batch_size', 128)
        discount_factor: float = kwargs.get('discount_factor', 0.9999)
        learning_rate: float = kwargs.get('learning_rate', 0.0001)
        eps_start: float = kwargs.get('eps_start', 0.9)
        eps_end: float = kwargs.get('eps_end', 0.05)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', 2000)
        update_target_every: int = kwargs.get('update_target_every', 1000)
        memory_capacity: int = kwargs.get('memory_capacity', 10000)

        memory_queue = ParallelQueue()
        model_update_queue = ParallelQueue()
        done_queue = ParallelQueue()

        print('====      AGENT ID: {}      ===='.format(self.id))

        trainers = [self._start_trainer_process(self.create_env,
                                                memory_queue,
                                                model_update_queue,
                                                done_queue,
                                                n_steps,
                                                n_episodes,
                                                eps_start,
                                                eps_end,
                                                eps_decay_steps,
                                                update_target_every) for _ in range(n_envs)]

        self._start_optimizer_process(self.model,
                                      n_envs,
                                      memory_queue,
                                      model_update_queue,
                                      done_queue,
                                      discount_factor,
                                      batch_size,
                                      learning_rate,
                                      memory_capacity)

        while done_queue.qsize() < n_envs:
            time.sleep(5)

        total_reward = 0

        while done_queue.qsize() > 0:
            total_reward += done_queue.get()

        for queue in [memory_queue, model_update_queue, done_queue]:
            queue.close()

        for queue in [memory_queue, model_update_queue, done_queue]:
            queue.join_thread()

        for trainer in trainers:
            trainer.terminate()

        for trainer in trainers:
            trainer.join()

        mean_reward = total_reward / n_envs

        return mean_reward
