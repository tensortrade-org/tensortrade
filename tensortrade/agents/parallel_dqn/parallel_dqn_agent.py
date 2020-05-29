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
import multiprocessing as mp
import os
import time
from multiprocessing import Queue

import numpy as np
import tensorflow as tf

from tensortrade.agents import Agent
from tensortrade.agents.parallel_dqn.parallel_dqn_agent_tester import ParallelDQNAgentTester
from tensortrade.agents.parallel_dqn.parallel_dqn_model import ParallelDQNModel
from tensortrade.agents.parallel_dqn.parallel_dqn_optimizer import ParallelDQNOptimizer
from tensortrade.agents.parallel_dqn.parallel_dqn_trainer import ParallelDQNTrainer


class ParallelDQNAgent(Agent):

    def __init__(self,
                 create_env_func,
                 model: ParallelDQNModel = None):

        self.create_env_func = create_env_func
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
        self.model = model or ParallelDQNModel(create_env_func())

    def restore(self, path: str, **kwargs):
        self.model.restore(path, **kwargs)

    def save(self, path: str, **kwargs):
        self.model.save(path, agent_id=self.id, **kwargs)

    def get_action(self, state: np.ndarray, **kwargs) -> int:
        return self.model.get_action(state, **kwargs)

    def _start_trainer_process(self,
                               create_env_func,
                               agent_id,
                               process_id,
                               n_envs,
                               memory_queue,
                               model_update_queue,
                               sync_queue,
                               done_queue,
                               n_steps,
                               n_episodes,
                               eps_start,
                               eps_end,
                               eps_decay_steps):
        trainer_process = ParallelDQNTrainer(create_env_func,
                                             agent_id,
                                             process_id,
                                             n_envs,
                                             memory_queue,
                                             model_update_queue,
                                             sync_queue,
                                             done_queue,
                                             n_steps,
                                             n_episodes,
                                             eps_start,
                                             eps_end,
                                             eps_decay_steps)

        trainer_process.start()

        return trainer_process

    def test(self,
             n_steps,
             eps_end,
             eps_start,
             eps_decay_steps):

        print("TESTING")

        test_env = self.create_env_func()
        tester = ParallelDQNAgentTester(test_env, self.model)
        reward = tester.test(n_steps,
                             eps_end=eps_end,
                             eps_start=eps_start,
                             eps_decay_steps=eps_decay_steps)
        print("Reward: {}   P/L: {}".format(reward, test_env.portfolio.profit_loss))
        return reward, test_env

    def train(self,
              n_steps: int = None,
              n_episodes: int = None,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              test_model: bool = True,
              **kwargs) -> float:
        n_envs: int = kwargs.get('n_envs', mp.cpu_count() - 1)
        batch_size: int = kwargs.get('batch_size', 128)
        discount_factor: float = kwargs.get('discount_factor', 0.9999)
        learning_rate: float = kwargs.get('learning_rate', 0.0001)
        eps_start: float = kwargs.get('eps_start', 0.9)
        eps_end: float = kwargs.get('eps_end', 0.05)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', 2000)
        memory_capacity: int = kwargs.get('memory_capacity', 10000)

        memory_queue = Queue()
        model_update_queue = Queue()
        sync_queue = Queue()
        done_queue = Queue()

        print('====      AGENT ID: {}      ===='.format(self.id))

        trainers = [self._start_trainer_process(self.create_env_func,
                                                self.id,
                                                process_id,
                                                n_envs,
                                                memory_queue,
                                                model_update_queue,
                                                sync_queue,
                                                done_queue,
                                                n_steps,
                                                n_episodes,
                                                eps_start,
                                                eps_end,
                                                eps_decay_steps) for process_id in range(n_envs)]

        optimizer_process = ParallelDQNOptimizer(self.create_env_func,
                                                 self.model,
                                                 n_envs,
                                                 n_steps,
                                                 n_episodes,
                                                 memory_queue,
                                                 model_update_queue,
                                                 sync_queue,
                                                 done_queue,
                                                 discount_factor,
                                                 batch_size,
                                                 learning_rate,
                                                 memory_capacity)
        model = optimizer_process.run()
        self.model = model

        while done_queue.qsize() < n_envs:
            time.sleep(.5)

        for queue in [memory_queue, model_update_queue, sync_queue, done_queue]:
            queue.close()

        for queue in [memory_queue, model_update_queue, sync_queue, done_queue]:
            queue.join_thread()

        for trainer in trainers:
            trainer.terminate()

        for trainer in trainers:
            trainer.join()

        if save_path:
            self.save(save_path)

        if test_model:
            return self.test(n_steps,
                             eps_end,
                             eps_start,
                             eps_decay_steps)
        else:
            return self.model

