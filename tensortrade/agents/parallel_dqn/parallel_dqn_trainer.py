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
import os
import time
from datetime import datetime
from multiprocessing import Process, Queue

import numpy as np
import tensorflow as tf

from tensortrade.agents.parallel_dqn.parallel_dqn_model import ParallelDQNModel

class ParallelDQNTrainer(Process):
    def __init__(self,
                 create_env_func,
                 agent_id: str,
                 process_id: int,
                 n_envs: int,
                 memory_queue: Queue,
                 model_update_queue: Queue,
                 sync_queue: Queue,
                 done_queue: Queue,
                 n_steps: int,
                 n_episodes: int,
                 eps_end: int = 0.05,
                 eps_start: int = 0.99,
                 eps_decay_steps: int = 2000):
        super().__init__()

        self.create_env_func = create_env_func
        self.agent_id = agent_id
        self.process_id = process_id
        self.n_envs = n_envs
        self.memory_queue = memory_queue
        self.model_update_queue = model_update_queue
        self.sync_queue = sync_queue
        self.done_queue = done_queue
        self.n_steps = n_steps or np.iinfo(np.int32).max
        self.n_episodes = n_episodes or np.iinfo(np.int32).max
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

    def run(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        try:
            tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
        except IndexError:
            pass  # no GPU on this machine

        env = self.create_env_func()
        env.agent_id = self.agent_id

        episode = 1
        total_steps_done = 0
        total_reward = 0

        while episode <= self.n_episodes:

            while not self.model_update_queue.qsize() > 0:
                time.sleep(.1)

            state = env.reset()
            model = ParallelDQNModel(env=env, policy_network=self.model_update_queue.get())

            while self.model_update_queue.qsize() > 0:
                time.sleep(.1)

            done = False
            steps_done = 0

            while not done:
                threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                            np.exp(-steps_done / self.eps_decay_steps)
                action = model.get_action(state, threshold=threshold)
                next_state, reward, done, _ = env.step(action)

                self.memory_queue.put((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward
                steps_done += 1

                if self.n_steps and steps_done >= self.n_steps:
                    done = True
                    self.sync_queue.put(1)

                    print("[{}] Episode: ({}/{}) Process ID: {}   P/L: {}"
                          .format(datetime.now(), episode, self.n_episodes, self.process_id, env.portfolio.profit_loss))

                    while self.sync_queue.qsize() < self.n_envs * 2:
                        time.sleep(.1)

            episode += 1
            total_steps_done += steps_done

        mean_reward = total_reward / total_steps_done

        self.done_queue.put(mean_reward)
