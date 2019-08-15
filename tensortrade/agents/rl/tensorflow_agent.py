import pandas as pd
import tensorflow as tf

from typing import Union, Callable, List

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.features.feature_pipeline import FeaturePipeline
from tensortrade.agents import TradingAgent


""" [WIP] """


class TensorflowAgent(TradingAgent):
    """A trading agent capable of self tuning, training, and evaluating with the TensorFlow 2 `agents` API."""

    def __init__(self, env: TradingEnvironment, feature_pipeline: FeaturePipeline):
        """
        Args:
            env: A `TradingEnvironment` instance for the agent to trade within.
            feature_pipeline: A `FeaturePipeline` instance of feature transformations.
        """
        super().__init__(env=env, feature_pipeline=feature_pipeline)

    def tune(self, steps_per_train: int, steps_per_test: int, step_cb: Callable[[pd.DataFrame], bool]) -> pd.DataFrame:
        pass

    def train(self, steps: int, callback: Callable[[pd.DataFrame], bool]) -> pd.DataFrame:
        # train_summary_writer = tf.compat.v2.summary.create_file_writer(
        #     train_dir, flush_millis=summaries_flush_secs * 1000)
        # train_summary_writer.set_as_default()

        # eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        #     eval_dir, flush_millis=summaries_flush_secs * 1000)
        # eval_metrics = [
        #     tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        #     tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
        # ]

        # global_step = tf.compat.v1.train.get_or_create_global_step()

        # with tf.compat.v2.summary.record_if(
        #         lambda: tf.math.equal(global_step % summary_interval, 0)):
        #     tf.compat.v1.set_random_seed(random_seed)
        #     eval_tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(env_name))
        #     tf_env = tf_py_environment.TFPyEnvironment(
        #         parallel_py_environment.ParallelPyEnvironment(
        #             [lambda: env_load_fn(env_name)] * num_parallel_environments))
        #     optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

        # if use_rnns:
        #     actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        #         tf_env.observation_spec(),
        #         tf_env.action_spec(),
        #         input_fc_layer_params=actor_fc_layers,
        #         output_fc_layer_params=None)
        #     value_net = value_rnn_network.ValueRnnNetwork(
        #         tf_env.observation_spec(),
        #         input_fc_layer_params=value_fc_layers,
        #         output_fc_layer_params=None)

        # tf_agent = ppo_agent.PPOAgent(
        #     tf_env.time_step_spec(),
        #     tf_env.action_spec(),
        #     optimizer,
        #     actor_net=actor_net,
        #     value_net=value_net,
        #     num_epochs=num_epochs,
        #     debug_summaries=debug_summaries,
        #     summarize_grads_and_vars=summarize_grads_and_vars,
        #     train_step_counter=global_step)
        # tf_agent.initialize()

        # environment_steps_metric = tf_metrics.EnvironmentSteps()
        # step_metrics = [
        #     tf_metrics.NumberOfEpisodes(),
        #     environment_steps_metric,
        # ]

        # train_metrics = step_metrics + [
        #     tf_metrics.AverageReturnMetric(
        #         batch_size=num_parallel_environments),
        #     tf_metrics.AverageEpisodeLengthMetric(
        #         batch_size=num_parallel_environments),
        # ]

        # eval_policy = tf_agent.policy
        # collect_policy = tf_agent.collect_policy

        # replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        #     tf_agent.collect_data_spec,
        #     batch_size=num_parallel_environments,
        #     max_length=replay_buffer_capacity)

        # collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        #     tf_env,
        #     collect_policy,
        #     observers=[replay_buffer.add_batch] + train_metrics,
        #     num_episodes=collect_episodes_per_iteration)

        # def train_step():
        #     trajectories = replay_buffer.gather_all()
        #     return tf_agent.train(experience=trajectories)

        # collect_time = 0
        # train_time = 0
        # timed_at_step = global_step.numpy()

        # while environment_steps_metric.result() < num_environment_steps:
        #     global_step_val = global_step.numpy()

        # start_time = time.time()
        # collect_driver.run()
        # collect_time += time.time() - start_time

        # start_time = time.time()
        # total_loss, _ = train_step()
        # replay_buffer.clear()
        # train_time += time.time() - start_time

        # for train_metric in train_metrics:
        #     train_metric.tf_summaries(train_step=global_step, step_metrics=step_metrics)

        # if global_step_val % log_interval == 0:
        #     logging.info('step = %d, loss = %f', global_step_val, total_loss)
        #     steps_per_sec = (
        #         (global_step_val - timed_at_step) / (collect_time + train_time))
        #     logging.info('%.3f steps/sec', steps_per_sec)
        #     logging.info('collect_time = {}, train_time = {}'.format(
        #         collect_time, train_time))

        #     with tf.compat.v2.summary.record_if(True):
        #         tf.compat.v2.summary.scalar(
        #             name='global_steps_per_sec', data=steps_per_sec, step=global_step)

        #     timed_at_step = global_step_val
        #     collect_time = 0
        #     train_time = 0
        pass

    def evaluate(self, steps: int, callback: Callable[[pd.DataFrame], bool]) -> pd.DataFrame:
        # metric_utils.eager_compute(
        #     eval_metrics,
        #     eval_tf_env,
        #     eval_policy,
        #     num_episodes=num_eval_episodes,
        #     train_step=global_step,
        #     summary_writer=eval_summary_writer,
        #     summary_prefix='Metrics',
        # )
        pass

    def get_action(self, observation: pd.DataFrame) -> Union[float, List[float]]:
        pass
