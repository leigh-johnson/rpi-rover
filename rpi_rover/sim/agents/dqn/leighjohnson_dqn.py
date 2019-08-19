
# Modified Work Copyright (c) 2019, Leigh Johnson
# 
# MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Original Work Copyright 2018 The TF-Agents Authors.
# https://github.com/tensorflow/agents/blob/master/tf_agents/agents/dqn/examples/v2/setup_summary_writers.py
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
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import namedtuple
import argparse
import logging
from logging.config import fileConfig
import time
from multiprocessing import Process

import gin
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments.examples import masked_cartpole  # pylint: disable=unused-import
from tf_agents.environments.wrappers import MultiDiscreteToDiscreteWrapper

from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from gym.envs.registration import register

THREAD_MAP = {}
register(id='train-donkey-generated-track-multidiscrete-v0', entry_point='gym_donkeycar.envs.donkey_env:MultiDiscreteGeneratedTrackEnv',
    kwargs={'headless': False, 'thread_name': 'train', 'dispatcher_map': THREAD_MAP, 'thread_name': 'TrainSimThread'}
)

register(id='eval-donkey-generated-track-multidiscrete-v0', entry_point='gym_donkeycar.envs.donkey_env:MultiDiscreteGeneratedTrackEnv',
    kwargs={'headless': False, 'thread_name': 'eval',  'dispatcher_map': THREAD_MAP, 'thread_name': 'EvalSimThread'}
)

from gym_donkeycar import envs as gym_donkeycar_envs
from rpi_rover.sim.config import DONKEY_SIM_PATH

fileConfig('logging.ini')
logger = logging.getLogger()

@gin.configurable
def train_eval(
    root_dir,
    env_name='donkey-generated-track-multidiscrete-v0',
    # env_list = [
    #    "donkey-warehouse-v0",
    #    "donkey-generated-roads-v0",
    #    "donkey-avc-sparkfun-v0",
    #    "donkey-generated-track-v0"
    # ]
    num_iterations=100,
    max_episode_steps=10000,
    train_sequence_length=1,
    # Params for QNetwork
    fc_layer_params=(100,),
    # Params for QRnnNetwork
    input_fc_layer_params=(50,),
    lstm_size=(20,),
    output_fc_layer_params=(20,),

    # Params for collect
    initial_collect_steps=100,
    collect_steps_per_iteration=1,
    epsilon_greedy=0.1,
    replay_buffer_capacity=100000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=64,
    learning_rate=1e-3,
    n_step_update=1,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=1000,
    # Params for checkpoints
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=20000,
    # Params for summaries and logging
    log_interval=10,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
        eval_metrics_callback=None):

    dirs = setup_dirs(root_dir)
    summary_writers = setup_summary_writers(dirs, summaries_flush_secs)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        
        tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(f'train-{env_name}', max_episode_steps=max_episode_steps,
            env_wrappers=(MultiDiscreteToDiscreteWrapper,)
        ))
        logging.info('Initialize tf_eval_env')


        #tf_env = MultiDiscreteToDiscreteWrapper(tf_env)

        #import pdb; pdb.set_trace()

        # eval_tf_env = tf_py_environment.TFPyEnvironment(
        #     suite_gym.load('headless-' + env_name, max_episode_steps=max_episode_steps,
        #     env_wrappers=(MultiDiscreteToDiscreteWrapper,)
        #     ))

        
        #eval_tf_env = MultiDiscreteToDiscreteWrapper(eval_tf_env)
        q_net = setup_qnet(train_sequence_length, n_step_update,
                           tf_env, fc_layer_params, input_fc_layer_params, output_fc_layer_params, lstm_size)

        tf_agent = dqn_agent.DqnAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            q_network=q_net,
            epsilon_greedy=epsilon_greedy,
            n_step_update=n_step_update,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate),
            td_errors_loss_fn=common.element_wise_squared_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)
        tf_agent.initialize()

        metrics = setup_metrics(num_eval_episodes)

        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity)

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + metrics.train,
            num_steps=collect_steps_per_iteration)

        checkpointers = setup_checkpointers(
            dirs.train, tf_agent, global_step, eval_policy, replay_buffer, metrics.train)
        if use_tf_functions:
            # To speed up collect use common.function.
            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        logging.info('Init a random action policy')
        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(), tf_env.action_spec())
        

        logging.info(
            'Initialize replay buffer by collecting experience for %d steps with '
            'a random policy.', initial_collect_steps)

        dynamic_step_driver.DynamicStepDriver(
            tf_env,
            initial_collect_policy,
            observers=[replay_buffer.add_batch] + metrics.train,
            num_steps=initial_collect_steps).run()
        
        logger.info('Initialized replay buffer. Resetting environment')
        tf_env.reset()
        logger.info('Initialize eval environment for metrics collection')

        eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(f'eval-{env_name}', max_episode_steps=max_episode_steps,
            env_wrappers=(MultiDiscreteToDiscreteWrapper,)
        ))
        results = metric_utils.eager_compute(
            metrics.eval,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=summary_writers.eval,
            summary_prefix='Metrics',
        )

        logging.info('Done initializing metric')

        if eval_metrics_callback is not None:
            eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(metrics.eval)
        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0

        logging.info('Loading replay_buffer into generator')
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=train_sequence_length + 1).prefetch(3)
        iterator = iter(dataset)

        def train_step():
            experience, _ = next(iterator)
            return tf_agent.train(experience)

        if use_tf_functions:
            train_step = common.function(train_step)

        for _ in range(num_iterations):
            start_time = time.time()
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )
            for _ in range(train_steps_per_iteration):
                train_loss = train_step()
            time_acc += time.time() - start_time

            if global_step.numpy() % log_interval == 0:
                logging.info('step = %d, loss = %f', global_step.numpy(),
                            train_loss.loss)
                steps_per_sec = (global_step.numpy() -
                                 timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(
                    name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                timed_at_step = global_step.numpy()
                time_acc = 0

            for train_metric in metrics.train:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=metrics.train[:2])

            if global_step.numpy() % train_checkpoint_interval == 0:
                checkpointers.train.save(global_step=global_step.numpy())

            if global_step.numpy() % policy_checkpoint_interval == 0:
                checkpointers.policy.save(global_step=global_step.numpy())

            if global_step.numpy() % rb_checkpoint_interval == 0:
                checkpointers.replay_buffer.save(
                    global_step=global_step.numpy())

            if global_step.numpy() % eval_interval == 0:

                results = metric_utils.eager_compute(
                    metrics.eval,
                    eval_tf_env,
                    eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=summary_writers.eval,
                    summary_prefix='Metrics',
                )
                if eval_metrics_callback is not None:
                    eval_metrics_callback(results, global_step.numpy())
                metric_utils.log_metrics(metrics.eval)
            #logging.info(f'train_loss {train_loss}')

        tf_env.pyenv.close()
        eval_tf_env.pyenv.close()
        return train_loss


def setup_checkpointers(train_dir, tf_agent, global_step, eval_policy, replay_buffer, train_metrics):

    checkpointers = namedtuple(
        'checkpointers', ['train', 'policy', 'replay_buffer'])

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer)
    return checkpointers(train_checkpointer, policy_checkpointer, rb_checkpointer)

def setup_metrics(num_eval_episodes):
    metrics = namedtuple('metrics', ['train', 'eval'])

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    return metrics(train_metrics, eval_metrics)


def setup_qnet(train_sequence_length, n_step_update, tf_env, fc_layer_params, input_fc_layer_params, output_fc_layer_params, lstm_size):
    if train_sequence_length != 1 and n_step_update != 1:
        raise NotImplementedError(
            'setup_summary_writers does not currently support n-step updates with stateful '
            'networks (i.e., RNNs)')
    if train_sequence_length > 1:
        q_net = q_rnn_network.QRnnNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            input_fc_layer_params=input_fc_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params)
    else:
        q_net = q_network.QNetwork(
            tf_env.observation_spec(),
            tf_env.action_spec(),
            fc_layer_params=fc_layer_params)
    return q_net


def setup_summary_writers(dirs, summaries_flush_secs):
    writers = namedtuple('summary_writers', ['train', 'eval'])
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        dirs.train, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    summary_writers_eval = tf.compat.v2.summary.create_file_writer(
        dirs.eval, flush_millis=summaries_flush_secs * 1000)

    return writers(train_summary_writer, summary_writers_eval)


def setup_dirs(root_dir):
    dirs = namedtuple('dirs', ['root', 'train', 'eval'])
    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    return dirs(root_dir, train_dir, eval_dir)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gin-file', default=None)
    parser.add_argument('--dir', required=True)
    parser.add_argument('--sim', type=str, default=DONKEY_SIM_PATH, help='path to unity simulator. maybe be left at manual if you would like to start the sim on your own.')
    parser.add_argument('--headless', type=int, default=1, help='1 to supress graphics')
    parser.add_argument('--port', type=int, default=9091, help='port to use for websockets')


    return parser.parse_args()


if __name__ == '__main__':

    tf.compat.v1.enable_v2_behavior()
    args = parse_args()
    #we pass arguments to the donkey_gym init via these
    os.environ['DONKEY_SIM_PATH'] = args.sim
    os.environ['DONKEY_SIM_PORT'] = str(args.port)
    #os.environ['DONKEY_SIM_HEADLESS'] = str(args.headless)
    os.environ['DONKEY_SIM_MULTI'] = '1'

    gin.parse_config_files_and_bindings(args.gin_file, None)
    train_eval(args.dir)
