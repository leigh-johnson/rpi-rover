# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
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
# limitations under the License.

r"""Train and Eval DQN.

To run DQN on CartPole:

```bash
tensorboard --logdir $HOME/tmp/dqn/gym/CartPole-v0/ --port 2223 &

python tf_agents/agents/dqn/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/dqn/gym/CartPole-v0/ \
  --alsologtostderr
```

To run DQN-RNNs on MaskedCartPole:

```bash
python tf_agents/agents/dqn/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/dqn_rnn/gym/MaskedCartPole-v0/ \
  --gin_param='train_eval.env_name="MaskedCartPole-v0"' \
  --gin_param='train_eval.train_sequence_length=10' \
  --alsologtostderr
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys

from absl import app
from absl import flags
import logging
from logging.config import fileConfig


import gin
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments.examples import masked_cartpole  # pylint: disable=unused-import
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.networks import q_rnn_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.environments.wrappers import MultiDiscreteToDiscreteWrapper
from gym.envs.registration import register


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 100000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS

register(id='train-donkey-generated-track-multidiscrete-v0', entry_point='gym_donkeycar.envs.donkey_env:MultiDiscreteGeneratedTrackEnv',
    kwargs={'headless': False, 'thread_name': 'TrainSimThread'}
)

register(id='eval-donkey-generated-track-multidiscrete-v0', entry_point='gym_donkeycar.envs.donkey_env:MultiDiscreteGeneratedTrackEnv',
    kwargs={'headless': False,  'thread_name': 'EvalSimThread', 'port': 9091}
)

from gym_donkeycar import envs as gym_donkeycar_envs
from rpi_rover.sim.config import DONKEY_SIM_PATH

#fileConfig('logging.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



@gin.configurable
def train_eval(
    root_dir,
    env_name='donkey-generated-track-multidiscrete-v0',
    num_iterations=100000,
    train_sequence_length=10,
    # Params for QNetwork
    fc_layer_params=(100,),
    # Params for QRnnNetwork
    input_fc_layer_params=(50,),
    lstm_size=(20,),
    output_fc_layer_params=(20,),

    # Params for collect
    boltzmann_temperature=None,
# Temperature value to use for Boltzmann sampling of
#         the actions during data collection. The closer to 0.0, the higher the
#         probability of choosing the best action.
    initial_collect_episodes=100,
    collect_episodes_per_iteration=10,
    epsilon_greedy=0.1,
    replay_buffer_capacity=10000,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=1,

    # Params for train
    train_steps_per_iteration=1,
    batch_size=64,
    learning_rate=1e-4,
    n_step_update=1,
    gamma=0.99,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=100,
    eval_interval=10,
    # Params for checkpoints
    train_checkpoint_interval=20,
    policy_checkpoint_interval=20,
    rb_checkpoint_interval=20,
    # Params for summaries and logging
    log_interval=1,
    summary_interval=1,
    summaries_flush_secs=10,
    debug_summaries=True,
    summarize_grads_and_vars=True,
        eval_metrics_callback=None,
     # donkey gym env
    sim=DONKEY_SIM_PATH,
    max_episode_steps=100000
    ):
    """A simple train and eval for DQN."""

    os.environ['DONKEY_SIM_PATH'] = sim
    #os.environ['DONKEY_SIM_PORT'] = str(port)
    #os.environ['DONKEY_SIM_HEADLESS'] = str(args.headless)
    os.environ['DONKEY_SIM_MULTI'] = '1'

    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(global_step % summary_interval, 0)):
        tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(f'train-{env_name}', max_episode_steps=max_episode_steps,
            env_wrappers=(MultiDiscreteToDiscreteWrapper,)
        ))
        eval_tf_env = tf_py_environment.TFPyEnvironment(suite_gym.load(f'eval-{env_name}',
            max_episode_steps=max_episode_steps,
            env_wrappers=(MultiDiscreteToDiscreteWrapper,)
        ))


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
            train_sequence_length = n_step_update

        # TODO(b/127301657): Decay epsilon based on global step, cf. cl/188907839
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
            #td_errors_loss_fn=common.element_wise_huba_loss,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step,
            boltzmann_temperature=boltzmann_temperature
            )
        tf_agent.initialize()
        
        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]

        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=replay_buffer_capacity,
            
            )

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=collect_episodes_per_iteration
        )

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

        train_checkpointer.initialize_or_restore()
        rb_checkpointer.initialize_or_restore()

        if use_tf_functions:
            # To speed up collect use common.function.
            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        initial_collect_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(), tf_env.action_spec())

        # Collect initial replay data.
        logger.info(
            'Initializing replay buffer by collecting experience for %d episodes with '
            'a random policy.', initial_collect_episodes)
        dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            initial_collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=initial_collect_episodes).run()

        logger.info('Computing initial eval_metrics and eval_policy')
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            #tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        #eval_tf_env.reset()

        if eval_metrics_callback is not None:
            eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics)
        time_step = None
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=train_sequence_length + 1
        ).prefetch(3)
        iterator = iter(dataset)

        def train_step():
            experience, _ = next(iterator)
            return tf_agent.train(experience)

        if use_tf_functions:
            train_step = common.function(train_step)
        
        logger.info(f'Done with initial seed. Training for {num_iterations} iterations')
        for _ in range(num_iterations):
            start_time = time.time()
            tf_env.reset()
            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
                num_episodes=collect_episodes_per_iteration
            )

            train_loss = train_step()

            #import pdb; pdb.set_trace()
            #for _ in range(train_steps_per_iteration):
            time_acc += time.time() - start_time

            if global_step.numpy() % log_interval == 0:
                logger.info('step = %d, loss = %f', global_step.numpy(),
                             train_loss.loss)
                steps_per_sec = (global_step.numpy() -
                                 timed_at_step) / time_acc
                logger.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(
                    name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                timed_at_step = global_step.numpy()
                time_acc = 0

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=train_metrics[:2])

            if global_step.numpy() % train_checkpoint_interval == 0:
                train_checkpointer.save(global_step=global_step.numpy())

            if global_step.numpy() % policy_checkpoint_interval == 0:
                policy_checkpointer.save(global_step=global_step.numpy())

            if global_step.numpy() % rb_checkpoint_interval == 0:
                rb_checkpointer.save(global_step=global_step.numpy())

            if global_step.numpy() % eval_interval == 0:
                logger.info('Computing eval_policy metrics')
                eval_tf_env.reset()
                results = metric_utils.eager_compute(
                    eval_metrics,
                    eval_tf_env,
                    eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=eval_summary_writer,
                    summary_prefix='Metrics',
                )
                if eval_metrics_callback is not None:
                    eval_metrics_callback(results, global_step.numpy())
                metric_utils.log_metrics(eval_metrics)
        return train_loss


def main(_):
    #logger.set_verbosity(logger.INFO)
    tf.compat.v1.enable_v2_behavior()
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train_eval(FLAGS.root_dir, num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
