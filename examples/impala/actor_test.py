# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Actor test."""
from absl.testing import absltest
from bsuite.experiments.catch import catch
import dm_env
from haiku.examples.impala import actor as actor_lib
from haiku.examples.impala import agent as agent_lib
from haiku.examples.impala import haiku_nets
from haiku.examples.impala import learner as learner_lib
from haiku.examples.impala import util
import jax
import mock
import numpy as np
import tree


class CatchTest(absltest.TestCase):

  def setUp(self):
    super(CatchTest, self).setUp()
    self.env = catch.Catch()
    self.action_spec = self.env.action_spec()
    self.num_actions = self.action_spec.num_values
    self.obs_spec = self.env.observation_spec()
    self.agent = agent_lib.Agent(
        num_actions=self.num_actions,
        obs_spec=self.obs_spec,
        net_factory=haiku_nets.CatchNet,
    )

    self.key = jax.random.PRNGKey(42)
    self.key, subkey = jax.random.split(self.key)
    self.initial_params = self.agent.initial_params(subkey)

  def test_unroll(self):
    mock_learner = mock.MagicMock()
    traj_len = 10
    actor = actor_lib.Actor(
        agent=self.agent,
        env=self.env,
        learner=mock_learner,
        unroll_length=traj_len,
    )
    self.key, subkey = jax.random.split(self.key)
    act_out = actor.unroll(
        rng_key=subkey,
        frame_count=0,
        params=self.initial_params,
        unroll_length=traj_len)

    self.assertIsInstance(act_out, util.Transition)
    self.assertIsInstance(act_out.timestep, dm_env.TimeStep)
    self.assertLen(act_out.timestep.reward.shape, 1)
    self.assertEqual(act_out.timestep.reward.shape, (traj_len + 1,))
    self.assertLen(act_out.timestep.discount.shape, 1)
    self.assertEqual(act_out.timestep.discount.shape, (traj_len + 1,))
    self.assertLen(act_out.timestep.step_type.shape, 1)
    self.assertEqual(act_out.timestep.step_type.shape, (traj_len + 1,))

    self.assertLen(act_out.timestep.observation.shape, 3)
    self.assertEqual(act_out.timestep.observation.shape,
                     (traj_len + 1,) + self.obs_spec.shape)

    self.assertIsInstance(act_out.agent_out, agent_lib.AgentOutput)
    self.assertLen(act_out.agent_out.action.shape, 1)
    self.assertEqual(act_out.agent_out.action.shape, (traj_len + 1,))

    self.assertLen(act_out.agent_out.policy_logits.shape, 2)
    self.assertEqual(act_out.agent_out.policy_logits.shape,
                     (traj_len + 1, self.num_actions))

    self.assertLen(act_out.agent_out.values.shape, 1)
    self.assertEqual(act_out.agent_out.values.shape, (traj_len + 1,))

    self.assertEqual(act_out.agent_state.shape, (traj_len + 1,))

  def test_sync_params(self):
    mock_learner = mock.MagicMock()
    frame_count = 428
    params = self.initial_params
    mock_learner.params_for_actor.return_value = frame_count, params
    traj_len = 10
    actor = actor_lib.Actor(
        agent=self.agent,
        env=self.env,
        learner=mock_learner,
        unroll_length=traj_len,
    )
    received_frame_count, received_params = actor.pull_params()
    self.assertEqual(received_frame_count, frame_count)
    tree.assert_same_structure(received_params, params)
    tree.map_structure(np.testing.assert_array_almost_equal, received_params,
                       params)

  def test_unroll_and_push(self):
    traj_len = 3
    mock_learner = mock.create_autospec(learner_lib.Learner, instance=True)
    actor = actor_lib.Actor(
        agent=self.agent,
        env=self.env,
        learner=mock_learner,
        unroll_length=traj_len,
    )
    actor.unroll_and_push(0, self.initial_params)

    mock_learner.enqueue_traj.assert_called_once()
    act_out = mock_learner.enqueue_traj.call_args[0][0]

    self.assertIsInstance(act_out, util.Transition)
    self.assertIsInstance(act_out.timestep, dm_env.TimeStep)
    self.assertLen(act_out.timestep.reward.shape, 1)
    self.assertEqual(act_out.timestep.reward.shape, (traj_len + 1,))
    self.assertLen(act_out.timestep.discount.shape, 1)
    self.assertEqual(act_out.timestep.discount.shape, (traj_len + 1,))
    self.assertLen(act_out.timestep.step_type.shape, 1)
    self.assertEqual(act_out.timestep.step_type.shape, (traj_len + 1,))

    self.assertLen(act_out.timestep.observation.shape, 3)
    self.assertEqual(act_out.timestep.observation.shape,
                     (traj_len + 1,) + self.obs_spec.shape)

    self.assertIsInstance(act_out.agent_out, agent_lib.AgentOutput)
    self.assertLen(act_out.agent_out.action.shape, 1)
    self.assertEqual(act_out.agent_out.action.shape, (traj_len + 1,))

    self.assertLen(act_out.agent_out.policy_logits.shape, 2)
    self.assertEqual(act_out.agent_out.policy_logits.shape,
                     (traj_len + 1, self.num_actions))

    self.assertLen(act_out.agent_out.values.shape, 1)
    self.assertEqual(act_out.agent_out.values.shape, (traj_len + 1,))

    self.assertEqual(act_out.agent_state.shape, (traj_len + 1,))


if __name__ == '__main__':
  absltest.main()
