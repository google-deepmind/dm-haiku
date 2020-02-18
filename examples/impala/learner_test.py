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
"""Tests for haiku.examples.impala.learner."""
from absl.testing import absltest
from bsuite.experiments.catch import catch
from haiku.examples.impala import actor as actor_lib
from haiku.examples.impala import agent as agent_lib
from haiku.examples.impala import haiku_nets
from haiku.examples.impala import learner as learner_lib
import jax
from jax.experimental import optix


class LearnerTest(absltest.TestCase):

  def test_integration(self):
    env = catch.Catch()
    action_spec = env.action_spec()
    num_actions = action_spec.num_values
    obs_spec = env.observation_spec()
    agent = agent_lib.Agent(
        num_actions=num_actions,
        obs_spec=obs_spec,
        net_factory=haiku_nets.CatchNet,
    )
    unroll_length = 20
    learner = learner_lib.Learner(
        agent=agent,
        rng_key=jax.random.PRNGKey(42),
        opt=optix.sgd(1e-2),
        batch_size=1,
        discount_factor=0.99,
        frames_per_iter=unroll_length,
    )
    actor = actor_lib.Actor(
        agent=agent,
        env=env,
        learner=learner,
        unroll_length=unroll_length,
    )
    frame_count, params = actor.pull_params()
    actor.unroll_and_push(frame_count=frame_count, params=params)
    learner.run(max_iterations=1)


if __name__ == '__main__':
  absltest.main()
