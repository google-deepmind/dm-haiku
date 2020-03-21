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
"""Tests for haiku.examples.impala_lite."""
import queue
import threading

import jax
import numpy as np
from absl.testing import absltest
from jax.experimental import optix
from bsuite.experiments.catch import catch

import haiku as hk
from examples.impala_lite import Agent, Learner, SimpleNet, preprocess_step, \
  run_actor


class ImpalaLiteTest(absltest.TestCase):

  def test_impala_integration(self):

    # Construct the agent network. We need a sample environment for its spec.
    env = catch.Catch()
    num_actions = env.action_spec().num_values
    net = hk.transform(lambda ts: SimpleNet(num_actions)(ts))  # pylint: disable=unnecessary-lambda
    self.assertEqual(len(net), 2)

    # Construct the agent and learner.
    agent = Agent(net.apply)
    opt = optix.rmsprop(1e-1, decay=0.99, eps=0.1)
    learner = Learner(agent, opt.update)

    # Initialize the optimizer state.
    sample_ts = env.reset()
    sample_ts = preprocess_step(sample_ts)
    self.assertEqual(sample_ts.observation.shape, (10, 5))

    ts_with_batch = jax.tree_map(lambda t: np.expand_dims(t, 0), sample_ts)
    params = jax.jit(net.init)(jax.random.PRNGKey(428), ts_with_batch)
    opt_state = opt.init(params)

    # Create accessor and queueing functions.
    current_params = lambda: params
    batch_size = 2
    q = queue.Queue(maxsize=batch_size)

    def dequeue():
      batch = []
      for _ in range(batch_size):
        batch.append(q.get())
      batch = jax.tree_multimap(lambda *ts: np.stack(ts, axis=1), *batch)
      return jax.device_put(batch)

    # Start the actors.
    num_actors = 2
    trajectories_per_actor = 1
    unroll_len = 20
    for i in range(num_actors):
      key = jax.random.PRNGKey(i)
      args = (agent, key, current_params, q.put, unroll_len,
              trajectories_per_actor)
      threading.Thread(target=run_actor, args=args).start()

    # Run the learner.
    num_steps = num_actors * trajectories_per_actor // batch_size
    self.assertEqual(num_steps, 1)
    for i in range(num_steps):
      traj = dequeue()
      params, opt_state = learner.update(params, opt_state, traj)


if __name__ == '__main__':
  absltest.main()
