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
"""IMPALA actor class."""
import dm_env
import haiku as hk
from haiku.examples.impala import agent as agent_lib
from haiku.examples.impala import learner as learner_lib
from haiku.examples.impala import util
import jax
import numpy as np


class Actor:
  """Manages the state of a single agent/environment interaction loop."""

  def __init__(
      self,
      agent: agent_lib.Agent,
      env: dm_env.Environment,
      unroll_length: int,
      learner: learner_lib.Learner,
      rng_seed: int = 42,
      logger=None,
  ):
    self._agent = agent
    self._env = env
    self._unroll_length = unroll_length
    self._learner = learner
    self._timestep = env.reset()
    self._agent_state = agent.initial_state(None)
    self._traj = []
    self._rng_key = jax.random.PRNGKey(rng_seed)

    if logger is None:
      logger = util.NullLogger()
    self._logger = logger

    self._episode_return = 0.

  def unroll(self, rng_key, frame_count: int, params: hk.Params,
             unroll_length: int) -> util.Transition:
    """Run unroll_length agent/environment steps, returning the trajectory."""
    timestep = self._timestep
    agent_state = self._agent_state
    # Unroll one longer if trajectory is empty.
    num_interactions = unroll_length + int(not self._traj)
    subkeys = jax.random.split(rng_key, num_interactions)
    for i in range(num_interactions):
      timestep = util.preprocess_step(timestep)
      agent_out, next_state = self._agent.step(subkeys[i], params, timestep,
                                               agent_state)
      transition = util.Transition(
          timestep=timestep,
          agent_out=agent_out,
          agent_state=agent_state)
      self._traj.append(transition)
      agent_state = next_state
      timestep = self._env.step(agent_out.action)

      if timestep.last():
        self._episode_return += timestep.reward
        self._logger.write({
            'num_frames': frame_count,
            'episode_return': self._episode_return,
        })
        self._episode_return = 0.
      else:
        self._episode_return += timestep.reward or 0.

      # Elide a manual agent_state reset on step_type.first(), as the ResetCore
      # already takes care of this for us.

    # Pack the trajectory and reset parent state.
    trajectory = jax.device_get(self._traj)
    trajectory = jax.tree_multimap(lambda *xs: np.stack(xs), *trajectory)
    self._timestep = timestep
    self._agent_state = agent_state
    # Keep the bootstrap timestep for next trajectory.
    self._traj = self._traj[-1:]
    return trajectory

  def unroll_and_push(self, frame_count: int, params: hk.Params):
    """Run one unroll and send trajectory to learner."""
    params = jax.device_put(params)
    self._rng_key, subkey = jax.random.split(self._rng_key)
    act_out = self.unroll(
        rng_key=subkey,
        frame_count=frame_count,
        params=params,
        unroll_length=self._unroll_length)
    self._learner.enqueue_traj(act_out)

  def pull_params(self):
    return self._learner.params_for_actor()
