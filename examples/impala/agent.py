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
"""A stateless agent interface."""
import collections
import functools
from typing import Any, Callable, Optional, Tuple

import dm_env
import haiku as hk
from haiku.examples.impala import util
import jax
import jax.numpy as jnp

AgentOutput = collections.namedtuple("AgentOutput",
                                     ["policy_logits", "values", "action"])

Action = int
Nest = Any
NetFactory = Callable[[int], hk.RNNCore]


class Agent:
  """A stateless agent interface."""

  def __init__(self, num_actions: int, obs_spec: Nest,
               net_factory: NetFactory):
    """Constructs an Agent object.

    Args:
      num_actions: Number of possible actions for the agent. Assumes a flat,
        discrete, 0-indexed action space.
      obs_spec: The observation spec of the environment.
      net_factory: A function from num_actions to a Haiku module representing
        the agent. This module should have an initial_state() function and an
        unroll function.
    """
    self._obs_spec = obs_spec
    net_factory = functools.partial(net_factory, num_actions)
    # Instantiate two hk.transforms() - one for getting the initial state of the
    # agent, another for actually initializing and running the agent.
    _, self._initial_state_apply_fn = hk.transform(
        lambda batch_size: net_factory().initial_state(batch_size))

    self._init_fn, self._apply_fn = hk.transform(
        lambda obs, state: net_factory().unroll(obs, state))

  @functools.partial(jax.jit, static_argnums=0)
  def initial_params(self, rng_key):
    """Initializes the agent params given the RNG key."""
    dummy_inputs = jax.tree_map(lambda t: jnp.zeros(t.shape, t.dtype),
                                self._obs_spec)
    dummy_inputs = util.preprocess_step(dm_env.restart(dummy_inputs))
    dummy_inputs = jax.tree_map(lambda t: t[None, None, ...], dummy_inputs)
    return self._init_fn(rng_key, dummy_inputs, self.initial_state(1))

  @functools.partial(jax.jit, static_argnums=(0, 1))
  def initial_state(self, batch_size: Optional[int]):
    """Returns agent initial state."""
    # We expect that generating the initial_state does not require parameters.
    return self._initial_state_apply_fn(None, batch_size)

  @functools.partial(jax.jit, static_argnums=(0,))
  def step(
      self,
      rng_key,
      params: hk.Params,
      timestep: dm_env.TimeStep,
      state: Nest,
  ) -> Tuple[AgentOutput, Nest]:
    """For a given single-step, unbatched timestep, output the chosen action."""
    # Pad timestep, state to be [T, B, ...] and [B, ...] respectively.
    timestep = jax.tree_map(lambda t: t[None, None, ...], timestep)
    state = jax.tree_map(lambda t: t[None, ...], state)

    net_out, next_state = self._apply_fn(params, timestep, state)
    # Remove the padding from above.
    net_out = jax.tree_map(lambda t: jnp.squeeze(t, axis=(0, 1)), net_out)
    next_state = jax.tree_map(lambda t: jnp.squeeze(t, axis=0), next_state)
    # Sample an action and return.
    action = hk.multinomial(rng_key, net_out.policy_logits, num_samples=1)
    action = jnp.squeeze(action, axis=-1)
    return AgentOutput(net_out.policy_logits, net_out.value, action), next_state

  def unroll(
      self,
      params: hk.Params,
      trajectory: dm_env.TimeStep,
      state: Nest,
  ) -> AgentOutput:
    """Unroll the agent along trajectory."""
    net_out, _ = self._apply_fn(params, trajectory, state)
    return AgentOutput(net_out.policy_logits, net_out.value, action=[])
