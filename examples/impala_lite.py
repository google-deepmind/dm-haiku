# python3
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""A simple, single-process IMPALA in JAX with Haiku.

This implementation is a simple, minimal implementation of IMPALA.
For a more full-fledged implementation, see examples/impala/README.md.

See: https://arxiv.org/abs/1802.01561
"""

import functools
import queue
import threading
from typing import Any, Callable, NamedTuple, Tuple

from absl import app
from absl import logging
from bsuite.experiments.catch import catch
import dm_env
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as np
import rlax

OptState = Any


class Transition(NamedTuple):
  timestep: dm_env.TimeStep
  action: int
  agent_out: Any


class SimpleNet(hk.Module):
  """A simple network."""

  def __init__(self, num_actions: int):
    super(SimpleNet, self).__init__()
    self._num_actions = num_actions

  def __call__(
      self,
      timestep: dm_env.TimeStep,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Process a batch of observations."""
    torso = hk.Sequential([hk.Flatten(),
                           hk.Linear(128), jax.nn.relu,
                           hk.Linear(64), jax.nn.relu])
    hidden = torso(timestep.observation)
    policy_logits = hk.Linear(self._num_actions)(hidden)
    baseline = hk.Linear(1)(hidden)
    baseline = jnp.squeeze(baseline, axis=-1)
    return policy_logits, baseline


class Agent:
  """A simple, feed-forward agent."""

  def __init__(self, net_apply):
    self._net = net_apply
    self._discount = 0.99

  @functools.partial(jax.jit, static_argnums=0)
  def step(
      self,
      params: hk.Params,
      rng: jnp.ndarray,
      timestep: dm_env.TimeStep,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Steps on a single observation."""
    timestep = jax.tree_map(lambda t: jnp.expand_dims(t, 0), timestep)
    logits, _ = self._net(params, timestep)
    logits = jnp.squeeze(logits, axis=0)
    action = hk.multinomial(rng, logits, num_samples=1)
    action = jnp.squeeze(action, axis=-1)
    return action, logits

  def loss(self, params: hk.Params, trajs: Transition) -> jnp.ndarray:
    """Computes a loss of trajs wrt params."""
    # Re-run the agent over the trajectories.
    # Due to https://github.com/google/jax/issues/1459, we use hk.BatchApply
    # instead of vmap.
    # BatchApply turns the input tensors from [T, B, ...] into [T*B, ...].
    # We `functools.partial` params in so it does not get transformed.
    net_curried = hk.BatchApply(functools.partial(self._net, params))
    learner_logits, baseline_with_bootstrap = net_curried(trajs.timestep)

    # Separate the bootstrap from the value estimates.
    baseline = baseline_with_bootstrap[:-1]
    baseline_tp1 = baseline_with_bootstrap[1:]

    # Remove bootstrap timestep from non-observations.
    _, actions, behavior_logits = jax.tree_map(lambda t: t[:-1], trajs)
    learner_logits = learner_logits[:-1]

    # Shift step_type/reward/discount back by one, so that actions match the
    # timesteps caused by the action.
    timestep = jax.tree_map(lambda t: t[1:], trajs.timestep)
    discount = timestep.discount * self._discount
    # The step is uninteresting if we transitioned LAST -> FIRST.
    mask = jnp.not_equal(timestep.step_type, int(dm_env.StepType.FIRST))
    mask = mask.astype(jnp.float32)

    # Compute v-trace returns.
    vtrace_td_error_and_advantage = jax.vmap(
        rlax.vtrace_td_error_and_advantage, in_axes=1, out_axes=1)
    rhos = rlax.categorical_importance_sampling_ratios(learner_logits,
                                                       behavior_logits, actions)
    vtrace_returns = vtrace_td_error_and_advantage(baseline, baseline_tp1,
                                                   timestep.reward, discount,
                                                   rhos)

    # Note that we use mean here, rather than sum as in canonical rawbeast.
    # Compute policy gradient loss.
    pg_advantage = jax.lax.stop_gradient(vtrace_returns.pg_advantage)
    tb_pg_loss_fn = jax.vmap(rlax.policy_gradient_loss, in_axes=1, out_axes=1)
    pg_loss = tb_pg_loss_fn(learner_logits, actions, pg_advantage, mask)
    pg_loss = jnp.mean(pg_loss)

    # Baseline loss.
    bl_loss = 0.5 * jnp.mean(jnp.square(vtrace_returns.errors) * mask)

    # Entropy regularization.
    ent_loss_fn = jax.vmap(rlax.entropy_loss, in_axes=1, out_axes=1)
    ent_loss = ent_loss_fn(learner_logits, mask)
    ent_loss = jnp.mean(ent_loss)

    total_loss = pg_loss + 0.5 * bl_loss + 0.01 * ent_loss
    return total_loss


def preprocess_step(ts: dm_env.TimeStep) -> dm_env.TimeStep:
  # reward: None -> 0, discount: None -> 1,
  # scalar -> np.array(), and StepType -> int.
  if ts.reward is None:
    ts = ts._replace(reward=0.)
  if ts.discount is None:
    ts = ts._replace(discount=1.)
  return jax.tree_map(np.asarray, ts)


def run_actor(
    agent: Agent,
    rng_key: jnp.ndarray,
    get_params: Callable[[], hk.Params],
    enqueue_traj: Callable[[Transition], None],
    unroll_len: int,
    num_trajectories: int,
):
  """Runs an actor to produce num_trajectories trajectories."""
  env = catch.Catch()
  state = env.reset()
  traj = []

  for i in range(num_trajectories):
    params = get_params()
    # The first rollout is one step longer.
    for _ in range(unroll_len + int(i == 0)):
      rng_key, step_key = jax.random.split(rng_key)
      state = preprocess_step(state)
      action, logits = agent.step(params, step_key, state)
      transition = Transition(state, action, logits)
      traj.append(transition)
      state = env.step(action)
      if state.step_type == dm_env.StepType.LAST:
        logging.log_every_n(logging.INFO, 'Episode ended with reward: %s', 5,
                            state.reward)

    # Stack and send the trajectory.
    stacked_traj = jax.tree_multimap(lambda *ts: np.stack(ts), *traj)
    enqueue_traj(stacked_traj)
    # Reset the trajectory, keeping the last timestep.
    traj = traj[-1:]


class Learner:
  """Slim wrapper around an agent/optimizer pair."""

  def __init__(self, agent: Agent, opt_update):
    self._agent = agent
    self._opt_update = opt_update

  @functools.partial(jax.jit, static_argnums=0)
  def update(
      self,
      params: hk.Params,
      opt_state: OptState,
      trajs: Transition,
  ) -> Tuple[hk.Params, OptState]:
    g = jax.grad(self._agent.loss)(params, trajs)
    updates, new_opt_state = self._opt_update(g, opt_state)
    return optix.apply_updates(params, updates), new_opt_state


def run(*, trajectories_per_actor, num_actors, unroll_len):
  """Runs the example."""

  # Construct the agent network. We need a sample environment for its spec.
  env = catch.Catch()
  num_actions = env.action_spec().num_values
  net = hk.transform(lambda ts: SimpleNet(num_actions)(ts))  # pylint: disable=unnecessary-lambda

  # Construct the agent and learner.
  agent = Agent(net.apply)
  opt = optix.rmsprop(1e-1, decay=0.99, eps=0.1)
  learner = Learner(agent, opt.update)

  # Initialize the optimizer state.
  sample_ts = env.reset()
  sample_ts = preprocess_step(sample_ts)
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
  for i in range(num_actors):
    key = jax.random.PRNGKey(i)
    args = (agent, key, current_params, q.put, unroll_len,
            trajectories_per_actor)
    threading.Thread(target=run_actor, args=args).start()

  # Run the learner.
  num_steps = num_actors * trajectories_per_actor // batch_size
  for i in range(num_steps):
    traj = dequeue()
    params, opt_state = learner.update(params, opt_state, traj)


def main(_):
  run(trajectories_per_actor=500, num_actors=2, unroll_len=20)

if __name__ == '__main__':
  app.run(main)
