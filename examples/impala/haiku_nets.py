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
"""Common networks."""
import collections

import dm_env
import haiku as hk
import jax.nn
import jax.numpy as jnp

NetOutput = collections.namedtuple('NetOutput', ['policy_logits', 'value'])


class CatchNet(hk.RNNCore):
  """A simple neural network for catch."""

  def __init__(self, num_actions, name=None):
    super(CatchNet, self).__init__(name=name)
    self._num_actions = num_actions

  def initial_state(self, batch_size):
    if batch_size is None:
      shape = []
    else:
      shape = [batch_size]
    return jnp.zeros(shape)  # Dummy.

  def __call__(self, x: dm_env.TimeStep, state):
    torso_net = hk.Sequential(
        [hk.Flatten(),
         hk.Linear(128), jax.nn.relu,
         hk.Linear(64), jax.nn.relu])
    torso_output = torso_net(x.observation)
    policy_logits = hk.Linear(self._num_actions)(torso_output)
    value = hk.Linear(1)(torso_output)
    value = jnp.squeeze(value, axis=-1)
    return NetOutput(policy_logits=policy_logits, value=value), state

  def unroll(self, x, state):
    """Unrolls more efficiently than dynamic_unroll."""
    out, _ = hk.BatchApply(self)(x, None)
    return out, state


class AtariShallowTorso(hk.Module):
  """Shallow torso for Atari, from the DQN paper."""

  def __init__(self, name=None):
    super(AtariShallowTorso, self).__init__(name=name)

  def __call__(self, x):
    torso_net = hk.Sequential([
        lambda x: x / 255.,
        hk.Conv2D(32, kernel_shape=[8, 8], stride=[4, 4], padding='VALID'),
        jax.nn.relu,
        hk.Conv2D(64, kernel_shape=[4, 4], stride=[2, 2], padding='VALID'),
        jax.nn.relu,
        hk.Conv2D(64, kernel_shape=[3, 3], stride=[1, 1], padding='VALID'),
        jax.nn.relu,
        hk.Flatten(),
        hk.Linear(512),
        jax.nn.relu,
    ])
    return torso_net(x)


class ResidualBlock(hk.Module):
  """Residual block."""

  def __init__(self, num_channels, name=None):
    super(ResidualBlock, self).__init__(name=name)
    self._num_channels = num_channels

  def __call__(self, x):
    main_branch = hk.Sequential([
        jax.nn.relu,
        hk.Conv2D(
            self._num_channels,
            kernel_shape=[3, 3],
            stride=[1, 1],
            padding='SAME'),
        jax.nn.relu,
        hk.Conv2D(
            self._num_channels,
            kernel_shape=[3, 3],
            stride=[1, 1],
            padding='SAME'),
    ])
    return main_branch(x) + x


class AtariDeepTorso(hk.Module):
  """Deep torso for Atari, from the IMPALA paper."""

  def __init__(self, name=None):
    super(AtariDeepTorso, self).__init__(name=name)

  def __call__(self, x):
    torso_out = x / 255.
    for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
      conv = hk.Conv2D(
          num_channels, kernel_shape=[3, 3], stride=[1, 1], padding='SAME')
      torso_out = conv(torso_out)
      torso_out = hk.max_pool(
          torso_out,
          window_shape=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding='SAME',
      )
      for j in range(num_blocks):
        block = ResidualBlock(num_channels, name='residual_{}_{}'.format(i, j))
        torso_out = block(torso_out)

    torso_out = jax.nn.relu(torso_out)
    torso_out = hk.Flatten()(torso_out)
    torso_out = hk.Linear(256)(torso_out)
    torso_out = jax.nn.relu(torso_out)
    return torso_out


class AtariNet(hk.RNNCore):
  """Network for Atari."""

  def __init__(self, num_actions, use_resnet, use_lstm, name=None):
    super(AtariNet, self).__init__(name=name)
    self._num_actions = num_actions
    self._use_resnet = use_resnet
    self._use_lstm = use_lstm
    self._core = hk.ResetCore(hk.LSTM(256))

  def initial_state(self, batch_size):
    return self._core.initial_state(batch_size)

  def __call__(self, x: dm_env.TimeStep, state):
    x = jax.tree_map(lambda t: t[None, ...], x)
    return self.unroll(x, state)

  def unroll(self, x, state):
    """Unrolls more efficiently than dynamic_unroll."""
    if self._use_resnet:
      torso = AtariDeepTorso()
    else:
      torso = AtariShallowTorso()

    torso_output = hk.BatchApply(torso)(x.observation)
    if self._use_lstm:
      should_reset = jnp.equal(x.step_type, int(dm_env.StepType.FIRST))
      core_input = (torso_output, should_reset)
      core_output, state = hk.dynamic_unroll(self._core, core_input, state)
    else:
      core_output = torso_output
      # state passes through.

    return hk.BatchApply(self._head)(core_output), state

  def _head(self, core_output):
    policy_logits = hk.Linear(self._num_actions)(core_output)
    value = hk.Linear(1)(core_output)
    value = jnp.squeeze(value, axis=-1)
    return NetOutput(policy_logits=policy_logits, value=value)
