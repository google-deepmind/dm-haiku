# Lint as: python3
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
"""Haiku recurrent core."""

import abc
from haiku._src import base
from haiku._src import basic
from haiku._src import conv
from haiku._src import initializers
from haiku._src import module
import jax
import jax.nn
import jax.numpy as jnp


def add_batch(nest, batch_size):
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_map(broadcast, nest)


def static_unroll(core, inputs, state):
  """Unroll core over inputs, starting from state."""
  outs = []
  num_steps = jax.tree_leaves(inputs)[0].shape[0]
  for t in range(num_steps):
    next_input = jax.tree_map(lambda x, t=t: x[t], inputs)
    out, state = core(next_input, state)
    outs.append(out)
  return jnp.stack(outs), state


def dynamic_unroll(core, inputs, state):
  """Unroll core over inputs, starting from state."""
  # Swap the input and output of core.
  def scan_f(prev_state, next_input):
    out, next_state = core(next_input, prev_state)
    return next_state, out
  state, outs = jax.lax.scan(scan_f, state, inputs)
  return outs, state


class RNNCore(module.Module, metaclass=abc.ABCMeta):
  """Base class for RNN cores.

  Cores can be dynamically unrolled with jax.lax.scan().
  """

  @abc.abstractmethod
  def __call__(self, inputs, state):
    """Run one step of the RNN.

    Args:
      inputs: Arbitrary nest of inputs.
      state: Previous core state.
    Returns:
      Tuple of (output, next_state).
    """

  @abc.abstractmethod
  def initial_state(self, batch_size):
    """Construct an initial state for the core.

    Args:
      batch_size: Specifies the batch size of the initial state. Cores may
        experimentally support returning an initial state without a batch
        dimension if batch_size is None.
    """


class VanillaRNN(RNNCore):
  """Vanilla RNN."""

  def __init__(self, hidden_size, name=None):
    super().__init__(name=name)
    self.hidden_size = hidden_size

  def __call__(self, inputs, state):
    in2h = basic.Linear(self.hidden_size)(inputs)
    h2h = basic.Linear(self.hidden_size)(state)
    output = jax.nn.relu(in2h + h2h)
    new_h = output
    return output, new_h

  def initial_state(self, batch_size):
    state = jnp.zeros([self.hidden_size])
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class LSTM(RNNCore):
  """LSTM.

  Following https://dl.acm.org/doi/10.5555/3045118.304536, we add a constant
  bias of 1 to the forget gate in order to reduce the scale of forgetting in
  the beginning of the training.
  """

  def __init__(self, hidden_size, name=None):
    super().__init__(name=name)
    self.hidden_size = hidden_size

  def __call__(self, inputs, state):
    if len(inputs.shape) > 2 or not inputs.shape:
      raise ValueError("LSTM input must be rank-1 or rank-2.")
    prev_h, prev_c = state
    x_and_h = jnp.concatenate([inputs, prev_h], axis=-1)
    gated = basic.Linear(4 * self.hidden_size)(x_and_h)
    # i = input, g = cell_gate, f = forget_gate, o = output_gate
    i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
    f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
    c = f * prev_c + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)
    return h, (h, c)

  def initial_state(self, batch_size):
    state = (jnp.zeros([self.hidden_size]), jnp.zeros([self.hidden_size]))
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class ConvNDLSTM(RNNCore):
  """N-D convolutional LSTM.

  The implementation is based on https://arxiv.org/abs/1506.04214.

  Following https://dl.acm.org/doi/10.5555/3045118.304536, we add a constant
  bias of 1 to the forget gate in order to reduce the scale of forgetting in
  the beginning of the training.
  """

  def __init__(self,
               num_spatial_dims,
               input_shape,
               output_channels,
               kernel_shape,
               name=None):
    super().__init__(name=name)
    self._num_spatial_dims = num_spatial_dims
    self.input_shape = input_shape
    self.output_channels = output_channels
    self.kernel_shape = kernel_shape

  def __call__(self, inputs, state):
    prev_h, prev_c = state

    gates = conv.ConvND(
        num_spatial_dims=self._num_spatial_dims,
        output_channels=4*self.output_channels,
        kernel_shape=self.kernel_shape,
        name="input_to_hidden")(
            inputs)
    gates += conv.ConvND(
        num_spatial_dims=self._num_spatial_dims,
        output_channels=4*self.output_channels,
        kernel_shape=self.kernel_shape,
        name="hidden_to_hidden")(
            prev_h)
    i, g, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)

    f = jax.nn.sigmoid(f + 1)
    c = f * prev_c + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)
    return h, (h, c)

  def initial_state(self, batch_size):
    state = (jnp.zeros(list(self.input_shape) + [self.output_channels]),
             jnp.zeros(list(self.input_shape) + [self.output_channels]))
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class Conv1DLSTM(ConvNDLSTM):
  """Conv1D module."""

  def __init__(self, input_shape, output_channels, kernel_shape, name=None):
    """Initializes a Conv1DLSTM module. See superclass for documentation."""
    super().__init__(
        num_spatial_dims=1,
        input_shape=input_shape,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        name=name)


class Conv2DLSTM(ConvNDLSTM):
  """Conv2D module."""

  def __init__(self, input_shape, output_channels, kernel_shape, name=None):
    """Initializes a Conv2DLSTM module. See superclass for documentation."""
    super().__init__(
        num_spatial_dims=2,
        input_shape=input_shape,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        name=name)


class Conv3DLSTM(ConvNDLSTM):
  """Conv3D module."""

  def __init__(self, input_shape, output_channels, kernel_shape, name=None):
    """Initializes a Conv3DLSTM module. See superclass for documentation."""
    super().__init__(
        num_spatial_dims=3,
        input_shape=input_shape,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        name=name)


class GRU(RNNCore):
  r"""Gated Recurrent Unit.

  The implementation is based on: https://arxiv.org/pdf/1412.3555v1.pdf with
  biases.

  Given :math:`x_t` and the previous state :math:`h_{t-1}` the core computes

  .. math::

     \begin{array}{ll}
     z_t &= \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z) \\
     r_t &= \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r) \\
     a_t &= \tanh(W_{ia} x_t + W_{ha} (r_t \bigodot h_{t-1}) + b_a) \\
     h_t &= (1 - z_t) \bigodot h_{t-1} + z_t \bigodot a_t
     \end{array}

  where :math:`z_t` and :math:`r_t` are reset and update gates.

  Warning: Backwards compatibility of GRU weights is currently unsupported.

  TODO(tycai): Make policy decision/benchmark performance for GRU variants.
  """

  def __init__(self,
               hidden_size,
               w_i_init: base.Initializer = None,
               w_h_init: base.Initializer = None,
               b_init: base.Initializer = None,
               name=None):
    super().__init__(name=name)
    self.hidden_size = hidden_size
    self._w_i_init = w_i_init or initializers.VarianceScaling()
    self._w_h_init = w_h_init or initializers.VarianceScaling()
    self._b_init = b_init or jnp.zeros

  def __call__(self, inputs, state):
    if len(inputs.shape) > 2 or not inputs.shape:
      raise ValueError("GRU input must be rank-1 or rank-2.")

    input_size = inputs.shape[-1]
    hidden_size = self.hidden_size
    w_i = base.get_parameter(
        name="w_i", shape=[input_size, 3 * hidden_size], init=self._w_i_init)
    w_h = base.get_parameter(
        name="w_h", shape=[hidden_size, 3 * hidden_size], init=self._w_h_init)
    b = base.get_parameter(
        name="b",
        shape=[3 * hidden_size],
        dtype=inputs.dtype,
        init=self._b_init)
    w_h_z, w_h_a = jnp.split(w_h, indices_or_sections=[2 * hidden_size], axis=1)
    b_z, b_a = jnp.split(b, indices_or_sections=[2 * hidden_size], axis=0)

    gates_x = jnp.matmul(inputs, w_i)
    zr_x, a_x = jnp.split(
        gates_x, indices_or_sections=[2 * hidden_size], axis=-1)
    zr_h = jnp.matmul(state, w_h_z)
    zr = zr_x + zr_h + jnp.broadcast_to(b_z, zr_h.shape)
    z, r = jnp.split(jax.nn.sigmoid(zr), indices_or_sections=2, axis=-1)

    a_h = jnp.matmul(r * state, w_h_a)
    a = jnp.tanh(a_x + a_h + jnp.broadcast_to(b_a, a_h.shape))

    next_state = (1 - z) * state + z * a
    return next_state, next_state

  def initial_state(self, batch_size):
    state = jnp.zeros([self.hidden_size])
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class ResetCore(RNNCore):
  """A wrapper for managing state resets during unrolls.

  When unrolling an `RNNCore` on a batch of inputs sequences it may be necessary
  to reset the core's state at different timesteps for different elements of the
  batch. The `ResetCore` class enables this by taking a batch of `should_reset`
  booleans in addition to the batch of inputs, and conditionally resetting the
  core's state for individual elements of the batch.
  """

  def __init__(self, core, name=None):
    super().__init__(name=name)
    self._core = core

  def __call__(self, inputs, state):
    inputs, should_reset = inputs
    # Protect against an easy, invisible error class. This should be jitted out.
    # >>> np.where(np.asarray([False, True]), np.zeros([2,2]), np.ones([2,2]))
    # ... array([[1., 0.], [1., 0.]])
    # Using a should_reset of rank R - 1 could result in one example
    # affecting another.
    for x in jax.tree_leaves(state):
      if len(x.shape) - 1 != len(should_reset.shape):
        raise ValueError("should_reset must have rank-1 of state.")
    should_reset = jnp.expand_dims(should_reset, axis=-1)

    batch_size = jax.tree_leaves(inputs)[0].shape[0]
    initial_state = jax.tree_multimap(lambda s, i: i.astype(s.dtype),
                                      state, self.initial_state(batch_size))
    state = jax.tree_multimap(lambda i, s: jnp.where(should_reset, i, s),
                              initial_state, state)
    return self._core(inputs, state)

  def initial_state(self, batch_size):
    return self._core.initial_state(batch_size)


class _DeepRNN(RNNCore):
  """Underlying implementation of DeepRNN with skip connections."""

  def __init__(self, layers, skip_connections, name=None):
    super().__init__(name=name)
    self._layers = layers
    self._skip_connections = skip_connections

    if skip_connections:
      for layer in layers:
        if not isinstance(layer, RNNCore):
          raise ValueError("skip_connections requires for all layers to be "
                           "`hk.RNNCore`s. Layers is: {}".format(layers))

  def __call__(self, inputs, state):
    current_inputs = inputs
    next_states = []
    outputs = []
    state_idx = 0
    concat = lambda *args: jnp.concatenate(args, axis=-1)
    for idx, layer in enumerate(self._layers):
      if self._skip_connections and idx > 0:
        current_inputs = jax.tree_multimap(concat, inputs, current_inputs)

      if isinstance(layer, RNNCore):
        current_inputs, next_state = layer(current_inputs, state[state_idx])
        outputs.append(current_inputs)
        next_states.append(next_state)
        state_idx += 1
      else:
        current_inputs = layer(current_inputs)

    if self._skip_connections:
      output = jax.tree_multimap(concat, *outputs)
    else:
      output = current_inputs

    return output, tuple(next_states)

  def initial_state(self, batch_size):
    return tuple(
        layer.initial_state(batch_size)
        for layer in self._layers
        if isinstance(layer, RNNCore))


class DeepRNN(_DeepRNN):
  """Wraps a sequence of cores and callables as a single core.

      >>> deep_rnn = hk.DeepRNN([
      ...     hk.LSTM(hidden_size=4),
      ...     jax.nn.relu,
      ...     hk.LSTM(hidden_size=2),
      ... ])

  The state of a `DeepRNN` is a tuple with one element per `RNNCore`.
  If no layers are `RNNCore`s, the state is an empty tuple.
  """

  def __init__(self, layers, name=None):
    super().__init__(layers, skip_connections=False, name=name)


def deep_rnn_with_skip_connections(layers, name=None):
  """Constructs a DeepRNN with skip connections.

  Skip connections alter the dependency structure within a `DeepRNN`.
  Specifically, input to the i-th layer (i > 0) is given by a
  concatenation of the core's inputs and the outputs of the (i-1)-th layer.

  The output of the `DeepRNN` is the concatenation of the outputs of all cores.

  .. code-block:: python

     outputs0, ... = layers[0](inputs, ...)
     outputs1, ... = layers[1](tf.concat([inputs, outputs0], axis=-1], ...)
     outputs2, ... = layers[2](tf.concat([inputs, outputs1], axis=-1], ...)
     ...

  Args:
    layers: List of `RNNCore`s.
    name: Name of the module.

  Returns:
    A `_DeepRNN` with skip connections.

  Raises:
    ValueError: If any of the layers is not an `RNNCore`.
  """
  return _DeepRNN(layers, skip_connections=True, name=name)
