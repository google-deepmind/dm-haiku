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
import types
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

from haiku._src import base
from haiku._src import basic
from haiku._src import conv
from haiku._src import initializers
from haiku._src import module
from haiku._src import stateful
import jax
import jax.nn
import jax.numpy as jnp

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.initializers = initializers
hk.Linear = basic.Linear
hk.ConvND = conv.ConvND
hk.get_parameter = base.get_parameter
hk.Module = module.Module
hk.scan = stateful.scan
inside_transform = base.inside_transform
del base, basic, conv, initializers, module


class RNNCore(hk.Module):
  """Base class for RNN cores.

  This class defines the basic functionality that every core should
  implement: :meth:`initial_state`, used to construct an example of the
  core state; and :meth:`__call__` which applies the core parameterized
  by a previous state to an input.

  Cores may be used with :func:`dynamic_unroll` and :func:`static_unroll` to
  iteratively construct an output sequence from the given input sequence.
  """

  @abc.abstractmethod
  def __call__(self, inputs, prev_state) -> Tuple[Any, Any]:
    """Run one step of the RNN.

    Args:
      inputs: An arbitrarily nested structure.
      prev_state: Previous core state.

    Returns:
      A tuple with two elements ``output, next_state``. ``output`` is an
      arbitrarily nested structure. ``next_state`` is the next core state, this
      must be the same shape as ``prev_state``.
    """

  @abc.abstractmethod
  def initial_state(self, batch_size: Optional[int]):
    """Constructs an initial state for this core.

    Args:
      batch_size: Optional int or an integral scalar tensor representing
        batch size. If None, the core may either fail or (experimentally)
        return an initial state without a batch dimension.

    Returns:
      Arbitrarily nested initial state for this core.
    """


def static_unroll(core, input_sequence, initial_state, time_major=True):
  """Performs a static unroll of an RNN.

  An *unroll* corresponds to calling the core on each element of the
  input sequence in a loop, carrying the state through::

      state = initial_state
      for t in range(len(input_sequence)):
         outputs, state = core(input_sequence[t], state)

  A *static* unroll replaces a loop with its body repeated multiple
  times when executed inside :func:`jax.jit`::

      state = initial_state
      outputs0, state = core(input_sequence[0], state)
      outputs1, state = core(input_sequence[1], state)
      outputs2, state = core(input_sequence[2], state)
      ...

  See :func:`dynamic_unroll` for a loop-preserving unroll function.

  Args:
    core: An :class:`RNNCore` to unroll.
    input_sequence: An arbitrarily nested structure of tensors of shape
      ``[T, ...]`` if time-major=True, or ``[B, T, ...]`` if time_major=False,
      where ``T`` is the number of time steps.
    initial_state: An initial state of the given core.
    time_major: If True, inputs are expected time-major, otherwise they are
      expected batch-major.

  Returns:
    A tuple with two elements:
      * **output_sequence** - An arbitrarily nested structure of tensors
        of shape ``[T, ...]`` if time-major, otherwise ``[B, T, ...]``.
      * **final_state** - Core state at time step ``T``.
  """
  output_sequence = []
  time_axis = 0 if time_major else 1
  num_steps = jax.tree_leaves(input_sequence)[0].shape[time_axis]
  state = initial_state
  for t in range(num_steps):
    if time_major:
      inputs = jax.tree_map(lambda x, _t=t: x[_t], input_sequence)
    else:
      inputs = jax.tree_map(lambda x, _t=t: x[:, _t], input_sequence)
    outputs, state = core(inputs, state)
    output_sequence.append(outputs)

  # Stack outputs along the time axis.
  output_sequence = jax.tree_multimap(
      lambda *args: jnp.stack(args, axis=time_axis),
      *output_sequence)
  return output_sequence, state


def _swap_batch_time(inputs):
  """Swaps batch and time axes, assumed to be the first two axes."""
  return jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), inputs)


def dynamic_unroll(core,
                   input_sequence,
                   initial_state,
                   time_major=True,
                   reverse=False):
  """Performs a dynamic unroll of an RNN.

  An *unroll* corresponds to calling the core on each element of the
  input sequence in a loop, carrying the state through::

      state = initial_state
      for t in range(len(input_sequence)):
         outputs, state = core(input_sequence[t], state)

  A *dynamic* unroll preserves the loop structure when executed inside
  :func:`jax.jit`. See :func:`static_unroll` for an unroll function which
  replaces a loop with its body repeated multiple times.

  Args:
    core: An :class:`RNNCore` to unroll.
    input_sequence: An arbitrarily nested structure of tensors of shape
      ``[T, ...]`` if time-major=True, or ``[B, T, ...]`` if time_major=False,
      where ``T`` is the number of time steps.
    initial_state: An initial state of the given core.
    time_major: If True, inputs are expected time-major, otherwise they are
      expected batch-major.
    reverse: If True, inputs are scanned in the reversed order. Equivalent to
      reversing the time dimension in both inputs and outputs. See
      https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html for
      more details.

  Returns:
    A tuple with two elements:
      * **output_sequence** - An arbitrarily nested structure of tensors
        of shape ``[T, ...]`` if time-major, otherwise ``[B, T, ...]``.
      * **final_state** - Core state at time step ``T``.
  """
  scan = hk.scan if inside_transform() else jax.lax.scan
  # Swap the input and output of core.
  def scan_f(prev_state, inputs):
    outputs, next_state = core(inputs, prev_state)
    return next_state, outputs
  # TODO(hamzamerzic): Remove axis swapping once scan supports time axis arg.
  if not time_major:
    input_sequence = _swap_batch_time(input_sequence)
  final_state, output_sequence = scan(
      scan_f,
      initial_state,
      input_sequence,
      reverse=reverse)
  if not time_major:
    output_sequence = _swap_batch_time(output_sequence)
  return output_sequence, final_state


def add_batch(nest, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_map(broadcast, nest)


class VanillaRNN(RNNCore):
  r"""Basic fully-connected RNN core.

  Given :math:`x_t` and the previous hidden state :math:`h_{t-1}` the
  core computes

  .. math::

     h_t = \operatorname{ReLU}(w_i x_t + b_i + w_h h_{t-1} + b_h)

  The output is equal to the new state, :math:`h_t`.
  """

  def __init__(
      self,
      hidden_size: int,
      double_bias: bool = True,
      name: Optional[str] = None
  ):
    """Constructs a vanilla RNN core.

    Args:
      hidden_size: Hidden layer size.
      double_bias: Whether to use a bias in the two linear layers. This changes
        nothing to the learning performance of the cell. However, doubling will
        create two sets of bias parameters rather than one.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.hidden_size = hidden_size
    self.double_bias = double_bias

  def __call__(self, inputs, prev_state):
    input_to_hidden = hk.Linear(self.hidden_size)
    # TODO(b/173771088): Consider changing default to double_bias=False.
    hidden_to_hidden = hk.Linear(self.hidden_size, with_bias=self.double_bias)
    out = jax.nn.relu(input_to_hidden(inputs) + hidden_to_hidden(prev_state))
    return out, out

  def initial_state(self, batch_size: Optional[int]):
    state = jnp.zeros([self.hidden_size])
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class LSTMState(NamedTuple):
  """An LSTM core state consists of hidden and cell vectors.

  Attributes:
    hidden: Hidden state.
    cell: Cell state.
  """
  hidden: jnp.ndarray
  cell: jnp.ndarray


class LSTM(RNNCore):
  r"""Long short-term memory (LSTM) RNN core.

  The implementation is based on :cite:`zaremba2014recurrent`. Given
  :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})` the core
  computes

  .. math::

     \begin{array}{ll}
     i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}

  where :math:`i_t`, :math:`f_t`, :math:`o_t` are input, forget and
  output gate activations, and :math:`g_t` is a vector of cell updates.

  The output is equal to the new hidden, :math:`h_t`.

  Notes:
    Forget gate initialization:
      Following :cite:`jozefowicz2015empirical` we add 1.0 to :math:`b_f`
      after initialization in order to reduce the scale of forgetting in
      the beginning of the training.
  """

  def __init__(self, hidden_size: int, name: Optional[str] = None):
    """Constructs an LSTM.

    Args:
      hidden_size: Hidden layer size.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.hidden_size = hidden_size

  def __call__(
      self,
      inputs: jnp.ndarray,
      prev_state: LSTMState,
  ) -> Tuple[jnp.ndarray, LSTMState]:
    if len(inputs.shape) > 2 or not inputs.shape:
      raise ValueError("LSTM input must be rank-1 or rank-2.")
    x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)
    gated = hk.Linear(4 * self.hidden_size)(x_and_h)
    # TODO(slebedev): Consider aligning the order of gates with Sonnet.
    # i = input, g = cell_gate, f = forget_gate, o = output_gate
    i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
    f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
    c = f * prev_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)
    return h, LSTMState(h, c)

  def initial_state(self, batch_size: Optional[int]) -> LSTMState:
    state = LSTMState(hidden=jnp.zeros([self.hidden_size]),
                      cell=jnp.zeros([self.hidden_size]))
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class ConvNDLSTM(RNNCore):
  r"""``num_spatial_dims``-D convolutional LSTM.

  The implementation is based on :cite:`xingjian2015convolutional`.
  Given :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})`
  the core computes

  .. math::

     \begin{array}{ll}
     i_t = \sigma(W_{ii} * x_t + W_{hi} * h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} * x_t + W_{hf} * h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} * x_t + W_{hg} * h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} * x_t + W_{ho} * h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}

  where :math:`*` denotes the convolution operator; :math:`i_t`,
  :math:`f_t`, :math:`o_t` are input, forget and output gate activations,
  and :math:`g_t` is a vector of cell updates.

  The output is equal to the new hidden state, :math:`h_t`.

  Notes:
    Forget gate initialization:
      Following :cite:`jozefowicz2015empirical` we add 1.0 to :math:`b_f`
      after initialization in order to reduce the scale of forgetting in
      the beginning of the training.
  """

  def __init__(
      self,
      num_spatial_dims: int,
      input_shape: Sequence[int],
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      name: Optional[str] = None,
  ):
    """Constructs a convolutional LSTM.

    Args:
      num_spatial_dims: Number of spatial dimensions of the input.
      input_shape: Shape of the inputs excluding batch size.
      output_channels: Number of output channels.
      kernel_shape: Sequence of kernel sizes (of length ``num_spatial_dims``),
        or an int. ``kernel_shape`` will be expanded to define a kernel size in
        all dimensions.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.num_spatial_dims = num_spatial_dims
    self.input_shape = tuple(input_shape)
    self.output_channels = output_channels
    self.kernel_shape = kernel_shape

  def __call__(
      self,
      inputs,
      state: LSTMState,
  ) -> Tuple[jnp.ndarray, LSTMState]:
    input_to_hidden = hk.ConvND(
        num_spatial_dims=self.num_spatial_dims,
        output_channels=4 * self.output_channels,
        kernel_shape=self.kernel_shape,
        name="input_to_hidden")

    hidden_to_hidden = hk.ConvND(
        num_spatial_dims=self.num_spatial_dims,
        output_channels=4 * self.output_channels,
        kernel_shape=self.kernel_shape,
        name="hidden_to_hidden")

    gates = input_to_hidden(inputs) + hidden_to_hidden(state.hidden)
    i, g, f, o = jnp.split(gates, indices_or_sections=4, axis=-1)

    f = jax.nn.sigmoid(f + 1)
    c = f * state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)
    return h, LSTMState(h, c)

  def initial_state(self, batch_size: Optional[int]) -> LSTMState:
    shape = self.input_shape + (self.output_channels,)
    state = LSTMState(jnp.zeros(shape), jnp.zeros(shape))
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class Conv1DLSTM(ConvNDLSTM):  # pylint: disable=empty-docstring
  __doc__ = ConvNDLSTM.__doc__.replace("``num_spatial_dims``", "1")

  def __init__(
      self,
      input_shape: Sequence[int],
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      name: Optional[str] = None,
  ):
    """Constructs a 1-D convolutional LSTM.

    Args:
      input_shape: Shape of the inputs excluding batch size.
      output_channels: Number of output channels.
      kernel_shape: Sequence of kernel sizes (of length 1), or an int.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      name: Name of the module.
    """
    super().__init__(
        num_spatial_dims=1,
        input_shape=input_shape,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        name=name)


class Conv2DLSTM(ConvNDLSTM):  # pylint: disable=empty-docstring
  __doc__ = ConvNDLSTM.__doc__.replace("``num_spatial_dims``", "2")

  def __init__(
      self,
      input_shape: Sequence[int],
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      name: Optional[str] = None,
  ):
    """Constructs a 2-D convolutional LSTM.

    Args:
      input_shape: Shape of the inputs excluding batch size.
      output_channels: Number of output channels.
      kernel_shape: Sequence of kernel sizes (of length 2), or an int.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      name: Name of the module.
    """
    super().__init__(
        num_spatial_dims=2,
        input_shape=input_shape,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        name=name)


class Conv3DLSTM(ConvNDLSTM):  # pylint: disable=empty-docstring
  __doc__ = ConvNDLSTM.__doc__.replace("``num_spatial_dims``", "3")

  def __init__(
      self,
      input_shape: Sequence[int],
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      name: Optional[str] = None,
  ):
    """Constructs a 3-D convolutional LSTM.

    Args:
      input_shape: Shape of the inputs excluding batch size.
      output_channels: Number of output channels.
      kernel_shape: Sequence of kernel sizes (of length 3), or an int.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      name: Name of the module.
    """
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

  The output is equal to the new hidden state, :math:`h_t`.

  Warning: Backwards compatibility of GRU weights is currently unsupported.

  TODO(tycai): Make policy decision/benchmark performance for GRU variants.
  """

  def __init__(
      self,
      hidden_size: int,
      w_i_init: Optional[hk.initializers.Initializer] = None,
      w_h_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.hidden_size = hidden_size
    self.w_i_init = w_i_init or hk.initializers.VarianceScaling()
    self.w_h_init = w_h_init or hk.initializers.VarianceScaling()
    self.b_init = b_init or jnp.zeros

  def __call__(self, inputs, state):
    if inputs.ndim not in (1, 2):
      raise ValueError("GRU input must be rank-1 or rank-2.")

    input_size = inputs.shape[-1]
    hidden_size = self.hidden_size
    w_i = hk.get_parameter("w_i", [input_size, 3 * hidden_size], inputs.dtype,
                           init=self.w_i_init)
    w_h = hk.get_parameter("w_h", [hidden_size, 3 * hidden_size], inputs.dtype,
                           init=self.w_h_init)
    b = hk.get_parameter("b", [3 * hidden_size], inputs.dtype, init=self.b_init)
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

  def initial_state(self, batch_size: Optional[int]):
    state = jnp.zeros([self.hidden_size])
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


class IdentityCore(RNNCore):
  """A recurrent core that forwards the inputs and an empty state.

  This is commonly used when switching between recurrent and feedforward
  versions of a model while preserving the same interface.
  """

  def __call__(self, inputs, state):
    return inputs, state

  def initial_state(self, batch_size: Optional[int]):
    return ()


def _validate_and_conform(should_reset, state):
  """Ensures that should_reset is compatible with state."""
  if should_reset.shape == state.shape[:should_reset.ndim]:
    broadcast_shape = should_reset.shape + (1,)*(state.ndim - should_reset.ndim)
    return jnp.reshape(should_reset, broadcast_shape)

  raise ValueError(
      "should_reset signal shape {} is not compatible with "
      "state shape {}".format(should_reset.shape, state.shape))


class ResetCore(RNNCore):
  """A wrapper for managing state resets during unrolls.

  When unrolling an :class:`RNNCore` on a batch of inputs sequences it may be
  necessary to reset the core's state at different timesteps for different
  elements of the batch. The :class:`ResetCore` class enables this by taking a
  batch of ``should_reset`` booleans in addition to the batch of inputs, and
  conditionally resetting the core's state for individual elements of the batch.
  You may also reset individual entries of the state by passing a
  ``should_reset`` nest compatible with the state structure.
  """

  def __init__(self, core: RNNCore, name: Optional[str] = None):
    super().__init__(name=name)
    self.core = core

  def __call__(self, inputs, state):
    """Run one step of the wrapped core, handling state reset.

    Args:
      inputs: Tuple with two elements, ``inputs, should_reset``, where
        ``should_reset`` is the signal used to reset the wrapped core's state.
        ``should_reset`` can be either tensor or nest. If nest, ``should_reset``
        must match the state structure, and its components' shapes must be
        prefixes of the corresponding entries tensors' shapes in the state nest.
        If tensor, supported shapes are all commom shape prefixes of the state
        component tensors, e.g. ``[batch_size]``.
      state: Previous wrapped core state.

    Returns:
      Tuple of the wrapped core's ``output, next_state``.
    """
    inputs, should_reset = inputs
    if jax.treedef_is_leaf(jax.tree_structure(should_reset)):
      # Equivalent to not tree.is_nested, but with support for Jax extensible
      # pytrees.
      should_reset = jax.tree_map(lambda _: should_reset, state)

    # We now need to manually pad 'on the right' to ensure broadcasting operates
    # correctly.
    # Automatic broadcasting would in fact implicitly pad 'on the left',
    # resulting in the signal to trigger resets for parts of the state
    # across batch entries. For example:
    #
    # import jax
    # import jax.numpy as jnp
    #
    # shape = (2, 2, 2)
    # x = jnp.zeros(shape)
    # y = jnp.ones(shape)
    # should_reset = jnp.array([False, True])
    # v = jnp.where(should_reset, x, y)
    # for batch_entry in range(shape[0]):
    #   print("batch_entry {}:\n".format(batch_entry), v[batch_entry])
    #
    # >> batch_entry 0:
    # >>  [[1. 0.]
    # >>  [1. 0.]]
    # >> batch_entry 1:
    # >>  [[1. 0.]
    # >>  [1. 0.]]
    #
    # Note how manually padding the should_reset tensor yields the desired
    # behavior.
    #
    # import jax
    # import jax.numpy as jnp
    #
    # shape = (2, 2, 2)
    # x = jnp.zeros(shape)
    # y = jnp.ones(shape)
    # should_reset = jnp.array([False, True])
    # dims_to_add = x.ndim - should_reset.ndim
    # should_reset = should_reset.reshape(should_reset.shape + (1,)*dims_to_add)
    # v = jnp.where(should_reset, x, y)
    # for batch_entry in range(shape[0]):
    #   print("batch_entry {}:\n".format(batch_entry), v[batch_entry])
    #
    # >> batch_entry 0:
    # >>  [[1. 1.]
    # >>  [1. 1.]]
    # >> batch_entry 1:
    # >>  [[0. 0.]
    # >>  [0. 0.]]
    should_reset = jax.tree_multimap(_validate_and_conform, should_reset, state)
    if self._is_batched(state):
      batch_size = jax.tree_leaves(inputs)[0].shape[0]
    else:
      batch_size = None
    initial_state = jax.tree_multimap(
        lambda s, i: i.astype(s.dtype), state, self.initial_state(batch_size))
    state = jax.tree_multimap(jnp.where, should_reset, initial_state, state)
    return self.core(inputs, state)

  def initial_state(self, batch_size: Optional[int]):
    return self.core.initial_state(batch_size)

  def _is_batched(self, state):
    state = jax.tree_leaves(state)
    if not state:  # Empty state is treated as unbatched.
      return False
    batched = jax.tree_leaves(self.initial_state(batch_size=1))
    return all(b.shape[1:] == s.shape[1:] for b, s in zip(batched, state))


class _DeepRNN(RNNCore):
  """Underlying implementation of DeepRNN with skip connections."""

  def __init__(
      self,
      layers: Sequence[Any],
      skip_connections: bool,
      name: Optional[str] = None
  ):
    super().__init__(name=name)
    self.layers = layers
    self.skip_connections = skip_connections

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
    for idx, layer in enumerate(self.layers):
      if self.skip_connections and idx > 0:
        current_inputs = jax.tree_multimap(concat, inputs, current_inputs)

      if isinstance(layer, RNNCore):
        current_inputs, next_state = layer(current_inputs, state[state_idx])
        outputs.append(current_inputs)
        next_states.append(next_state)
        state_idx += 1
      else:
        current_inputs = layer(current_inputs)

    if self.skip_connections:
      out = jax.tree_multimap(concat, *outputs)
    else:
      out = current_inputs

    return out, tuple(next_states)

  def initial_state(self, batch_size: Optional[int]):
    return tuple(
        layer.initial_state(batch_size)
        for layer in self.layers
        if isinstance(layer, RNNCore))


class DeepRNN(_DeepRNN):
  r"""Wraps a sequence of cores and callables as a single core.

      >>> deep_rnn = hk.DeepRNN([
      ...     hk.LSTM(hidden_size=4),
      ...     jax.nn.relu,
      ...     hk.LSTM(hidden_size=2),
      ... ])

  The state of a :class:`DeepRNN` is a tuple with one element per
  :class:`RNNCore`. If no layers are :class:`RNNCore`\ s, the state is an empty
  tuple.
  """

  def __init__(self, layers: Sequence[Any], name: Optional[str] = None):
    super().__init__(layers, skip_connections=False, name=name)


def deep_rnn_with_skip_connections(layers: Sequence[RNNCore],
                                   name: Optional[str] = None) -> RNNCore:
  r"""Constructs a :class:`DeepRNN` with skip connections.

  Skip connections alter the dependency structure within a :class:`DeepRNN`.
  Specifically, input to the i-th layer (i > 0) is given by a
  concatenation of the core's inputs and the outputs of the (i-1)-th layer.

  The output of the :class:`DeepRNN` is the concatenation of the outputs of all
  cores.

  .. code-block:: python

     outputs0, ... = layers[0](inputs, ...)
     outputs1, ... = layers[1](tf.concat([inputs, outputs0], axis=-1], ...)
     outputs2, ... = layers[2](tf.concat([inputs, outputs1], axis=-1], ...)
     ...

  Args:
    layers: List of :class:`RNNCore`\ s.
    name: Name of the module.

  Returns:
    A :class:`_DeepRNN` with skip connections.

  Raises:
    ValueError: If any of the layers is not an :class:`RNNCore`.
  """
  return _DeepRNN(layers, skip_connections=True, name=name)
