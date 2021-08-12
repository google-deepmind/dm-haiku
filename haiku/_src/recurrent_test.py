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
"""Tests for haiku._src.recurrent."""
import itertools as it
from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import basic
from haiku._src import recurrent
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import tree


class DuplicateCore(recurrent.RNNCore):
  """A wrapper which duplicates the outputs of the wrapped :class:`RNNCore`."""

  def __init__(self, base_core: recurrent.RNNCore):
    super().__init__()
    self.base_core = base_core

  def __call__(self, inputs, prev_state):
    """See base class."""
    outputs, next_state = self.base_core(inputs, prev_state)
    return [outputs, outputs], next_state

  def initial_state(self, batch_size):
    """See base class."""
    return self.base_core.initial_state(batch_size)


def make_sequence(shape):
  # Skips 0 for meaningful multiplicative interactions.
  return np.arange(1, np.product(shape) + 1, dtype=np.float32).reshape(shape)


class RecurrentTest(parameterized.TestCase):

  UNROLLS = (recurrent.dynamic_unroll, recurrent.static_unroll)
  CORES = (recurrent.VanillaRNN, recurrent.LSTM, recurrent.GRU)

  def test_add_batch(self):
    sample_tree = dict(
        a=[jnp.zeros([]), jnp.zeros([1])],
        b=jnp.zeros([1, 1]),
    )
    batch_size = 2
    out = recurrent.add_batch(sample_tree, batch_size)
    tree.assert_same_structure(sample_tree, out)
    flat_in = tree.flatten(sample_tree)
    flat_out = tree.flatten(out)
    for in_array, out_array in zip(flat_in, flat_out):
      self.assertEqual(out_array.shape[0], batch_size)
      self.assertEqual(out_array.shape[1:], in_array.shape)

  # These two tests assume that the core takes argument hidden_size, and the
  # output is a single tensor with the same size as hidden_size.
  # They should be generalized when new cores are added.
  @parameterized.parameters(*it.product(UNROLLS, CORES))
  @test_utils.transform_and_run
  def test_core_unroll_unbatched(self, unroll, core_cls):
    seqs = make_sequence([8, 1])  # [T, F]
    core = core_cls(hidden_size=4)
    out, _ = unroll(core, seqs, core.initial_state(batch_size=None))
    self.assertEqual(out.shape, (8, 4))

  @parameterized.parameters(*it.product(UNROLLS, CORES))
  @test_utils.transform_and_run
  def test_core_unroll_batched(self, unroll, core_cls):
    seqs = make_sequence([4, 8, 1])  # [T, B, F]
    core = core_cls(hidden_size=4)
    batch_size = seqs.shape[1]
    out, _ = unroll(core, seqs, core.initial_state(batch_size))
    self.assertEqual(out.shape, (4, 8, 4))

  @parameterized.parameters(*UNROLLS)
  @test_utils.transform_and_run
  def test_core_unroll_nested(self, unroll):
    seqs = make_sequence([4, 8, 1])
    batch_size = seqs.shape[1]
    core = DuplicateCore(recurrent.VanillaRNN(hidden_size=4))
    outs, _ = unroll(core, seqs, core.initial_state(batch_size))
    self.assertLen(outs, 2)
    for out in outs:
      self.assertEqual(out.shape, (4, 8, 4))

  @parameterized.parameters(*UNROLLS)
  def test_unroll_outside_transform(self, unroll):
    core = lambda x, s: (x + 1, s + 1)
    seqs = jnp.arange(8)
    outs, state = unroll(core, seqs, 0)
    np.testing.assert_allclose(outs, jnp.arange(9)[1:])
    np.testing.assert_allclose(state, 8)


class VanillaRNNTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_double_bias_length_parameters(self):
    double_bias = recurrent.VanillaRNN(1, double_bias=True)
    double_bias(jnp.zeros([1]), double_bias.initial_state(None))
    double_bias_params = jax.tree_leaves(double_bias.params_dict())

    vanilla = recurrent.VanillaRNN(1, double_bias=False)
    vanilla(jnp.zeros([1]), vanilla.initial_state(None))
    vanilla_params = jax.tree_leaves(vanilla.params_dict())

    self.assertLen(double_bias_params, len(vanilla_params) + 1)


class LSTMTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_lstm_raises(self):
    core = recurrent.LSTM(4)
    with self.assertRaisesRegex(ValueError, "rank-1 or rank-2"):
      core(jnp.zeros([]), core.initial_state(None))

    with self.assertRaisesRegex(ValueError, "rank-1 or rank-2"):
      expanded_state = tree.map_structure(lambda x: jnp.expand_dims(x, 0),
                                          core.initial_state(1))
      core(jnp.zeros([1, 1, 1]), expanded_state)


class ConvLSTMTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3)
  @test_utils.transform_and_run
  def test_connect_conv_same(self, n):
    batch_size = 2
    input_shape = (16,) * n
    input_shape_b = (batch_size,) + input_shape + (4,)

    data = jnp.zeros(input_shape_b)
    core = recurrent.ConvNDLSTM(
        n, input_shape=input_shape, output_channels=3, kernel_shape=3)
    state = core.initial_state(batch_size=batch_size)
    out, state = core(data, state)
    expected_output_shape = (batch_size,) + input_shape + (3,)
    self.assertEqual(out.shape, expected_output_shape)
    self.assertEqual(state[0].shape, expected_output_shape)
    self.assertEqual(state[1].shape, expected_output_shape)


class GRUTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_gru_raises(self):
    core = recurrent.GRU(4)
    with self.assertRaisesRegex(ValueError, "rank-1 or rank-2"):
      core(jnp.zeros([]), core.initial_state(None))


class _DummyCore(recurrent.RNNCore):

  def __init__(self, state, name="dummy"):
    super().__init__(name=name)
    self._state = state

  def __call__(self, inputs, prev_state):
    return inputs, prev_state

  def initial_state(self, batch_size):
    return jax.tree_map(jnp.zeros_like, self._state)


class _IncrementByOneCore(recurrent.RNNCore):

  def __init__(self, state_size=4, name=None):
    super().__init__(name=name)
    self._state_size = state_size

  def __call__(self, inputs, prev_state):
    del inputs
    state = prev_state + 1.
    return state, state

  def initial_state(self, batch_size):
    if batch_size is not None:
      return jnp.zeros((batch_size, self._state_size))
    return jnp.zeros(self._state_size)


class _BatchedOnlyCore(recurrent.RNNCore):

  def __call__(self, inputs, prev_state):
    return inputs, prev_state

  def initial_state(self, batch_size):
    assert batch_size is not None
    return jnp.zeros([batch_size])


def static_unroll_with_states(core, inputs, state):
  outs = []
  states = []
  steps = tree.flatten(inputs)[0].shape[0]
  for i in range(steps):
    step_input = tree.map_structure(lambda x: x[i], inputs)  # pylint: disable=cell-var-from-loop
    out, state = core(step_input, state)
    outs.append(out)
    states.append(state)

  outs = jnp.stack(outs, axis=0)
  states = tree.map_structure(lambda *a: jnp.stack(a, axis=0), *states)
  return outs, states


class ResetCoreTest(parameterized.TestCase):

  @parameterized.parameters(recurrent.dynamic_unroll, recurrent.static_unroll)
  def test_resetting(self, unroll):
    def net(seqs, should_reset):
      # seqs is [T, B, F].
      core = recurrent.LSTM(hidden_size=4)
      reset_core = recurrent.ResetCore(core)
      batch_size = seqs.shape[1]

      # Statically unroll, collecting states.
      core_outs, core_states = static_unroll_with_states(
          core, seqs, core.initial_state(batch_size))
      reset_outs, reset_states = static_unroll_with_states(
          reset_core, (seqs, should_reset),
          reset_core.initial_state(batch_size))

      # Unroll without access to intermediate states.
      dynamic_core_outs, dynamic_core_state = unroll(
          core, seqs, core.initial_state(batch_size))
      dynamic_reset_outs, dynamic_reset_state = unroll(
          reset_core, (seqs, should_reset),
          reset_core.initial_state(batch_size))

      return dict(
          core_outs=core_outs,
          core_states=core_states,
          reset_outs=reset_outs,
          reset_states=reset_states,
          dynamic_core_outs=dynamic_core_outs,
          dynamic_core_state=dynamic_core_state,
          dynamic_reset_outs=dynamic_reset_outs,
          dynamic_reset_state=dynamic_reset_state,
      )

    batch_size = 4
    # Reset one batch element on the second step.
    resets = [[False] * batch_size, [True] + [False] * (batch_size - 1)]
    resets = np.asarray(resets)

    # Each sequence is the same input twice.
    seqs = make_sequence([batch_size, 1])
    seqs = np.stack([seqs, seqs], axis=0)

    init_fn, apply_fn = transform.transform(net)
    params = init_fn(jax.random.PRNGKey(428), seqs, resets)
    result = apply_fn(params, None, seqs, resets)

    # Verify dynamic and static unroll gave same outs and final states.
    np.testing.assert_allclose(
        result["core_outs"], result["dynamic_core_outs"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        result["reset_outs"],
        result["dynamic_reset_outs"],
        rtol=1e-6,
        atol=1e-6)
    for s, d in zip(result["core_states"], result["dynamic_core_state"]):
      np.testing.assert_allclose(s[-1], d, rtol=1e-6, atol=1e-6)
    for s, d in zip(result["reset_states"], result["dynamic_reset_state"]):
      np.testing.assert_allclose(s[-1], d, rtol=1e-6, atol=1e-6)

    # Now, test resetting behavior on static outputs.
    core_outs = result["core_outs"]
    core_states = result["core_states"]
    reset_outs = result["reset_outs"]
    reset_states = result["reset_states"]

    # If no reset occurred, the reset core should do nothing.
    np.testing.assert_allclose(
        core_outs[0], reset_outs[0], rtol=1e-6, atol=1e-6)
    for cs, rs in zip(core_states, reset_states):
      np.testing.assert_allclose(cs[0], rs[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        core_outs[1, 1:], reset_outs[1, 1:], rtol=1e-6, atol=1e-6)
    for cs, rs in zip(core_states, reset_states):
      np.testing.assert_allclose(cs[1, 1:], rs[1, 1:], rtol=1e-6, atol=1e-6)

    # Check that the reset occurred where specified.
    np.testing.assert_allclose(
        core_outs[0, 0], reset_outs[1, 0], rtol=1e-6, atol=1e-6)
    for cs, rs in zip(core_states, reset_states):
      np.testing.assert_allclose(cs[0, 0], rs[1, 0], rtol=1e-6, atol=1e-6)

  @parameterized.parameters(recurrent.dynamic_unroll, recurrent.static_unroll)
  @test_utils.transform_and_run
  def test_unbatched(self, unroll):
    reset_time = 2
    seq_len = 5
    state_size = 4

    core = recurrent.ResetCore(_IncrementByOneCore(state_size=state_size))
    inputs = jnp.arange(0, seq_len)
    batch_size = None  # Unbatched.
    should_reset = inputs == reset_time
    initial_state = core.initial_state(batch_size)
    result, _ = unroll(core, (inputs, should_reset), initial_state)

    expected_result = np.array([  # seq_len x state_size
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],  # reset_time = 2.
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0]
    ])
    np.testing.assert_allclose(result, expected_result, rtol=1e-6, atol=1e-6)

  @parameterized.parameters(None, 3)
  @test_utils.transform_and_run
  def test_reversed_dynamic_unroll(self, batch_size):
    reset_time = 2
    seq_len = 7
    state_size = 4

    core = recurrent.ResetCore(_IncrementByOneCore(state_size=state_size))
    initial_state = core.initial_state(batch_size)

    inputs = jnp.arange(0, seq_len)  # seq_len
    if batch_size is not None:
      # seq_len x batch_size
      inputs = jnp.stack([inputs] * batch_size, axis=1)

    should_reset = inputs == reset_time
    fwd_result, _ = recurrent.dynamic_unroll(
        core, (inputs[::-1], should_reset[::-1]), initial_state, reverse=False)
    rev_result, _ = recurrent.dynamic_unroll(
        core, (inputs, should_reset), initial_state, reverse=True)
    np.testing.assert_allclose(fwd_result[::-1], rev_result)

  @test_utils.transform_and_run
  def test_allow_batched_only_cores(self):
    # Ensures batched-only cores can be wrapped with ResetCore.
    core = recurrent.ResetCore(_BatchedOnlyCore())
    batch_size = 5
    inputs = jnp.ones((batch_size, 4))
    prev_state = core.initial_state(batch_size)
    should_reset = 0 * prev_state
    core((inputs, should_reset), prev_state)

  @parameterized.parameters(
      (np.array((True, False)),
       np.array(((0, 0), (0, 0)))),
      (np.array((True, False)),
       dict(core=np.array(((0, 0), (0, 0))))),
      (np.array((True, False)),
       np.array(((0, 0, 0, 0), (0, 0, 0, 0))).reshape((2, 1, 1, 4))),
      (dict(core=np.array((True, False))),
       dict(core=np.array(((0, 0), (0, 0))))),
  )
  @test_utils.transform_and_run
  def test_input_conform(self, reset, state):
    core = recurrent.ResetCore(core=_DummyCore(state=state))
    core((state, reset), state)

  @parameterized.parameters(
      (np.array((True, False)).reshape((2, 1, 1)),
       np.array(((0, 0), (0, 0)))),
      (dict(core=np.array((True, False))),
       dict(another_core=np.array(((0, 0), (0, 0))))),
  )
  @test_utils.transform_and_run
  def test_input_conform_fails(self, reset, state):
    core = recurrent.ResetCore(core=_DummyCore(state=state))
    with self.assertRaises(ValueError):
      core((state, reset), state)


class IdentityCoreTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_identity_core_call(self):
    core = recurrent.IdentityCore()
    inputs, state_in = object(), object()
    outputs, state_out = core(inputs, state_in)
    self.assertIs(inputs, outputs)
    self.assertIs(state_in, state_out)

  @test_utils.transform_and_run
  def test_identity_core_initial_state(self):
    core = recurrent.IdentityCore()
    self.assertEqual(core.initial_state(1), ())


class DeepRNNTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_only_callables(self):
    x = make_sequence([4, 3])  # [B, F]
    core = recurrent.DeepRNN([jnp.tanh, jnp.square])
    initial_state = core.initial_state(x.shape[0])
    out, next_state = core(x, initial_state)
    np.testing.assert_allclose(out, np.square(np.tanh(x)), rtol=1e-4)
    self.assertEmpty(next_state)
    self.assertEmpty(initial_state)

  @test_utils.transform_and_run
  def test_connection_and_shapes(self):
    batch_size = 4
    x = make_sequence([batch_size, 3])  # [B, F]
    core = recurrent.DeepRNN([
        recurrent.VanillaRNN(hidden_size=3),
        basic.Linear(2),
        jax.nn.relu,
        recurrent.VanillaRNN(hidden_size=5),
        jax.nn.relu,
    ])
    initial_state = core.initial_state(x.shape[0])
    out, next_state = core(x, initial_state)

    self.assertEqual(out.shape, (batch_size, 5))
    # Verifies that at least last layer of relu is applied.
    self.assertTrue(np.all(out >= np.zeros([batch_size, 5])))

    self.assertLen(next_state, 2)
    self.assertEqual(initial_state[0].shape, (batch_size, 3))
    self.assertEqual(initial_state[1].shape, (batch_size, 5))

    self.assertLen(initial_state, 2)
    np.testing.assert_allclose(initial_state[0], jnp.zeros([batch_size, 3]))
    np.testing.assert_allclose(initial_state[1], jnp.zeros([batch_size, 5]))

  @test_utils.transform_and_run
  def test_skip_connections(self):
    batch_size = 4
    x = make_sequence([batch_size, 3])  # [B, F]
    core = recurrent.deep_rnn_with_skip_connections([
        recurrent.VanillaRNN(hidden_size=3),
        recurrent.VanillaRNN(hidden_size=5),
    ])
    initial_state = core.initial_state(x.shape[0])
    out, _ = core(x, initial_state)
    self.assertEqual(out.shape, (batch_size, 8))
    # Previous tests test the correctness of state handling.

  @test_utils.transform_and_run
  def test_skip_validation(self):
    with self.assertRaisesRegex(ValueError, "skip_connections requires"):
      recurrent.deep_rnn_with_skip_connections([jax.nn.relu])


class BatchMajorUnrollTest(parameterized.TestCase):

  @parameterized.parameters(recurrent.dynamic_unroll, recurrent.static_unroll)
  @test_utils.transform_and_run
  def test_batch_major(self, unroll):
    core = recurrent.LSTM(4)
    sequence_len, batch_size = 10, 5

    inputs = np.random.randn(sequence_len, batch_size, 2)
    batch_major_inputs = jnp.swapaxes(inputs, 0, 1)

    initial_state = core.initial_state(batch_size)
    time_major_outputs, time_major_unroll_state_out = unroll(
        core, inputs, initial_state, time_major=True)
    batch_major_outputs, batch_major_unroll_state_out = unroll(
        core, batch_major_inputs, initial_state, time_major=False)

    jax.tree_multimap(np.testing.assert_array_equal,
                      time_major_unroll_state_out, batch_major_unroll_state_out)
    jax.tree_multimap(
        lambda x, y: np.testing.assert_array_equal(x, jnp.swapaxes(y, 0, 1)),
        time_major_outputs, batch_major_outputs)


if __name__ == "__main__":
  absltest.main()
