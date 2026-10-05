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
"""Low-precision EMA updates retain a representable decay complement."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def make_ema(decay, zero_debias=True, warmup=0):
  def fn(value, update=True):
    ema = hk.ExponentialMovingAverage(
        decay, zero_debias=zero_debias, warmup_length=warmup
    )
    return ema(value, update_stats=update)

  return hk.without_apply_rng(hk.transform_with_state(fn))


class MovingAveragesPrecisionTest(parameterized.TestCase):

  @parameterized.product(
      dtype=[jnp.float16, jnp.bfloat16, jnp.float32], decay=[0.9, 0.999, 0.9999]
  )
  def test_first_debiased_update_recovers_the_input(self, dtype, decay):
    value = jnp.array([1.0, 2.0, -3.0], dtype)
    fn = make_ema(decay)
    params, state = fn.init(jax.random.key(0), value)
    for apply in (fn.apply, jax.jit(fn.apply)):
      actual, new_state = apply(params, state, value)
      self.assertEqual(actual.dtype, value.dtype)
      np.testing.assert_allclose(actual, value, rtol=0.008, atol=0)
      self.assertTrue(bool(jnp.isfinite(actual).all()))
      self.assertEqual(jax.tree.structure(state), jax.tree.structure(new_state))
      self.assertEqual(
          [x.dtype for x in jax.tree.leaves(state)],
          [x.dtype for x in jax.tree.leaves(new_state)],
      )

  @parameterized.product(
      dtype=[jnp.float16, jnp.bfloat16], zero_debias=[False, True]
  )
  def test_scanned_updates_match_quantized_state_reference(
      self, dtype, zero_debias
  ):
    decay = 0.9999
    sequence = jnp.array(
        [[1.0, 2.0], [2.0, 1.0], [4.0, -1.0], [1.0, 0.0]], dtype
    )
    fn = make_ema(decay, zero_debias)
    params, state = fn.init(jax.random.key(0), sequence[0])

    def step(carry, x):
      out, carry = fn.apply(params, carry, x)
      return carry, out

    final, actual = jax.jit(lambda s, x: jax.lax.scan(step, s, x))(
        state, sequence
    )
    hidden = np.zeros(2)
    expected = []
    for count, value in enumerate(np.asarray(sequence, dtype=np.float64), 1):
      hidden = hidden * decay + value * (1 - decay)
      output = hidden / (1 - decay**count) if zero_debias else hidden
      expected.append(output)
      hidden = np.asarray(jnp.asarray(hidden, dtype), dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=0.015, atol=2e-6)
    self.assertEqual(
        [x.dtype for x in jax.tree.leaves(state)],
        [x.dtype for x in jax.tree.leaves(final)],
    )

  def test_preview_does_not_mutate_state_and_matches_update(self):
    fn = make_ema(0.9999)
    value = jnp.array([1.0, 2.0], jnp.float16)
    params, state = fn.init(jax.random.key(0), value)
    preview, unchanged = fn.apply(params, state, value, False)
    updated, _ = fn.apply(params, state, value, True)
    np.testing.assert_allclose(preview, value, rtol=0.001)
    np.testing.assert_array_equal(preview, updated)
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(unchanged)):
      np.testing.assert_array_equal(a, b)

  def test_warmup_keeps_state_types_and_starts_decay_after_warmup(self):
    fn = make_ema(0.9999, zero_debias=False, warmup=2)
    value = jnp.array([1.0, 2.0], jnp.float16)
    params, state = fn.init(jax.random.key(0), value)
    for n in range(3):
      out, state = jax.jit(fn.apply)(params, state, value * (n + 1))
      self.assertTrue(bool(jnp.isfinite(out).all()))
      self.assertEqual(out.dtype, jnp.float16)
      if n < 2:
        np.testing.assert_array_equal(out, value * (n + 1))

  def test_explicit_float32_initialization_retains_wider_state(self):
    def forward(x):
      ema = hk.ExponentialMovingAverage(0.9999)
      ema.initialize(x.shape, jnp.float32)
      return ema(x)

    fn = hk.without_apply_rng(hk.transform_with_state(forward))
    value = jnp.array([1.0, 2.0], jnp.float16)
    params, state = fn.init(jax.random.key(0), value)
    actual, updated = jax.jit(fn.apply)(params, state, value)
    self.assertEqual(actual.dtype, jnp.float32)
    self.assertEqual(
        [x.dtype for x in jax.tree.leaves(state)],
        [x.dtype for x in jax.tree.leaves(updated)],
    )
    np.testing.assert_allclose(actual, value, rtol=1e-6)

  @parameterized.parameters(False, True)
  def test_integer_input_dtype_behavior_is_unchanged(self, zero_debias):
    fn = make_ema(0.5, zero_debias=zero_debias)
    value = jnp.array([1, 2], jnp.int32)
    params, state = fn.init(jax.random.key(0), value)
    actual, _ = fn.apply(params, state, value)
    self.assertEqual(actual.dtype, jnp.float32 if zero_debias else jnp.int32)
    np.testing.assert_array_equal(actual, value)

  def test_wider_inputs_keep_the_existing_state_promotion(self):
    fn = make_ema(0.9999)
    initial = jnp.array([1.0, 2.0], jnp.float16)
    params, state = fn.init(jax.random.key(0), initial)
    value = initial.astype(jnp.float32)
    actual, updated = jax.jit(fn.apply)(params, state, value)
    self.assertEqual(actual.dtype, jnp.float32)
    self.assertEqual(
        updated["exponential_moving_average"]["hidden"].dtype, jnp.float32
    )
    np.testing.assert_allclose(actual, value, rtol=1e-6)

  def test_integer_input_with_float_state_keeps_existing_decay_cast(self):
    def forward(value):
      ema = hk.ExponentialMovingAverage(0.5, zero_debias=False)
      ema.initialize(value.shape, jnp.float32)
      return ema(value)

    fn = hk.without_apply_rng(hk.transform_with_state(forward))
    initial = jnp.array([1, 2], jnp.int32)
    params, state = fn.init(jax.random.key(0), initial)
    for value in (initial, initial * 3):
      actual, state = fn.apply(params, state, value)
      self.assertEqual(actual.dtype, jnp.float32)
      np.testing.assert_array_equal(actual, value)

  def test_parameter_tree_wrapper_returns_finite_original_dtypes(self):
    fn = hk.without_apply_rng(
        hk.transform_with_state(lambda p: hk.EMAParamsTree(0.9999)(p))
    )
    values = {
        "layer": {
            "w": jnp.array([1.0, 2.0], jnp.float16),
            "b": jnp.array([3.0], jnp.bfloat16),
        }
    }
    params, state = fn.init(jax.random.key(0), values)
    actual, _ = jax.jit(fn.apply)(params, state, values)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(values)):
      self.assertEqual(a.dtype, b.dtype)
      np.testing.assert_allclose(a, b, rtol=0.008)


if __name__ == "__main__":
  absltest.main()
