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
"""Tests for haiku._src.batch_norm."""

from absl.testing import absltest
from haiku._src import base
from haiku._src import batch_norm
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp
import numpy as np


_EPS = 1e-10


def _make_batch_norm(**kwargs):
  return batch_norm.BatchNorm(
      create_scale=kwargs.pop("create_scale", True),
      create_offset=kwargs.pop("create_offset", True),
      decay_rate=kwargs.pop("decay_rate", 0.9),
      **kwargs,
  )


def _expected_group_norm(inputs, groups):
  """Computes expected per-group normalized outputs for pmap tests."""
  expected = np.empty_like(inputs)
  for group in groups:
    group_inputs = inputs[group]
    mean = np.mean(group_inputs, axis=0, keepdims=True)
    std = np.std(group_inputs, axis=0, keepdims=True) + _EPS
    expected[group] = (group_inputs - mean) / std
  return expected


class BatchNormTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_basic(self):
    data = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape([2, 3, 4])

    bn = batch_norm.BatchNorm(True, True, 0.9)
    result = bn(data, is_training=True)

    # Input data is symmetric across channels, so every channel should match.
    np.testing.assert_allclose(result, np.broadcast_to(result[:, :, :1], result.shape))

    # Running in eval mode should reproduce the same result.
    np.testing.assert_allclose(bn(data, is_training=False), result, rtol=2e-2, atol=2e-2)

  @test_utils.transform_and_run
  def test_simple_training(self):
    layer = batch_norm.BatchNorm(
        create_scale=False,
        create_offset=False,
        decay_rate=0.9,
    )

    inputs = np.ones([2, 3, 3, 5], dtype=np.float32)
    scale = np.full((5,), 0.5, dtype=np.float32)
    offset = np.full((5,), 2.0, dtype=np.float32)

    result = layer(inputs, True, scale=scale, offset=offset)
    np.testing.assert_array_equal(result, np.full(inputs.shape, 2.0, dtype=np.float32))

  @test_utils.transform_and_run
  def test_simple_training_nchw(self):
    layer = batch_norm.BatchNorm(
        create_scale=False,
        create_offset=False,
        decay_rate=0.9,
        data_format="NCHW",
    )

    inputs = np.ones([2, 5, 3, 3], dtype=np.float32)
    scale = np.full((5, 1, 1), 0.5, dtype=np.float32)
    offset = np.full((5, 1, 1), 2.0, dtype=np.float32)

    result = layer(inputs, True, scale=scale, offset=offset)
    np.testing.assert_array_equal(result, np.full(inputs.shape, 2.0, dtype=np.float32))

  @test_utils.transform_and_run
  def test_simple_training_normalized_axes(self):
    layer = batch_norm.BatchNorm(
        create_scale=False,
        create_offset=False,
        decay_rate=0.9,
        axis=[0, 2, 3],  # Not the second axis.
    )

    # This differs only in the second axis.
    inputs = np.stack(
        [2.0 * np.ones([5, 3, 3], dtype=np.float32),
         np.ones([5, 3, 3], dtype=np.float32)],
        axis=1,
    )

    result = layer(inputs, True)

    # Normalizing each slice separately should collapse to zeros.
    np.testing.assert_array_equal(result, np.zeros_like(inputs))

  def test_simple_training_cross_replica_axis(self):
    ldc = jax.local_device_count()

    def f(x, is_training=True):
      return batch_norm.BatchNorm(
          create_scale=False,
          create_offset=False,
          decay_rate=0.9,
          cross_replica_axis="i",
      )(x, is_training=is_training)

    f = transform.transform_with_state(f)

    inputs = np.arange(ldc * 4, dtype=np.float32).reshape(ldc, 4)
    key = jax.random.PRNGKey(42)
    key = jnp.broadcast_to(key, (ldc, *key.shape))

    params, state = jax.pmap(f.init, axis_name="i")(key, inputs)
    result, _ = jax.pmap(f.apply, axis_name="i")(params, state, key, inputs)

    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0) + _EPS
    expected = (inputs - mean) / std

    np.testing.assert_allclose(result, expected)

  def test_simple_training_cross_replica_axis_index_groups(self):
    ldc = jax.local_device_count()
    if ldc < 2:
      self.skipTest("Cross-replica test requires at least 2 devices.")

    num_groups = ldc // 2
    num_group_devices = ldc // num_groups
    groups = np.arange(ldc).reshape(num_groups, num_group_devices).tolist()

    def f(x, is_training=True):
      return batch_norm.BatchNorm(
          create_scale=False,
          create_offset=False,
          decay_rate=0.9,
          cross_replica_axis="i",
          cross_replica_axis_index_groups=groups,
      )(x, is_training=is_training)

    f = transform.transform_with_state(f)

    inputs = np.arange(ldc * 4, dtype=np.float32).reshape(ldc, 4)
    key = jax.random.PRNGKey(42)
    key = jnp.broadcast_to(key, (ldc, *key.shape))

    params, state = jax.pmap(f.init, axis_name="i")(key, inputs)
    result, _ = jax.pmap(f.apply, axis_name="i")(params, state, key, inputs)

    expected = _expected_group_norm(inputs, groups)
    np.testing.assert_allclose(result, expected)

  @test_utils.transform_and_run
  def test_no_scale_and_offset(self):
    layer = batch_norm.BatchNorm(
        create_scale=False,
        create_offset=False,
        decay_rate=0.9,
    )

    inputs = jnp.ones([2, 5, 3, 3, 3], dtype=jnp.float32)
    result = layer(inputs, True)
    np.testing.assert_array_equal(result, np.zeros_like(inputs))

  @test_utils.transform_and_run
  def test_no_scale_and_init_provided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `scale_init` if `create_scale=False`"):
      batch_norm.BatchNorm(
          create_scale=False,
          create_offset=True,
          decay_rate=0.9,
          scale_init=jnp.ones,
      )

  @test_utils.transform_and_run
  def test_no_offset_beta_init_provided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `offset_init` if `create_offset=False`"):
      batch_norm.BatchNorm(
          create_scale=True,
          create_offset=False,
          decay_rate=0.9,
          offset_init=jnp.zeros,
      )

  def test_eps_cast_to_var_dtype(self):
    # This checks that a Python float eps does not break bfloat16 outputs.

    def f(x, is_training):
      return batch_norm.BatchNorm(True, True, 0.9, eps=0.1)(x, is_training)

    f = transform.transform_with_state(f)

    x = np.ones([], dtype=jnp.bfloat16)
    key = jax.random.PRNGKey(42)
    params, state = jax.device_get(f.init(key, x, True))
    y, _ = f.apply(params, state, None, x, False)
    self.assertEqual(y.dtype, jnp.bfloat16)

  def test_no_type_promotion(self):
    def get_batch_norm():
      return batch_norm.BatchNorm(
          create_scale=True,
          create_offset=True,
          decay_rate=0.99,
      )

    @transform.transform_with_state
    def forward_training(x):
      return get_batch_norm()(x, is_training=True)

    input_float32 = np.random.normal(size=[100, 5]).astype(np.float32)
    rng = jax.random.PRNGKey(0)
    params, state = forward_training.init(rng, input_float32)
    output, state = forward_training.apply(params, state, rng, input_float32)
    self.assertEqual(output.dtype, jnp.float32)

    def _bfloat16_getter(next_getter, value, context):
      if context.original_dtype == jnp.bfloat16:
        self.assertEqual(value.dtype, jnp.float32)
        value = value.astype(jnp.bfloat16)
      return next_getter(value)

    @transform.transform_with_state
    def forward_eval_bfloat16(x):
      with base.custom_getter(_bfloat16_getter, state=True):
        return get_batch_norm()(x, is_training=False)

    input_bfloat16 = input_float32.astype(jnp.bfloat16)
    output, _ = forward_eval_bfloat16.apply(params, state, rng, input_bfloat16)
    self.assertEqual(output.dtype, jnp.bfloat16)


if __name__ == "__main__":
  absltest.main()
