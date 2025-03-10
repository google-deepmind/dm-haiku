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

import os
from absl.testing import absltest
from haiku._src import base
from haiku._src import batch_norm
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp
import numpy as np


class BatchNormTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_basic(self):
    data = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape([2, 3, 4])

    norm = batch_norm.BatchNorm(True, True, 0.9)
    result = norm(data, is_training=True)
    result_0_replicated = jnp.broadcast_to(result[:, :, :1], result.shape)
    # Input data is symmetrical variance per-channel.
    np.testing.assert_allclose(result, result_0_replicated)
    # Running through again in test mode produces same output.
    np.testing.assert_allclose(norm(data, is_training=False), result, rtol=2e-2)

  @test_utils.transform_and_run
  def test_simple_training(self):
    layer = batch_norm.BatchNorm(
        create_scale=False, create_offset=False, decay_rate=0.9)

    inputs = np.ones([2, 3, 3, 5])
    scale = np.full((5,), 0.5)
    offset = np.full((5,), 2.0)

    result = layer(inputs, True, scale=scale, offset=offset)
    np.testing.assert_equal(result, np.full(inputs.shape, 2.0))

  @test_utils.transform_and_run
  def test_simple_training_nchw(self):
    layer = batch_norm.BatchNorm(
        create_scale=False,
        create_offset=False,
        decay_rate=0.9,
        data_format="NCHW")

    inputs = np.ones([2, 5, 3, 3])
    scale = np.full((5, 1, 1), 0.5)
    offset = np.full((5, 1, 1), 2.0)

    result = layer(inputs, True, scale=scale, offset=offset)
    np.testing.assert_equal(result, np.full(inputs.shape, 2.0))

  @test_utils.transform_and_run
  def test_simple_training_normalized_axes(self):
    layer = batch_norm.BatchNorm(
        create_scale=False,
        create_offset=False,
        decay_rate=0.9,
        axis=[0, 2, 3])  # Not the second axis.

    # This differs only in the second axis.
    inputs = np.stack([2.0 * np.ones([5, 3, 3]), np.ones([5, 3, 3])], 1)

    result = layer(inputs, True)

    # Despite not all values being identical, treating slices from the first
    # axis separately leads to a fully normalized = equal array.
    np.testing.assert_equal(result, np.zeros(inputs.shape))

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

    inputs = np.arange(ldc * 4).reshape(ldc, 4)
    key = jax.random.PRNGKey(42)
    key = jnp.broadcast_to(key, (ldc, *key.shape))
    params, state = jax.pmap(f.init, axis_name="i")(key, inputs)
    result, _ = jax.pmap(f.apply, axis_name="i")(params, state, key, inputs)

    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0) + 1e-10
    expected = (inputs - mean) / std

    np.testing.assert_array_almost_equal(result, expected)

  def test_simple_training_cross_replica_axis_index_groups(self):
    ldc = jax.local_device_count()
    if ldc < 2:
      self.skipTest("Cross-replica test requires at least 2 devices.")
    num_groups = ldc // 2
    num_group_devices = ldc // num_groups
    # for 8 devices this produces [[0, 1], [2, 3], [4, 5], [6, 7]] groups.
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

    inputs = np.arange(ldc * 4).reshape(ldc, 4).astype(np.float32)
    key = jax.random.PRNGKey(42)
    key = jnp.broadcast_to(key, (ldc, *key.shape))
    params, state = jax.pmap(f.init, axis_name="i")(key, inputs)
    result, _ = jax.pmap(f.apply, axis_name="i")(params, state, key, inputs)

    expected = np.empty_like(inputs)
    for g in range(num_groups):
      group_inputs = inputs[num_group_devices*g:num_group_devices*(g + 1)]
      group_mean = np.mean(group_inputs, axis=0)
      group_std = np.std(group_inputs, axis=0) + 1e-10
      group_inputs = (group_inputs - group_mean) / group_std
      expected[num_group_devices*g:num_group_devices*(g + 1)] = group_inputs

    np.testing.assert_array_almost_equal(result, expected)

  def test_vmap_cross_replica_axis(self):
    """Tests that BatchNorm works correctly with vmap and cross_replica_axis."""
    batch_size = 4
    height = 32
    width = 32
    channels = 3
    
    def f(x, is_training=True):
      return batch_norm.BatchNorm(
          create_scale=True,
          create_offset=True,
          decay_rate=0.9,
          cross_replica_axis=0,  # Normalize across batch dimension
      )(x, is_training=is_training)

    f = transform.transform_with_state(f)

    # Test with 4D input (batch, height, width, channels)
    rng = jax.random.PRNGKey(42)
    inputs = jax.random.normal(rng, (batch_size, height, width, channels))
    
    # Initialize with first example
    params, state = f.init(rng, inputs[0])
    
    # Test training mode
    batch_f = jax.vmap(f.apply, in_axes=(None, None, None, 0))
    result, new_state = batch_f(params, state, None, inputs)

    # Calculate expected output (normalized across batch dimension)
    spatial_axes = (0, 1, 2)  # batch, height, width
    mean = jnp.mean(inputs, axis=spatial_axes, keepdims=True)
    var = jnp.var(inputs, axis=spatial_axes, keepdims=True)
    expected = (inputs - mean) / jnp.sqrt(var + 1e-5)
    
    # Scale and offset should be applied
    scale = params['batch_norm']['scale']
    offset = params['batch_norm']['offset']
    expected = expected * scale + offset

    # Test results match
    np.testing.assert_array_almost_equal(result, expected, decimal=4)
    
    # Test inference mode
    result_eval, _ = batch_f(params, new_state, None, inputs)
    
    # In eval mode, should use accumulated statistics
    mean_ema = new_state['batch_norm/~/mean_ema']['average']
    var_ema = new_state['batch_norm/~/var_ema']['average']
    expected_eval = ((inputs - mean_ema) / jnp.sqrt(var_ema + 1e-5)) * scale + offset
    
    np.testing.assert_array_almost_equal(result_eval, expected_eval, decimal=4)

  @test_utils.transform_and_run
  def test_no_scale_and_offset(self):
    layer = batch_norm.BatchNorm(
        create_scale=False, create_offset=False, decay_rate=0.9)

    inputs = jnp.ones([2, 5, 3, 3, 3])
    result = layer(inputs, True)
    np.testing.assert_equal(result, np.zeros_like(inputs))

  @test_utils.transform_and_run
  def test_no_scale_and_init_provided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `scale_init` if `create_scale=False`"):
      batch_norm.BatchNorm(
          create_scale=False,
          create_offset=True,
          decay_rate=0.9,
          scale_init=jnp.ones)

  @test_utils.transform_and_run
  def test_no_offset_beta_init_provided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `offset_init` if `create_offset=False`"):
      batch_norm.BatchNorm(
          create_scale=True,
          create_offset=False,
          decay_rate=0.9,
          offset_init=jnp.zeros)

  def test_eps_cast_to_var_dtype(self):
    # See https://github.com/jax-ml/jax/issues/4718 for more info. In the
    # context of this test we need to assert NumPy bf16 params/state and a
    # Python float for eps preserve bf16 output.

    def f(x, is_training):
      return batch_norm.BatchNorm(True, True, 0.9, eps=0.1)(x, is_training)

    f = transform.transform_with_state(f)

    x = np.ones([], jnp.bfloat16)
    key = jax.random.PRNGKey(42)
    params, state = jax.device_get(f.init(key, x, True))
    y, _ = f.apply(params, state, None, x, False)
    self.assertEqual(y.dtype, jnp.bfloat16)

  def test_no_type_promotion(self):

    def get_batch_norm():
      return batch_norm.BatchNorm(
          create_scale=True, create_offset=True, decay_rate=0.99)

    # Initialize the model and "train" it with float32, and check that
    # everything is float32.
    @transform.transform_with_state
    def forward_training(x):
      return get_batch_norm()(x, is_training=True)
    input_float32 = np.random.normal(size=[100, 5]).astype(np.float32)
    rng = jax.random.PRNGKey(0)
    params, state = forward_training.init(rng, input_float32)
    output, state = forward_training.apply(params, state, rng, input_float32)
    self.assertEqual(output.dtype, jnp.float32)

    # Now for eval we want to run with "bfloat16". We use custom getter that
    # will ensure that everytime we ask for a variable with type bfloat16 (as
    # types requested should usually be defined by the type of the inputs),
    # we cast what used to be a float32 into a bfloat16)
    def _bfloat16_getter(next_getter, value, context):
      if context.original_dtype == jnp.bfloat16:
        assert value.dtype == jnp.float32
        value = value.astype(jnp.bfloat16)
      return next_getter(value)

    @transform.transform_with_state
    def forward_eval_bfloat16(x):
      with base.custom_getter(_bfloat16_getter, state=True):
        return get_batch_norm()(x, is_training=False)

    # Run it with bfloat16, inputs, and check that output is still bfloat16.
    input_bfloat16 = input_float32.astype(jnp.bfloat16)
    output, _ = forward_eval_bfloat16.apply(params, state, rng, input_bfloat16)
    self.assertEqual(output.dtype, jnp.bfloat16)

if __name__ == "__main__":
  _xla_flags = os.environ.get("XLA_FLAGS", "")
  os.environ["XLA_FLAGS"] = (_xla_flags +
                             " --xla_force_host_platform_device_count=8")

  absltest.main()

  os.environ["XLA_FLAGS"] = _xla_flags
