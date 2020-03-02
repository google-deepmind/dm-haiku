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
"""Tests for haiku._src.batch_norm."""

from absl.testing import absltest
from haiku._src import batch_norm
from haiku._src import test_utils
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

if __name__ == "__main__":
  absltest.main()
