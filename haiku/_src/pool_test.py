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
"""Tests for haiku._src.pool."""

from absl.testing import absltest

from haiku._src import pool
from haiku._src import test_utils
import jax.numpy as jnp
import numpy as np


class MaxPoolTest(absltest.TestCase):

  def test_max_pool_basic(self):
    x = np.arange(6, dtype=jnp.float32).reshape([6, 1])
    x = np.broadcast_to(x, (2, 10, 6, 2))

    window_shape = [1, 2, 2, 1]
    result = pool.max_pool(
        x, window_shape=window_shape, strides=window_shape, padding="VALID")

    ground_truth = np.asarray([1., 3., 5.]).reshape([3, 1])
    ground_truth = np.broadcast_to(ground_truth, (2, 5, 3, 2))

    np.testing.assert_equal(result, ground_truth)

  def test_max_pool_overlapping_windows(self):
    x = np.arange(12, dtype=jnp.float32).reshape([6, 2])
    x = np.broadcast_to(x, (2, 10, 6, 2))

    window_shape = [1, 5, 3, 2]
    strides = [1, 1, 3, 2]
    result = pool.max_pool(
        x, window_shape=window_shape, strides=strides, padding="VALID")

    ground_truth = np.asarray([5., 11.,]).reshape([2, 1])
    ground_truth = np.broadcast_to(ground_truth, (2, 6, 2, 1))

    np.testing.assert_equal(result, ground_truth)

  def test_max_pool_same_padding(self):
    x = np.arange(6, dtype=jnp.float32)
    x = np.broadcast_to(x, (2, 3, 6))

    window_shape = [1, 3, 3]
    strides = [1, 1, 1]
    result = pool.max_pool(
        x, window_shape=window_shape, strides=strides, padding="SAME")

    np.testing.assert_equal(result.shape, x.shape)

  @test_utils.transform_and_run
  def test_max_pool_same_padding_class(self):
    x = np.arange(6, dtype=jnp.float32)
    x = np.broadcast_to(x, (2, 3, 6))

    window_shape = [1, 3, 3]
    strides = [1, 1, 1]
    max_pool = pool.MaxPool(
        window_shape=window_shape, strides=strides, padding="SAME")
    result = max_pool(x)

    np.testing.assert_equal(result.shape, x.shape)


class AvgPoolTest(absltest.TestCase):

  def test_avg_pool_basic(self):
    x = np.arange(6, dtype=jnp.float32).reshape([6, 1])
    x = np.broadcast_to(x, (2, 10, 6, 2))

    window_shape = [1, 2, 2, 1]
    result = pool.avg_pool(
        x, window_shape=window_shape, strides=window_shape, padding="VALID")

    ground_truth = np.asarray([0.5, 2.5, 4.5]).reshape([3, 1])
    ground_truth = np.broadcast_to(ground_truth, (2, 5, 3, 2))

    np.testing.assert_equal(result, ground_truth)

  def test_avg_pool_overlapping_windows(self):
    x = np.arange(12, dtype=jnp.float32).reshape([6, 2])
    x = np.broadcast_to(x, (2, 10, 6, 2))

    window_shape = [1, 5, 3, 2]
    strides = [1, 1, 3, 2]
    result = pool.avg_pool(
        x, window_shape=window_shape, strides=strides, padding="VALID")

    ground_truth = np.asarray([
        2.5,
        8.5,
    ]).reshape([2, 1])
    ground_truth = np.broadcast_to(ground_truth, (2, 6, 2, 1))

    np.testing.assert_almost_equal(result, ground_truth, decimal=5)

  def test_avg_pool_same_padding(self):
    x = np.ones((2, 3, 6))

    window_shape = [1, 3, 3]
    strides = [1, 1, 1]
    result = pool.avg_pool(
        x, window_shape=window_shape, strides=strides, padding="SAME")

    np.testing.assert_equal(result.shape, x.shape)
    # Since x is constant, its avg value should be itself.
    np.testing.assert_equal(result, x)

  @test_utils.transform_and_run
  def test_avg_pool_same_padding_class(self):
    x = np.ones((2, 3, 6))

    window_shape = [1, 3, 3]
    strides = [1, 1, 1]
    avg_pool = pool.AvgPool(
        window_shape=window_shape, strides=strides, padding="SAME")
    result = avg_pool(x)

    np.testing.assert_equal(result.shape, x.shape)
    # Since x is constant, its avg value should be itself.
    np.testing.assert_equal(result, x)


if __name__ == "__main__":
  absltest.main()
