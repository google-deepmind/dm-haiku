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
"""Tests for haiku._src.reshape."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import reshape
from haiku._src import test_utils
from haiku._src import transform
import jax.numpy as jnp
import numpy as np

B, H, W, C, D = 2, 3, 4, 5, 6


class ReshapeTest(parameterized.TestCase):

  @parameterized.parameters(
      (1, (B, H * W * C, D)),
      (2, (B, H, W * C, D)),
      (3, (B, H, W, C, D)),
      (4, (B, H, W, C, 1, D)),
  )
  def test_reshape(self, preserve_dims, expected_output_shape):
    def f(inputs):
      return reshape.Reshape(output_shape=(-1, D),
                             preserve_dims=preserve_dims)(inputs)
    init_fn, apply_fn = transform.transform(f)
    params = init_fn(None, jnp.ones([B, H, W, C, D]))
    outputs = apply_fn(params, np.ones([B, H, W, C, D]))
    self.assertEqual(outputs.shape, expected_output_shape)

  def test_invalid_multiple_wildcard(self):
    def f():
      mod = reshape.Reshape(output_shape=[-1, -1])
      return mod(np.ones([1, 2, 3]))

    init_fn, _ = transform.transform(f)
    with self.assertRaises(ValueError):
      init_fn(None)

  def test_invalid_type(self):
    def f():
      mod = reshape.Reshape(output_shape=[7, "string"])
      return mod(np.ones([1, 2, 3]))

    init_fn, _ = transform.transform(f)
    with self.assertRaises(TypeError):
      init_fn(None)

  def test_flatten(self):
    def f():
      return reshape.Flatten(preserve_dims=2)(jnp.zeros([2, 3, 4, 5]))

    init_fn, apply_fn = transform.transform(f)
    params = init_fn(None)
    self.assertEqual(apply_fn(params).shape, (2, 3, 20))

  @test_utils.transform_and_run
  def test_flatten_1d(self):
    mod = reshape.Flatten()
    x = jnp.zeros([10])
    y = mod(x)
    self.assertEqual(x.shape, y.shape)

  @test_utils.transform_and_run
  def test_flatten_nd(self):
    mod = reshape.Flatten(preserve_dims=2)
    x = jnp.zeros([2, 3])
    y = mod(x)
    self.assertEqual(x.shape, y.shape)

  @test_utils.transform_and_run
  def test_flatten_invalid_preserve_dims(self):
    with self.assertRaisesRegex(ValueError,
                                "Argument preserve_dims should be >= 1."):
      reshape.Flatten(preserve_dims=-1)


if __name__ == "__main__":
  absltest.main()
