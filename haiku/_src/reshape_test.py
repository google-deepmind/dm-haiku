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

import unittest

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import reshape
from haiku._src import test_utils
from haiku._src import transform
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np

B, H, W, C, D = 2, 3, 4, 5, 6


class ReshapeTest(parameterized.TestCase):

  @parameterized.parameters(
      (1, (B, H * W * C, D)),
      (-4, (B, H * W * C, D)),
      (2, (B, H, W * C, D)),
      (-3, (B, H, W * C, D)),
      (3, (B, H, W, C, D)),
      (-2, (B, H, W, C, D)),
      (4, (B, H, W, C, 1, D)),
      (-1, (B, H, W, C, 1, D)),
  )
  def test_reshape(self, preserve_dims, expected_output_shape):
    def f(inputs):
      return reshape.Reshape(output_shape=(-1, D),
                             preserve_dims=preserve_dims)(inputs)
    init_fn, apply_fn = transform.transform(f)
    params = init_fn(None, jnp.ones([B, H, W, C, D]))
    outputs = apply_fn(params, None, np.ones([B, H, W, C, D]))
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

  def test_reshape_convert(self):
    if jax.default_backend() in {"tpu"}:
      raise unittest.SkipTest(
          "Jax2tf native_serialization eager mode is not support in TPU"
      )

    # A function containing a hk.reshape on a polymorphic dimension.  We want
    # to make sure we can convert this method using `jax.jax2tf`.
    def f(inputs):
      mod = reshape.Reshape(output_shape=[1, -1])
      return mod(inputs)

    init_fn, apply_fn = transform.transform(f)
    x1 = jnp.ones([1, 2, 3])
    params = init_fn(None, x1)

    # We convert `f` using `jax2tf` with undefined shape
    converted_f = jax2tf.convert(
        apply_fn,
        polymorphic_shapes=[None, None, "_, T, ..."],  # pytype: disable=wrong-arg-count
        with_gradient=True,
    )

    # Test equality for different inputs shapes.
    original_output1 = apply_fn(params, None, x1)
    converted_output1 = converted_f(params, None, x1)
    self.assertTrue(np.allclose(original_output1, converted_output1))

    x2 = jnp.ones([1, 4, 3])
    converted_output2 = converted_f(params, None, x2)
    original_output2 = apply_fn(params, None, x2)
    self.assertTrue(np.allclose(original_output2, converted_output2))

  def test_flatten(self):
    def f():
      return reshape.Flatten(preserve_dims=2)(jnp.zeros([2, 3, 4, 5]))

    init_fn, apply_fn = transform.transform(f)
    params = init_fn(None)
    self.assertEqual(apply_fn(params, None).shape, (2, 3, 20))

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
  def test_flatten_1d_out_negative(self):
    mod = reshape.Flatten(preserve_dims=-2)
    x = jnp.zeros([2, 3])
    y = mod(x)
    self.assertEqual(y.shape, (6,))

  @test_utils.transform_and_run
  def test_flatten_nd_out_negative(self):
    mod = reshape.Flatten(preserve_dims=-2)
    x = jnp.zeros([5, 2, 3])
    y = mod(x)
    self.assertEqual(y.shape, (5, 6))

  @test_utils.transform_and_run
  def test_flatten_invalid_preserve_dims(self):
    with self.assertRaisesRegex(ValueError,
                                "Argument preserve_dims should be non-zero."):
      reshape.Flatten(preserve_dims=0)


if __name__ == "__main__":
  absltest.main()
