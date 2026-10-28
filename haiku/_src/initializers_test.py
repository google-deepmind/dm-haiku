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
"""Tests for haiku._src.initializers."""

import itertools as it

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import basic
from haiku._src import initializers
from haiku._src import mixed_precision
from haiku._src import test_utils
from haiku._src import transform
import jax
from jax import config
import jax.numpy as jnp
import jmp
import numpy as np


class InitializersTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_initializers(self):
    as_np_f64 = lambda t: np.array(t, dtype=np.float64)
    # This just makes sure we can call the initializers in accordance to the
    # API and get the right shapes and dtypes out.
    inits = [
        initializers.Constant(42.0),
        initializers.Constant(as_np_f64(42.0)),
        initializers.RandomNormal(),
        initializers.RandomNormal(2.0),
        initializers.RandomNormal(as_np_f64(2.0)),
        initializers.RandomUniform(),
        initializers.RandomUniform(3.0),
        initializers.RandomUniform(as_np_f64(3.0)),
        initializers.VarianceScaling(),
        initializers.VarianceScaling(2.0),
        initializers.VarianceScaling(as_np_f64(2.0)),
        initializers.VarianceScaling(2.0, mode="fan_in"),
        initializers.VarianceScaling(as_np_f64(2.0), mode="fan_in",
                                     fan_in_axes=[0]),
        initializers.VarianceScaling(2.0, mode="fan_in", fan_in_axes=[0]),
        initializers.VarianceScaling(as_np_f64(2.0), mode="fan_in"),
        initializers.VarianceScaling(2.0, mode="fan_out"),
        initializers.VarianceScaling(as_np_f64(2.0), mode="fan_out"),
        initializers.VarianceScaling(2.0, mode="fan_avg"),
        initializers.VarianceScaling(as_np_f64(2.0), mode="fan_avg"),
        initializers.VarianceScaling(2.0, distribution="truncated_normal"),
        initializers.VarianceScaling(
            as_np_f64(2.0), distribution="truncated_normal"),
        initializers.VarianceScaling(2.0, distribution="normal"),
        initializers.VarianceScaling(as_np_f64(2.0), distribution="normal"),
        initializers.VarianceScaling(2.0, distribution="uniform"),
        initializers.VarianceScaling(as_np_f64(2.0), distribution="uniform"),
        initializers.UniformScaling(),
        initializers.UniformScaling(2.0),
        initializers.UniformScaling(as_np_f64(2.0)),
        initializers.TruncatedNormal(),
        initializers.Orthogonal(),
        initializers.Identity(),
        initializers.Identity(as_np_f64(2.0)),

        # Users are supposed to be able to use these.
        jnp.zeros,
        jnp.ones,
    ]

    # TODO(ibab): Test other shapes as well.
    shape = (20, 42)

    dtype = jnp.float32
    for init in inits:
      generated = init(shape, dtype)
      self.assertEqual(generated.shape, shape)
      self.assertEqual(generated.dtype, dtype)

  @test_utils.transform_and_run
  def test_invalid_variance_scale(self):

    with self.assertRaisesRegex(ValueError, "scale.*must be a positive float"):
      initializers.VarianceScaling(scale=-1.0)

    with self.assertRaisesRegex(ValueError, "Invalid `mode` argument*"):
      initializers.VarianceScaling(mode="foo")

    with self.assertRaisesRegex(ValueError, "Invalid `distribution` argument*"):
      initializers.VarianceScaling(distribution="bar")

  @test_utils.transform_and_run
  def test_compute_fans(self):
    fan_in_out1 = initializers._compute_fans([])
    self.assertEqual(fan_in_out1, (1, 1))
    fan_in_out2 = initializers._compute_fans([2])
    self.assertEqual(fan_in_out2, (2, 2))
    fan_in_out3 = initializers._compute_fans([3, 4])
    self.assertEqual(fan_in_out3, (3, 4))
    fan_in_out4 = initializers._compute_fans([1, 2, 3, 4])
    self.assertEqual(fan_in_out4, (6, 8))
    fan_in_out5 = initializers._compute_fans([3, 5, 9], fan_in_axes=[0])
    self.assertEqual(fan_in_out5, (3, 45))
    fan_in_out6 = initializers._compute_fans([3, 5, 7, 4], fan_in_axes=[0, 1])
    self.assertEqual(fan_in_out6, (15, 28))

  @test_utils.transform_and_run
  def test_orthogonal_invalid_shape(self):
    init = initializers.Orthogonal()
    shape = (20,)
    with self.assertRaisesRegex(
        ValueError, "Orthogonal initializer requires at least a 2D shape."):
      init(shape, jnp.float32)

  @test_utils.transform_and_run
  def test_orthogonal_orthogonal(self):
    init = initializers.Orthogonal()
    shape = (42, 20)
    generated = init(shape, jnp.float32)
    self.assertEqual(generated.shape, shape)
    self.assertEqual(generated.dtype, jnp.float32)

  def assertOrthogonal(self, generated, dtype, axis=-1, scale=1.5):
    """Checks the Gram matrix along the requested axis in float64 NumPy."""
    self.assertEqual(generated.dtype, dtype)
    matrix = np.moveaxis(np.asarray(generated, np.float64), axis, 0)
    matrix = matrix.reshape((matrix.shape[0], -1))
    gram = matrix @ matrix.T if matrix.shape[0] < matrix.shape[1] else (
        matrix.T @ matrix)
    tolerance = {
        jnp.float16: 3e-3,
        jnp.bfloat16: 3e-2,
        jnp.float32: 2e-6,
        jnp.float64: 1e-12,
    }[dtype]
    np.testing.assert_allclose(
        gram, scale ** 2 * np.eye(gram.shape[0]),
        rtol=tolerance, atol=tolerance)

  @parameterized.named_parameters(*[
      (f"{dtype.__name__}_{name}", dtype, shape, axis)
      for dtype, (name, shape, axis) in it.product(
          [jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64],
          [("square", (3, 3), -1), ("tall", (4, 3), -1),
           ("wide", (3, 4), -1), ("tensor_axis", (2, 3, 4), 1),
           ("tensor_negative_axis", (2, 3, 4), -2)])
  ])
  def test_orthogonal_dtypes(self, dtype, shape, axis):
    with jax.enable_x64(dtype == jnp.float64):
      f = transform.transform(
          lambda: initializers.Orthogonal(scale=1.5, axis=axis)(shape, dtype))
      key = jax.random.PRNGKey(42)
      for apply_fn in (f.apply, jax.jit(f.apply)):
        generated = apply_fn({}, key)
        self.assertEqual(generated.shape, shape)
        self.assertTrue(jnp.all(jnp.isfinite(generated)))
        self.assertOrthogonal(generated, dtype, axis)

  @parameterized.parameters(jnp.float16, jnp.bfloat16, jnp.float32)
  def test_orthogonal_linear(self, dtype):
    def forward(x):
      return basic.Linear(
          4, with_bias=False, w_init=initializers.Orthogonal(1.5))(x)

    f = transform.transform(forward)
    x = jnp.array([[1, -2, 3], [2, 0, -1]], dtype)
    key = jax.random.PRNGKey(42)
    for init_fn, apply_fn in ((f.init, f.apply),
                             (jax.jit(f.init), jax.jit(f.apply))):
      params = init_fn(key, x)
      weight = params["linear"]["w"]
      self.assertEqual(weight.shape, (3, 4))
      self.assertOrthogonal(weight, dtype)
      actual = apply_fn(params, None, x)
      self.assertEqual(actual.dtype, dtype)
      expected = np.asarray(x, np.float64) @ np.asarray(weight, np.float64)
      np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-2)

  @parameterized.parameters((jnp.float16, "p=f16,c=f32,o=f32"),
                            (jnp.bfloat16, "p=bf16,c=f32,o=f32"))
  def test_orthogonal_linear_policy(self, dtype, policy):
    def forward(x):
      return basic.Linear(
          4, with_bias=False, w_init=initializers.Orthogonal(1.5))(x)

    with mixed_precision.push_policy(basic.Linear, jmp.get_policy(policy)):
      f = transform.transform(forward)
      x = jnp.array([[1, -2, 3], [2, 0, -1]], jnp.float32)
      key = jax.random.PRNGKey(42)
      for init_fn, apply_fn in ((f.init, f.apply),
                               (jax.jit(f.init), jax.jit(f.apply))):
        params = init_fn(key, x)
        weight = params["linear"]["w"]
        self.assertOrthogonal(weight, dtype)
        actual = apply_fn(params, None, x)
        self.assertEqual(actual.dtype, jnp.float32)
        expected = np.asarray(x, np.float64) @ np.asarray(weight, np.float64)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

  @test_utils.transform_and_run
  def test_identity_identity(self):
    init = initializers.Identity()
    shape = (42, 20)
    generated = init(shape, jnp.float32)
    self.assertEqual(generated.shape, shape)
    self.assertEqual(generated.dtype, jnp.float32)

    key = jax.random.PRNGKey(42)
    some_matrix = jax.random.normal(key, (62, 42), jnp.float32)
    np.testing.assert_allclose(some_matrix @ generated, some_matrix[:, :20],
                               rtol=1e-2)

  @test_utils.transform_and_run
  def test_identity_invalid_shape(self):
    init = initializers.Identity()
    shape = (20,)
    with self.assertRaisesRegex(ValueError, "requires at least a 2D shape."):
      init(shape, jnp.float32)

  @parameterized.parameters(
      *it.product([(4, 5), (3, 3), (3, 4, 5), (6, 2, 3, 3)],
                  [3, 1],
                  [jnp.float32, jnp.int32]))
  def testRange(self, shape, gain, dtype):
    init = initializers.Identity(gain)
    value = init(shape, dtype)
    self.assertEqual(value.shape, shape)
    np.testing.assert_almost_equal(value.mean(), gain / shape[-1], decimal=4)
    np.testing.assert_almost_equal(value.max(), gain, decimal=4)

  @test_utils.transform_and_run
  def test_complex_dtype(self):
    if jax.local_devices()[0].platform == "tpu":
      self.skipTest("Complex dtype not supported by TPU")
    # This just makes sure we can call the initializers in accordance to the
    # API and get the right shapes and dtypes out.
    inits = [
        initializers.Constant(42. + 1j * 1729.),
        initializers.RandomNormal(),
        initializers.RandomNormal(2.0),
        initializers.RandomNormal(2. - 3j),
        initializers.TruncatedNormal(),
        initializers.TruncatedNormal(2.),
        initializers.TruncatedNormal(2., 1. - 1j),

        # Users are supposed to be able to use these.
        jnp.zeros,
        jnp.ones,
    ]

    shape = (5, 13, 17)

    dtype = jnp.complex64
    for init in inits:
      generated = init(shape, dtype)
      self.assertEqual(generated.shape, shape)
      self.assertEqual(generated.dtype, dtype)

  @parameterized.parameters(
      *it.product([[1], [1, 2, 3], [1, 2, 3, 4]],
                  [jnp.int32, jnp.float32],
                  [True, False]))
  def test_constant_with_list(self, k, dtype, broadcast):
    init = initializers.Constant(k)
    shape = (1, 1, len(k)) if broadcast else (len(k),)
    actual = init(shape, dtype)
    expected = jnp.broadcast_to(jnp.asarray(k).astype(dtype), shape)
    np.testing.assert_array_equal(actual, expected)
    self.assertEqual(actual.shape, shape)
    self.assertEqual(actual.dtype, dtype)

if __name__ == "__main__":
  config.update("jax_enable_x64", True)
  absltest.main()
