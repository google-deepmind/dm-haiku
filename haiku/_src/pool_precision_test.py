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
"""Low-precision average pooling accumulates before narrowing its result."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def numpy_pool(x, window, strides, padding):
  x = np.asarray(x, dtype=np.float64)
  out_shape = []
  pads = []
  for n, w, s in zip(x.shape, window, strides):
    size = (n + s - 1) // s if padding == "SAME" else max((n - w) // s + 1, 0)
    total = max((size - 1) * s + w - n, 0) if padding == "SAME" else 0
    out_shape.append(size)
    pads.append(total // 2)
  out = np.empty(out_shape)
  for index in itertools.product(*(range(n) for n in out_shape)):
    slices = []
    for i, n, w, s, p in zip(index, x.shape, window, strides, pads):
      start = i * s - p
      slices.append(slice(max(start, 0), min(start + w, n)))
    out[index] = x[tuple(slices)].mean()
  return out


class PoolPrecisionTest(parameterized.TestCase):

  @parameterized.product(
      padding=["SAME", "VALID"],
      channels_first=[False, True],
      value=[256.0, -256.0],
  )
  def test_representable_constant_averages_do_not_overflow(
      self, padding, channels_first, value
  ):
    shape = (1, 2, 16, 16) if channels_first else (1, 16, 16, 2)
    window = (1, 1, 16, 16) if channels_first else (1, 16, 16, 1)
    strides = (1, 1, 8, 8) if channels_first else (1, 8, 8, 1)
    x = jnp.full(shape, value, jnp.float16)
    expected = numpy_pool(x, window, strides, padding)
    fn = lambda inputs: hk.avg_pool(inputs, window, strides, padding)
    for apply in (fn, jax.jit(fn)):
      actual = apply(x)
      self.assertEqual(actual.dtype, jnp.float16)
      np.testing.assert_array_equal(actual, expected)

  @parameterized.product(
      dtype=[jnp.float16, jnp.bfloat16, jnp.float32], padding=["SAME", "VALID"]
  )
  def test_nonuniform_windows_match_high_precision_reference(
      self, dtype, padding
  ):
    values = (np.arange(256).reshape(1, 16, 16, 1) % 13 - 6) * 100 + 400
    x = jnp.asarray(values, dtype)
    window = (1, 8, 8, 1)
    strides = (1, 5, 5, 1)
    expected = jnp.asarray(numpy_pool(x, window, strides, padding), dtype)
    actual = hk.avg_pool(x, window, strides, padding)
    self.assertEqual(actual.dtype, dtype)
    np.testing.assert_allclose(
        actual, expected, rtol=0.008 if dtype == jnp.bfloat16 else 0.001, atol=0
    )

  def test_avgpool_module_has_finite_loss_and_parameter_gradients(self):
    transformed = hk.without_apply_rng(
        hk.transform(lambda x: hk.AvgPool((1, 16, 16, 1), 1, "VALID")(x))
    )
    x = jnp.full((1, 16, 16, 1), 256.0, jnp.float16)
    params = transformed.init(jax.random.PRNGKey(0), x)
    objective = lambda z: jnp.square(
        transformed.apply(params, z).astype(jnp.float32) - 100
    ).sum()
    loss, grad = jax.jit(jax.value_and_grad(objective))(x)
    np.testing.assert_allclose(loss, 156.0**2)
    np.testing.assert_array_equal(grad, np.full(x.shape, 2 * 156 / 256))

  def test_batched_and_vmapped_pooling_agree(self):
    x = jnp.full((2, 16, 16, 1), 256.0, jnp.float16)
    individual = lambda image: hk.avg_pool(
        image, (16, 16, 1), (8, 8, 1), "SAME"
    )
    batched = hk.avg_pool(x, (16, 16, 1), (8, 8, 1), "SAME")
    np.testing.assert_array_equal(jax.jit(jax.vmap(individual))(x), batched)
    self.assertTrue(bool(jnp.isfinite(batched).all()))


if __name__ == "__main__":
  absltest.main()
