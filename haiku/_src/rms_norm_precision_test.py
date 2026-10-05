# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Regression tests for low-precision RMS normalization."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class RMSNormPrecisionTest(parameterized.TestCase):

  @parameterized.product(
      dtype=(jnp.float16, jnp.bfloat16, jnp.float32),
      case=('large', 'small', 'zero'),
      create_scale=(False, True),
      compiled=(False, True),
  )
  def test_values_and_gradients(self, dtype, case, create_scale, compiled):
    values = {
        'large': [256.0, -512.0, 1024.0],
        'small': [1e-4, -2e-4, 3e-4],
        'zero': [0.0, 0.0, 0.0],
    }[case]
    x = jnp.array([values], dtype=dtype)
    epsilon = 1e-8
    scale = 1.5 if create_scale else 1.0
    forward = hk.without_apply_rng(
        hk.transform(
            lambda x: hk.RMSNorm(
                axis=-1,
                eps=epsilon,
                create_scale=create_scale,
                scale_init=(
                    hk.initializers.Constant(scale) if create_scale else None
                ),
            )(x)
        )
    )
    params = forward.init(jax.random.key(1), x)
    evaluate = jax.jit(forward.apply) if compiled else forward.apply
    actual = evaluate(params, x)
    x64 = np.asarray(x).astype(np.float64)
    inverse_rms = 1 / np.sqrt(np.mean(x64**2, axis=-1, keepdims=True) + epsilon)
    expected = x64 * scale * inverse_rms
    self.assertEqual(actual.dtype, x.dtype)
    tolerance = 2e-2 if dtype == jnp.bfloat16 else 3e-3
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=1e-6)
    expected_grad = scale * (
        inverse_rms
        - x64 * np.mean(x64, axis=-1, keepdims=True) * inverse_rms**3
    )
    gradient = jax.grad(lambda x: jnp.sum(forward.apply(params, x)))
    if compiled:
      gradient = jax.jit(gradient)
    np.testing.assert_allclose(
        gradient(x), expected_grad, rtol=tolerance, atol=2e-6
    )
    for value in jax.tree.leaves(params):
      self.assertEqual(value.dtype, x.dtype)
      self.assertEqual(value.shape, (3,))

  @parameterized.parameters(((1, 2),), (slice(1, 3),))
  def test_multiple_axes_and_scale_gradients(self, axis):
    x = jnp.arange(1, 25, dtype=jnp.float16).reshape(2, 3, 4) * 100
    forward = hk.without_apply_rng(
        hk.transform(
            lambda x: hk.RMSNorm(axis=axis, param_axis=-1, eps=1e-5)(x)
        )
    )
    params = forward.init(jax.random.key(1), x)
    actual, param_grad = jax.value_and_grad(
        lambda p: jnp.sum(forward.apply(p, x))
    )(params)
    x64 = np.asarray(x).astype(np.float64)
    normalized = x64 / np.sqrt(
        np.mean(x64**2, axis=(1, 2), keepdims=True) + 1e-5
    )
    np.testing.assert_allclose(actual, normalized.sum(), rtol=2e-3)
    np.testing.assert_allclose(
        jax.tree.leaves(param_grad)[0], normalized.sum(axis=(0, 1)), rtol=2e-3
    )


if __name__ == '__main__':
  absltest.main()
