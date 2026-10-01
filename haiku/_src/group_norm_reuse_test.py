# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Regression tests for a GroupNorm instance reused across input shapes."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _reference(x, channels_first, scale, offset):
  # Explicitly normalize each channel group, without inferring a batch size.
  values = jnp.moveaxis(x, 1, -1) if channels_first else x
  groups = jnp.split(values, 2, axis=-1)
  axes = tuple(range(1, values.ndim))
  normalized = [
      (group - jnp.mean(group, axis=axes, keepdims=True))
      / jnp.sqrt(jnp.var(group, axis=axes, keepdims=True) + 1e-5)
      for group in groups
  ]
  result = jnp.concatenate(normalized, axis=-1) * scale + offset
  return jnp.moveaxis(result, -1, 1) if channels_first else result


class GroupNormReuseTest(parameterized.TestCase):

  @parameterized.product(
      channels_first=(False, True),
      second_shape=((1, 3, 2, 8), (1, 3, 4, 8), (1, 2, 2, 8), (3, 2, 3, 8)),
      affine=(False, True),
      compiled=(False, True),
  )
  def test_reuse_preserves_shape_values_and_gradients(
      self, channels_first, second_shape, affine, compiled
  ):
    first = jnp.sin(jnp.arange(48, dtype=jnp.float32)).reshape(1, 2, 3, 8)
    second = jnp.cos(jnp.arange(np.prod(second_shape), dtype=jnp.float32))
    second = second.reshape(second_shape)
    if channels_first:
      first = jnp.moveaxis(first, -1, 1)
      second = jnp.moveaxis(second, -1, 1)

    def forward(x, y):
      norm = hk.GroupNorm(
          2,
          create_scale=affine,
          create_offset=affine,
          scale_init=hk.initializers.Constant(0.7) if affine else None,
          offset_init=hk.initializers.Constant(0.2) if affine else None,
          data_format="channels_first" if channels_first else "channels_last",
      )
      return norm(x), norm(y)

    transformed = hk.without_apply_rng(hk.transform(forward))
    params = transformed.init(jax.random.PRNGKey(7), first, second)
    apply = jax.jit(transformed.apply) if compiled else transformed.apply
    actual_first, actual_second = apply(params, first, second)
    scale, offset = (0.7, 0.2) if affine else (1.0, 0.0)
    expected_first = _reference(first, channels_first, scale, offset)
    expected_second = _reference(second, channels_first, scale, offset)
    self.assertEqual(actual_first.shape, first.shape)
    self.assertEqual(actual_second.shape, second.shape)
    np.testing.assert_allclose(
        actual_first, expected_first, atol=1e-6, rtol=1e-5
    )
    np.testing.assert_allclose(
        actual_second, expected_second, atol=1e-6, rtol=1e-5
    )
    cotangent = jnp.sin(jnp.arange(second.size)).reshape(second.shape)
    actual_grad = jax.grad(
        lambda y: jnp.sum(apply(params, first, y)[1] * cotangent)
    )(second)
    expected_grad = jax.grad(
        lambda y: jnp.sum(
            _reference(y, channels_first, scale, offset) * cotangent
        )
    )(second)
    np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
    if affine:
      self.assertLen(params, 1)
      self.assertEqual(set(next(iter(params.values()))), {"scale", "offset"})


if __name__ == "__main__":
  absltest.main()
