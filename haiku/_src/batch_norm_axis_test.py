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
"""Negative reduction axes must match their nonnegative equivalents."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import batch_norm
from haiku._src import transform
import jax
import jax.numpy as jnp
import numpy as np


class BatchNormNegativeAxisTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("channels_last", (2, 3, 4), (-3, -2), (0, 1)),
      ("mixed_sign", (2, 3, 4), (0, -2), (0, 1)),
      ("channels_first", (2, 4, 3), (-3, -1), (0, 2)),
      ("batch_only", (2, 3, 4), (-3,), (0,)),
  )
  def test_negative_axes_match_positive_axes(self, shape, negative, positive):
    def forward(x, axis, is_training):
      return batch_norm.BatchNorm(True, True, 0.9, axis=axis)(x, is_training)

    f = transform.transform_with_state(forward)
    key = jax.random.PRNGKey(42)
    inputs = jax.random.normal(key, shape)
    params, state = f.init(key, inputs, negative, True)
    reference_params, reference_state = f.init(key, inputs, positive, True)
    self.assertEqual(
        jax.tree_util.tree_structure(params),
        jax.tree_util.tree_structure(reference_params),
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(reference_params),
    ):
      self.assertEqual(actual.shape, expected.shape)
      np.testing.assert_array_equal(actual, expected)
    for training in (True, False):
      expected, _ = f.apply(
          reference_params, reference_state, None, inputs, positive, training
      )

      def apply(x):
        return f.apply(params, state, None, x, negative, training)[0]

      np.testing.assert_allclose(apply(inputs), expected, atol=1e-6)
      np.testing.assert_allclose(jax.jit(apply)(inputs), expected, atol=1e-6)

    actual_grad = jax.grad(
        lambda p: jnp.sum(
            f.apply(p, state, None, inputs, negative, True)[0] ** 2
        )
    )(params)
    expected_grad = jax.grad(
        lambda p: jnp.sum(
            f.apply(p, reference_state, None, inputs, positive, True)[0] ** 2
        )
    )(reference_params)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(actual_grad),
        jax.tree_util.tree_leaves(expected_grad),
    ):
      np.testing.assert_allclose(actual, expected, atol=1e-5)

  @parameterized.parameters(True, False)
  def test_reduced_dimensions_can_change_size(self, training):
    def forward(x):
      return batch_norm.BatchNorm(True, True, 0.9, axis=(-3, -2))(x, training)

    # Initialize running statistics in training mode for both test cases.
    init_f = transform.transform_with_state(
        lambda x: batch_norm.BatchNorm(True, True, 0.9, axis=(-3, -2))(x, True)
    )
    f = transform.transform_with_state(forward)
    key = jax.random.PRNGKey(42)
    params, state = init_f.init(key, jnp.ones((2, 3, 4)))
    inputs = jnp.ones((5, 7, 4))
    result, _ = jax.jit(f.apply)(params, state, None, inputs)
    self.assertEqual(result.shape, inputs.shape)
    self.assertTrue(jnp.all(jnp.isfinite(result)))


if __name__ == "__main__":
  absltest.main()
