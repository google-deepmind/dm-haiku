# Lint as: python3
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
"""Tests to ensure all modules work with jax_numpy_rank_promotion == 'raise'."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
from haiku._src.typing import DType, Shape  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp


class RankPromotionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax.config.update('jax_numpy_rank_promotion', 'raise')

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_strict_promotion(
      self,
      module_fn: descriptors.ModuleFn,
      shape: Shape,
      dtype: DType,
  ):
    f = hk.transform_with_state(lambda x: module_fn()(x))  # pylint: disable=unnecessary-lambda
    rng = jax.random.PRNGKey(42)
    x = jnp.ones(shape, dtype)
    params, state = f.init(rng, x)
    self.assertIsNotNone(f.apply(params, state, rng, x))

  def test_rank_promotion_exception(self):
    with self.assertRaises(ValueError) as cm:
      _ = jnp.zeros((8, 1)) * jnp.zeros((8,))

    self.assertIn('could not be broadcast together', str(cm.exception))


if __name__ == '__main__':
  absltest.main()
