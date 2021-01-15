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
"""Tests for Haiku to dot functionality."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
import jax
import numpy as np

ModuleFn = descriptors.ModuleFn


class DotTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_abstract_to_dot(self, module_fn: ModuleFn, shape, dtype):
    f = hk.transform_with_state(lambda x: module_fn()(x))  # pylint: disable=unnecessary-lambda
    rng = jax.random.PRNGKey(42)
    x = np.ones(shape, dtype)
    params, state = jax.eval_shape(f.init, rng, x)
    self.assertIsNotNone(
        hk.experimental.abstract_to_dot(f.apply)(params, state, rng, x))

if __name__ == '__main__':
  absltest.main()
