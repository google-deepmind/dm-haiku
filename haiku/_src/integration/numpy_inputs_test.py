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
"""Tests whether modules produce similar output given np.ndarray inputs."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
import jax
import jax.numpy as jnp
import numpy as np

ModuleFn = descriptors.ModuleFn


class NumpyInputsTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_numpy_and_jax_results_close(self, module_fn: ModuleFn, shape, dtype):

    f = hk.transform_with_state(lambda x: module_fn()(x))  # pylint: disable=unnecessary-lambda

    rng = jax.random.PRNGKey(42)
    x = jnp.ones(shape, dtype)
    params, state = f.init(rng, x)
    out, new_state = f.apply(params, state, rng, x)

    np_rng = np.asarray(rng)
    np_x = np.asarray(x)

    with self.subTest('init'):
      params2, state2 = f.init(np_rng, np_x)
      jax.tree_multimap(np.testing.assert_allclose, params, params2)
      jax.tree_multimap(
          lambda x, y: np.testing.assert_allclose(x, y, atol=1e-9),
          state, state2)

    with self.subTest('apply'):
      np_params = jax.tree_map(np.asarray, params)
      np_state = jax.tree_map(np.asarray, state)
      out2, new_state2 = f.apply(np_params, np_state, np_rng, np_x)
      jax.tree_multimap(np.testing.assert_allclose, out, out2)
      jax.tree_multimap(
          lambda x, y: np.testing.assert_allclose(x, y, atol=1e-9),
          new_state, new_state2)


if __name__ == '__main__':
  absltest.main()
