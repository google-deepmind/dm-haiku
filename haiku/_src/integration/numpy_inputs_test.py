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

import functools
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
import jax
import jax.numpy as jnp
import numpy as np

ModuleFn = descriptors.ModuleFn


def tree_assert_allclose(a, b, *, atol=1e-6):
  jax.tree_multimap(
      functools.partial(np.testing.assert_allclose, atol=atol), a, b)


class NumpyInputsTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(
      descriptors.ALL_MODULES,
      test_utils.named_bools('np_inputs'),
      test_utils.named_bools('np_params'),
      test_utils.named_bools('close_over_params'))
  def test_numpy_and_jax_results_close(
      self,
      module_fn: ModuleFn,
      shape: Tuple[int, ...],
      dtype: jnp.dtype,
      np_params: bool,
      np_inputs: bool,
      close_over_params: bool,
  ):
    if not (np_params or np_inputs):
      self.skipTest('Pure JAX variants tested elsewhere')

    f = hk.transform_with_state(lambda x: module_fn()(x))  # pylint: disable=unnecessary-lambda

    rng = jax.random.PRNGKey(42)
    x = jnp.ones(shape, dtype)
    params, state = f.init(rng, x)
    if close_over_params:
      apply_fn = functools.partial(f.apply, params, state)
      out, new_state = jax.jit(apply_fn)(rng, x)
    else:
      out, new_state = jax.jit(f.apply)(params, state, rng, x)

    if np_inputs:
      rng, x = jax.device_get((rng, x))

      with self.subTest('init'):
        params2, state2 = f.init(rng, x)
        tree_assert_allclose(params, params2)
        tree_assert_allclose(state, state2)

    with self.subTest('apply'):
      if np_params:
        params, state = jax.device_get((params, state))

      if close_over_params:
        apply_fn = functools.partial(f.apply, params, state)
        out2, new_state2 = jax.jit(apply_fn)(rng, x)
      else:
        out2, new_state2 = jax.jit(f.apply)(params, state, rng, x)

      tree_assert_allclose(out, out2)
      tree_assert_allclose(new_state, new_state2)


if __name__ == '__main__':
  absltest.main()
