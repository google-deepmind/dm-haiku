# Lint as: python3
# Copyright 2020 The Haiku Authors. All Rights Reserved.
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
"""Tests for haiku transforms."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
from haiku._src.typing import DType, Shape  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp
import numpy as np


class HaikuTransformsTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES,
                                        test_utils.named_bools('init'))
  def test_hk_jit(
      self,
      module_fn: descriptors.ModuleFn,
      shape: Shape,
      dtype: DType,
      init: bool,
  ):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)

    def g(x, jit=False):
      mod = module_fn()
      if jit:
        mod = hk.jit(mod)
      return mod(x)

    f = hk.transform(g, state=True, apply_rng=True)

    assert_allclose = functools.partial(np.testing.assert_allclose, atol=1e-5)

    # NOTE: We shard init/apply tests since some modules are expensive to jit
    # (e.g. ResNet50 takes ~60s to compile and we compile it twice per test).
    if init:
      jax.tree_multimap(assert_allclose,
                        jax.jit(f.init)(rng, x),
                        f.init(rng, x, jit=True))

    else:
      params, state = f.init(rng, x)
      jax.tree_multimap(assert_allclose,
                        jax.jit(f.apply)(params, state, rng, x),
                        f.apply(params, state, rng, x, jit=True))

if __name__ == '__main__':
  absltest.main()
