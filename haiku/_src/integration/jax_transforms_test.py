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
"""Tests for haiku._src.conformance.descriptors."""

import functools
from typing import Any, Callable, Type

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
from haiku._src.typing import DType, Shape  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp
import numpy as np

ModuleFn = Callable[[], Callable[[jnp.ndarray], Any]]
ALL_MODULES = descriptors.BATCH_MODULES + descriptors.RECURRENT_MODULES


def module_type(module_fn: ModuleFn) -> Type[hk.Module]:
  f = hk.transform(lambda: type(descriptors.unwrap(module_fn())), state=True)
  return f.apply(*f.init(jax.random.PRNGKey(42)))[0]

DEFAULT_ATOL = 1e-5
CUSTOM_ATOL = {hk.nets.ResNet50: 0.05}


class JaxTransformsTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(ALL_MODULES)
  def test_jit(
      self,
      module_fn: ModuleFn,
      shape: Shape,
      dtype: DType,
  ):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)
    f = hk.transform(lambda x: module_fn()(x), state=True, apply_rng=True)  # pylint: disable=unnecessary-lambda

    atol = CUSTOM_ATOL.get(module_type(module_fn), DEFAULT_ATOL)
    assert_allclose = functools.partial(np.testing.assert_allclose, atol=atol)

    # Ensure initialization under jit is the same.
    jax.tree_multimap(assert_allclose,
                      f.init(rng, x),
                      jax.jit(f.init)(rng, x))

    # Ensure application under jit is the same.
    params, state = f.init(rng, x)
    jax.tree_multimap(assert_allclose,
                      f.apply(params, state, rng, x),
                      jax.jit(f.apply)(params, state, rng, x))


if __name__ == '__main__':
  absltest.main()
