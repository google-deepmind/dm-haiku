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
"""Tests for haiku._src.conformance.descriptors."""

import functools
from typing import Type

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
from haiku._src.typing import DType, Shape  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp
import numpy as np

ModuleFn = descriptors.ModuleFn


def module_type(module_fn: ModuleFn) -> Type[hk.Module]:
  f = hk.transform_with_state(lambda: type(descriptors.unwrap(module_fn())))
  return f.apply(*f.init(jax.random.PRNGKey(42)), None)[0]

DEFAULT_ATOL = 1e-5
CUSTOM_ATOL = {hk.nets.ResNet: 0.05, hk.nets.MobileNetV1: 0.05,
               hk.BatchNorm: 1e-4}


class JaxTransformsTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
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

    def g(x):
      return module_fn()(x)

    f = hk.transform_with_state(g)

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

  @test_utils.combined_named_parameters(descriptors.OPTIONAL_BATCH_MODULES)
  def test_vmap(
      self,
      module_fn: ModuleFn,
      shape: Shape,
      dtype: DType,
  ):
    batch_size, shape = shape[0], shape[1:]
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      sample = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      sample = jax.random.uniform(rng, shape, dtype)
    batch = jnp.broadcast_to(sample, (batch_size,) + sample.shape)

    def g(x):
      return module_fn()(x)

    f = hk.transform_with_state(g)

    # Ensure application under vmap is the same.
    params, state = f.init(rng, sample)
    v_apply = jax.vmap(f.apply, in_axes=(None, None, None, 0))
    jax.tree_multimap(
        lambda a, b: np.testing.assert_allclose(a, b, atol=DEFAULT_ATOL),
        f.apply(params, state, rng, batch),
        v_apply(params, state, rng, batch))

if __name__ == '__main__':
  absltest.main()
