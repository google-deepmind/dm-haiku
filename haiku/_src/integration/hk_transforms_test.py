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

DEFAULT_ATOL = 1e-5
CUSTOM_ATOL = {hk.nets.ResNet: 0.05, hk.nets.MobileNetV1: 0.05,
               hk.BatchNorm: 1e-4}


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

    f = hk.transform_with_state(g)

    assert_allclose = functools.partial(np.testing.assert_allclose, atol=1e-4)

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

  @test_utils.combined_named_parameters(
      # TODO(tomhennigan) Enable once grad for _scan_transpose implemented.
      set(descriptors.ALL_MODULES) - set(descriptors.RECURRENT_MODULES))
  def test_hk_remat(
      self,
      module_fn: descriptors.ModuleFn,
      shape: Shape,
      dtype: DType,
  ):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)

    def g(x, remat=False):
      mod = module_fn()
      if remat:
        mod = hk.remat(mod)
      out = mod(x)
      if isinstance(out, dict):
        out = out['loss']
      return jnp.mean(out)

    f = hk.transform_with_state(g)

    assert_allclose = functools.partial(np.testing.assert_allclose, atol=1e-5)

    grad_jax_remat = jax.grad(jax.remat(f.apply), has_aux=True)
    grad_hk_remat = jax.grad(functools.partial(f.apply, remat=True),
                             has_aux=True)

    params, state = f.init(rng, x)
    jax.tree_multimap(assert_allclose,
                      grad_jax_remat(params, state, rng, x),
                      grad_hk_remat(params, state, rng, x))

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_profiler_name_scopes(
      self,
      module_fn: descriptors.ModuleFn,
      shape: Shape,
      dtype: DType,
  ):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)

    def g(x, name_scopes=False):
      hk.experimental.profiler_name_scopes(enabled=name_scopes)
      mod = module_fn()
      return mod(x)

    f = hk.transform_with_state(g)

    assert_allclose = functools.partial(np.testing.assert_allclose, atol=1e-5)

    params, state = f.init(rng, x)
    jax.tree_multimap(assert_allclose,
                      f.apply(params, state, rng, x),
                      f.apply(params, state, rng, x, name_scopes=True))

    # TODO(lenamartens): flip to True when default changes
    hk.experimental.profiler_name_scopes(enabled=False)

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_optimize_rng_use_under_jit(
      self,
      module_fn: descriptors.ModuleFn,
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

    f = hk.transform_with_state(hk.experimental.optimize_rng_use(g))

    module_type = descriptors.module_type(module_fn)
    atol = CUSTOM_ATOL.get(module_type, DEFAULT_ATOL)
    assert_allclose = functools.partial(np.testing.assert_allclose, atol=atol)

    params, state = jax.jit(f.init)(rng, x)
    jax.tree_multimap(assert_allclose, (params, state), f.init(rng, x))

    if module_type in (hk.nets.VectorQuantizer, hk.nets.VectorQuantizerEMA):
      # For stochastic modules just test apply runs.
      jax.device_get(jax.jit(f.apply)(params, state, rng, x))

    else:
      jax.tree_multimap(assert_allclose,
                        jax.jit(f.apply)(params, state, rng, x),
                        f.apply(params, state, rng, x))

if __name__ == '__main__':
  absltest.main()
