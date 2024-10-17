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
from haiku._src import stateful
from haiku._src import test_utils
from haiku._src.integration import descriptors
import jax
import jax.numpy as jnp
import numpy as np

ModuleFn = descriptors.ModuleFn
DEFAULT_ATOL = 1e-5
CUSTOM_ATOL = {hk.nets.ResNet: 0.05, hk.nets.MobileNetV1: 0.05,
               hk.BatchNorm: 1e-4, hk.SeparableDepthwiseConv2D: 3e-3}


class HaikuTransformsTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES,
                                        test_utils.named_bools('init'))
  def test_hk_jit(self, module_fn: ModuleFn, shape, dtype, init):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)

    def g(x, jit=False):
      mod = module_fn()
      if jit:
        mod = stateful.jit(mod)
      return mod(x)

    f = hk.transform_with_state(g)

    assert_allclose = functools.partial(np.testing.assert_allclose, atol=1e-4)

    # NOTE: We shard init/apply tests since some modules are expensive to jit
    # (e.g. ResNet50 takes ~60s to compile and we compile it twice per test).
    if init:
      jax.tree.map(
          assert_allclose, jax.jit(f.init)(rng, x), f.init(rng, x, jit=True)
      )

    else:
      params, state = f.init(rng, x)
      jax.tree.map(
          assert_allclose,
          jax.jit(f.apply)(params, state, rng, x),
          f.apply(params, state, rng, x, jit=True),
      )

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES,
                                        test_utils.named_bools('init'))
  def test_hk_scan(self, module_fn: descriptors.ModuleFn, shape, dtype, init):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)

    def f(x):
      mod = module_fn()
      return mod(x)

    def u_f(xs):
      mod = module_fn()
      def s(carry, x):
        y = mod(x)
        return carry, y
      _, ys = hk.scan(s, (), xs)
      return ys

    u_f = hk.transform_with_state(u_f)
    f = hk.transform_with_state(f)

    assert_allclose = functools.partial(np.testing.assert_allclose, atol=1e-4)
    xs = jnp.broadcast_to(x, (8,) + x.shape)
    params, state = f.init(rng, x)

    if init:
      u_params, u_state = u_f.init(rng, xs)
      jax.tree.map(assert_allclose, u_params, params)
      jax.tree.map(assert_allclose, u_state, state)
      return

    def fun(state, x):
      y, state = f.apply(params, state, rng, x)
      return state, y
    s_state, s_ys = jax.lax.scan(fun, state, xs)
    u_ys, u_state = u_f.apply(params, state, rng, xs)

    jax.tree.map(assert_allclose, u_ys, s_ys)
    jax.tree.map(assert_allclose, u_state, s_state)

  @test_utils.combined_named_parameters(
      # TODO(tomhennigan) Enable once grad for _scan_transpose implemented.
      set(descriptors.ALL_MODULES) - set(descriptors.RECURRENT_MODULES))
  def test_hk_remat(self, module_fn: ModuleFn, shape, dtype):
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
    jax.tree.map(
        assert_allclose,
        grad_jax_remat(params, state, rng, x),
        grad_hk_remat(params, state, rng, x),
    )

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_optimize_rng_use_under_jit(self, module_fn: ModuleFn, shape, dtype):
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
    jax.tree.map(assert_allclose, (params, state), f.init(rng, x))

    if module_type in (hk.nets.VectorQuantizer, hk.nets.VectorQuantizerEMA):
      # For stochastic modules just test apply runs.
      jax.device_get(jax.jit(f.apply)(params, state, rng, x))

    else:
      jax.tree.map(
          assert_allclose,
          jax.jit(f.apply)(params, state, rng, x),
          f.apply(params, state, rng, x),
      )

  @test_utils.combined_named_parameters(descriptors.OPTIONAL_BATCH_MODULES)
  def test_vmap(self, module_fn: ModuleFn, shape, dtype):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)

    # Expand our input since we will map over it.
    x = jnp.broadcast_to(x, (2,) + x.shape)

    f = hk.transform_with_state(lambda x: module_fn()(x))  # pylint: disable=unnecessary-lambda
    f_mapped = hk.transform_with_state(
        lambda x: hk.vmap(lambda x: module_fn()(x), split_rng=False)(x))  # pylint: disable=unnecessary-lambda

    params, state = f_mapped.init(rng, x)

    # JAX vmap with explicitly unmapped params/state/rng. This should be
    # equivalent to `f_mapped.apply(..)` (since by default hk.vmap does not map
    # params/state/rng).
    v_apply = jax.vmap(f.apply,
                       in_axes=(None, None, None, 0),
                       out_axes=(0, None))

    module_type = descriptors.module_type(module_fn)
    atol = CUSTOM_ATOL.get(module_type, DEFAULT_ATOL)
    assert_allclose = functools.partial(np.testing.assert_allclose, atol=atol)
    jax.tree.map(
        assert_allclose,
        f_mapped.apply(params, state, rng, x),
        v_apply(params, state, rng, x),
    )

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_fast_eval_shape(self, module_fn: ModuleFn, shape, dtype):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)

    def g(x):
      return module_fn()(x)

    f = hk.transform_with_state(g)

    init_jax = jax.eval_shape(f.init, rng, x)
    init_hk = hk.experimental.fast_eval_shape(f.init, rng, x)
    self.assertEqual(init_jax, init_hk)

    apply_jax = jax.eval_shape(f.apply, *init_jax, rng, x)
    apply_hk = hk.experimental.fast_eval_shape(f.apply, *init_hk, rng, x)
    self.assertEqual(apply_jax, apply_hk)

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  @test_utils.transform_and_run(run_apply=False)
  def test_fast_eval_shape_inside_transform(self, module_fn: ModuleFn, shape,
                                            dtype):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)

    m = module_fn()
    m_slow = hk.eval_shape(m, x)
    m_fast = hk.experimental.fast_eval_shape(m, x)
    self.assertEqual(m_slow, m_fast)

if __name__ == '__main__':
  absltest.main()
