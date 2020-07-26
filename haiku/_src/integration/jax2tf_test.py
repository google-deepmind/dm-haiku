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

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

ModuleFn = descriptors.ModuleFn

DEFAULT_ATOL = 1e-2
CUSTOM_ATOL = {hk.nets.ResNet: 0.1}

# TODO(tomhennigan): Test with experimental_compile=True.
TF_TRANSFORM = (("identity", lambda f: f), ("tf.function", tf.function))
JAX_TRANSFORM = (("identity", lambda f: f), ("jax.jit", jax.jit))


class JaxToTfTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES,
                                        test_utils.named_bools("init"),
                                        TF_TRANSFORM, JAX_TRANSFORM)
  def test_convert(
      self,
      module_fn: ModuleFn,
      shape,
      dtype,
      init: bool,
      tf_transform,
      jax_transform,
  ):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)

    def g(x):
      return module_fn()(x)

    f = hk.transform_with_state(g)

    atol = CUSTOM_ATOL.get(descriptors.module_type(module_fn), DEFAULT_ATOL)
    assert_allclose = functools.partial(np.testing.assert_allclose, atol=atol)

    get = lambda t: jax.tree_map(lambda x: x.numpy(), t)

    if init:
      init_jax = jax_transform(f.init)
      init_tf = tf_transform(jax2tf.convert(f.init))
      jax.tree_multimap(assert_allclose, init_jax(rng, x), get(init_tf(rng, x)))

    else:
      params, state = f.init(rng, x)
      apply_jax = jax_transform(f.apply)
      apply_tf = tf_transform(jax2tf.convert(f.apply))
      jax.tree_multimap(assert_allclose,
                        apply_jax(params, state, rng, x),
                        get(apply_tf(params, state, rng, x)))

if __name__ == "__main__":
  absltest.main()
