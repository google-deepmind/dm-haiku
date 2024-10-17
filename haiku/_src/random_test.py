# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for haiku._src.random."""

from collections.abc import Sequence
import functools

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import base
from haiku._src import random
from haiku._src import transform
import jax
import jax.extend as jex
import jax.numpy as jnp
import numpy as np


class RandomTest(absltest.TestCase):

  def test_optimize_rng_splitting(self):
    def f():
      k1 = base.next_rng_key()
      k2 = base.next_rng_key()
      return k1, k2

    key = jax.random.PRNGKey(42)
    assert_allclose = functools.partial(np.testing.assert_allclose, atol=1e-5)

    # With optimize_rng_use the keys returned should be equal to split(n).
    f_opt = transform.transform(random.optimize_rng_use(f))
    jax.tree.map(
        assert_allclose,
        f_opt.apply({}, key),
        tuple(jax.random.split(key, 3))[1:],
    )

    # Without optimize_rng_use the keys should be equivalent to splitting in a
    # loop.
    f = transform.transform(f)
    jax.tree.map(assert_allclose, f.apply({}, key), tuple(split_for_n(key, 2)))

  def test_rbg_default_impl(self):
    with jax.default_prng_impl("rbg"):
      key = jax.random.PRNGKey(42)
      self.assertEqual(key.shape, (4,))
      _, apply = transform.transform(base.next_rng_key)
      out_key = apply({}, key)
      self.assertEqual(out_key.shape, (4,))

  def test_rbg_default_impl_invalid_key_shape(self):
    with jax.default_prng_impl("rbg"):
      key = jax.random.PRNGKey(42)[0:2]
      self.assertEqual(key.shape, (2,))
      init, _ = transform.transform(base.next_rng_key)
      with self.assertRaisesRegex(
          ValueError, "Init must be called with an RNG"
      ):
        init(key)

  def test_invalid_key(self):
    init, _ = transform.transform(base.next_rng_key)
    with self.assertRaisesRegex(ValueError, "Init must be called with an RNG"):
      init([1, 2])


class CustomRNGTest(parameterized.TestCase):

  def test_non_custom_key(self):
    init, _ = transform.transform(base.next_rng_key)
    init(jax.random.PRNGKey(42))  # does not crash

  @parameterized.parameters(False, True)
  def test_custom_key(self, do_jit):
    if do_jit:
      self.skipTest("init returns nothing, so compilation may DCE it")

    count = 0

    def count_splits(_, num):
      nonlocal count
      count += 1
      num = tuple(num) if isinstance(num, Sequence) else (num,)
      return jnp.zeros((*num, 13), np.uint32)

    # TODO(frostig): remove after JAX 0.4.20, use
    # jex.random.define_prng_impl directly
    if hasattr(jex.random, "define_prng_impl"):
      def_prng_impl = jex.random.define_prng_impl
    else:
      def_prng_impl = jex.random.PRNGImpl

    differently_shaped_prng_impl = def_prng_impl(
        # Testing a different key shape to make sure it's accepted by Haiku
        key_shape=(13,),
        seed=lambda _: jnp.zeros((13,), np.uint32),
        split=count_splits,
        random_bits=lambda *_, data: jnp.zeros(data, np.uint32),
        fold_in=lambda key, _: key,
    )

    init, _ = transform.transform(base.next_rng_key)
    if do_jit:
      init = jax.jit(init)
    key = jax.random.key(42, impl=differently_shaped_prng_impl)
    init(key)
    self.assertEqual(count, 1)


def split_for_n(key, n):
  for _ in range(n):
    key, subkey = jax.random.split(key)
    yield subkey


if __name__ == "__main__":
  absltest.main()
