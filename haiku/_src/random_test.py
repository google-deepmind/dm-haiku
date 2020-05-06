# Lint as: python3
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

import functools

from absl.testing import absltest
from haiku._src import base
from haiku._src import random
from haiku._src import transform
import jax
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
    f_opt = transform.transform(random.optimize_rng_use(f), apply_rng=True)
    jax.tree_multimap(assert_allclose,
                      f_opt.apply({}, key),
                      tuple(jax.random.split(key, 3))[1:])

    # Without optimize_rng_use the keys should be equivalent to splitting in a
    # loop.
    f = transform.transform(f, apply_rng=True)
    jax.tree_multimap(assert_allclose,
                      f.apply({}, key),
                      tuple(split_for_n(key, 2)))


def split_for_n(key, n):
  for _ in range(n):
    key, subkey = jax.random.split(key)
    yield subkey

if __name__ == "__main__":
  absltest.main()
