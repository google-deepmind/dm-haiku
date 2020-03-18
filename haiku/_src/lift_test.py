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
"""Lifting parameters in Haiku."""

from absl.testing import absltest
from haiku._src import base
from haiku._src import lift
from haiku._src import module
from haiku._src import transform
import jax
import jax.numpy as jnp
import numpy as np


class Bias(module.Module):

  def __call__(self, x):
    b = base.get_parameter("b", (), init=jnp.ones)
    return x + b


class LiftTest(absltest.TestCase):

  def test_functionalize(self):
    def inner_fn(x):
      assert x.ndim == 1
      return Bias()(x)

    def outer_fn(x):
      assert x.ndim == 2
      x = Bias()(x)
      inner = transform.transform(inner_fn)
      inner_p = lift.lift(inner.init)(base.next_rng_key(), x[0])
      vmap_inner = jax.vmap(inner.apply, in_axes=(None, 0))
      return vmap_inner(inner_p, x)

    key = jax.random.PRNGKey(428)
    init_key, apply_key = jax.random.split(key)
    data = np.zeros((3, 2))

    outer = transform.transform(outer_fn, apply_rng=True)
    outer_params = outer.init(init_key, data)
    self.assertEqual(outer_params, {
        "bias": {"b": np.ones(())},
        "lifted/bias": {"b": np.ones(())},
    })

    out = outer.apply(outer_params, apply_key, data)
    np.testing.assert_equal(out, 2 * np.ones((3, 2)))


if __name__ == "__main__":
  absltest.main()
