# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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

"""Tests for haiku._src.flax.flax_module."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
from haiku._src import base
from haiku._src import filtering
from haiku._src import module
from haiku._src import transform
from haiku._src.flax import flax_module
from haiku._src.flax import utils
from haiku._src.nets import resnet
import jax
import jax.numpy as jnp
import numpy as np


class Counter(module.Module):

  def __call__(self):
    c = base.get_state('c', [], init=jnp.zeros)
    base.set_state('c', c + 1)
    return c


class FlaxModuleTest(parameterized.TestCase):

  def test_transform(self):
    def f():
      w1 = base.get_parameter('w1', [], init=jnp.zeros)
      w2 = base.get_parameter('w2', [], init=jnp.ones)
      return w1, w2

    f = transform.transform(f)
    m = flax_module.Module(f)
    rng = jax.random.PRNGKey(42)
    variables = m.init(rng)
    self.assertEqual(variables, {'params': {'~': {'w1': 0, 'w2': 1}}})

    w1, w2 = m.apply(variables)
    self.assertIs(w1, variables['params']['~']['w1'])
    self.assertIs(w2, variables['params']['~']['w2'])

  def test_transform_with_state(self):
    def f():
      s1 = base.get_state('s1', [], init=jnp.zeros)
      s2 = base.get_state('s2', [], init=jnp.ones)
      base.set_state('s1', s1 + 1)
      base.set_state('s2', s2 + 1)
      return s1, s2

    f = transform.transform_with_state(f)
    m = flax_module.Module(f)
    rng = jax.random.PRNGKey(42)
    variables = m.init(rng)
    self.assertEqual(dict(variables), {'state': {'~': {'s1': 0, 's2': 1}}})

    for i in range(5):
      (s1, s2), variables = m.apply(variables, mutable=['state'])
      self.assertEqual(s1, i)
      self.assertEqual(variables['state']['~']['s1'], i + 1)
      self.assertEqual(s2, i + 1)
      self.assertEqual(variables['state']['~']['s2'], i + 2)

  def test_transform_with_state_not_mutating_state_after_ini(self):
    def f():
      s1 = base.get_state('s1', [], init=jnp.zeros)
      s2 = base.get_state('s2', [], init=jnp.ones)
      return s1, s2

    f = transform.transform_with_state(f)
    m = flax_module.Module(f)
    rng = jax.random.PRNGKey(42)
    variables = m.init(rng)
    self.assertEqual(variables, {'state': {'~': {'s1': 0, 's2': 1}}})

    # NOTE: Intentionally not making state collection mutable.
    s1, s2 = m.apply(variables)
    self.assertIs(s1, variables['state']['~']['s1'])
    self.assertIs(s2, variables['state']['~']['s2'])

  def test_stateful_module(self):
    c = flax_module.Module.create(Counter)
    rng = jax.random.PRNGKey(42)
    variables = c.init(rng)
    self.assertEqual(variables, {'state': {'counter': {'c': 0}}})

    for i in range(10):
      out, variables = c.apply(variables, mutable=['state'])
      self.assertEqual(out, i)
      self.assertEqual(variables, {'state': {'counter': {'c': i + 1}}})

  def test_resnet_50_init_equivalence_to_flax(self):
    mod = flax_module.Module.create(resnet.ResNet50, 10)
    rng = jax.random.PRNGKey(42)
    x = jnp.ones([1, 224, 224, 3])
    f_haiku = transform.transform_with_state(
        lambda x: resnet.ResNet50(10)(x, is_training=True)
    )
    # We check that init is equivalent when passed the RNG used by Flax. There
    # is no mechanism currently to test the inverse (init the Flax module to
    # match Haiku init) because of how Flax and Haiku disagree on RNG key
    # splitting.
    hk_params, hk_state = f_haiku.init(flax_init_rng(rng), x)
    variables = dict(mod.init(rng, x, is_training=True))
    assert_matches(variables['params'], hk_params)
    assert_matches(variables['state'], hk_state)

  def test_resnet_50_apply_equivalence(self):
    mod = flax_module.Module.create(resnet.ResNet50, 10)
    rng = jax.random.PRNGKey(42)
    x = jnp.ones([1, 224, 224, 3])
    f_haiku = transform.transform_with_state(
        lambda x: resnet.ResNet50(10)(x, is_training=True)
    )
    variables = dict(mod.init(rng, x, is_training=True))
    # Haiku and Flax have very different RNG key implementations, so parameter
    # initialisation does not match when using `flax_module`. There is no
    # mechanism to make Flax initialization match Haiku initialisation.
    hk_params = utils.flatten_flax_to_haiku(variables['params'])
    hk_state = utils.flatten_flax_to_haiku(variables['state'])

    for _ in range(5):
      out_flax, state = mod.apply(
          variables, x, is_training=True, mutable=['state']
      )
      out_hk, hk_state = f_haiku.apply(hk_params, hk_state, None, x)
      np.testing.assert_array_equal(out_flax, out_hk)
      variables = {**variables, **state}
      assert_matches(variables['state'], hk_state)


def assert_matches(flax_collection, haiku_collection):
  flax_collection = utils.flatten_flax_to_haiku(flax_collection)
  for mod_name, name, value in filtering.traverse(haiku_collection):
    # We expect equality (not close) because we are running the same
    # operations in the same order on the same data in both cases.
    np.testing.assert_array_equal(
        flax_collection[mod_name][name],
        value,
        err_msg=f'{mod_name}/{name}',
    )


class ParamsRngModule(nn.Module):
  """Module used to exfiltrate a key from the "params" collection."""

  def __call__(self):
    return self.make_rng('params')


def flax_init_rng(rng: jax.Array) -> jax.Array:
  """Returns the rng key that Flax will pass to the Haiku init function."""
  return ParamsRngModule().apply({}, rngs={'params': rng})


if __name__ == '__main__':
  absltest.main()
