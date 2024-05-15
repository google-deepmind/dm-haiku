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

"""Tests for haiku._src.flax.transform_flax."""

from absl.testing import absltest
from absl.testing import parameterized
import flax.errors
import flax.linen as nn
from haiku._src import base
from haiku._src import module
from haiku._src import transform
from haiku._src.flax import transform_flax
import jax
import jax.numpy as jnp


class Child(nn.Module):

  @nn.compact
  def __call__(self):
    if self.is_initializing():
      zeros = jnp.zeros([])
      self.put_variable('params', 'w', zeros + 0)
      self.put_variable('flim_state', 's', zeros + 1)
      self.put_variable('flam_state', 's', zeros + 2)

    w = self.get_variable('params', 'w')
    flim = self.get_variable('flim_state', 's')
    flam = self.get_variable('flam_state', 's')
    return w, flim, flam


class Parent(nn.Module):

  @nn.compact
  def __call__(self):
    Child()()


class Outer(nn.Module):

  @nn.compact
  def __call__(self):
    Parent()()


class Empty(nn.Module):

  def __call__(self):
    pass


class OuterHk(module.Module):

  def __call__(self):
    mod = transform_flax.lift(Outer(), name='outer')
    mod()


class ParamsRNG(nn.Module):

  @nn.compact
  def __call__(self):
    return self.make_rng('params')


class UsesRNG(nn.Module):

  @nn.compact
  def __call__(self):
    self.make_rng('foo')
    self.param('w', nn.initializers.normal(), [], jnp.float32)


class Counter(nn.Module):

  @nn.compact
  def __call__(self):
    if self.is_initializing():
      c = jnp.zeros([])
      self.put_variable('state', 'count', c)
      return c
    else:
      c = self.get_variable('state', 'count')
      self.put_variable('state', 'count', c + 1)
      return c


class TransformFlaxTest(parameterized.TestCase):

  def test_lift_empty_unused(self):
    def f():
      # NOTE: Intentionally ignoring returned lifted object.
      transform_flax.lift(Empty(), name='foo')

    f = transform.transform_with_state(f)
    params, state = f.init(jax.random.PRNGKey(42))
    self.assertEmpty(params)
    self.assertEmpty(state)

  def test_lift_empty(self):
    def f():
      mod = transform_flax.lift(Empty(), name='foo')
      mod()

    f = transform.transform_with_state(f)
    params, state = f.init(jax.random.PRNGKey(42))
    self.assertEmpty(params)
    self.assertEmpty(state)

  def test_lift_toplevel(self):
    def f():
      mod = transform_flax.lift(Child(), name='foo')
      mod()

    f = transform.transform_with_state(f)

    params, state = f.init(jax.random.PRNGKey(42))
    self.assertEqual(params, {'foo/~': {'w': 0}})
    self.assertEqual(state, {'foo/flim_state/~': {'s': 1},
                             'foo/flam_state/~': {'s': 2}})

  def test_lift(self):
    def f():
      mod = transform_flax.lift(Outer(), name='foo')
      mod()

    f = transform.transform_with_state(f)
    rng = jax.random.PRNGKey(42)
    params, state = f.init(rng)
    self.assertEqual(params, {'foo/Parent_0/Child_0': {'w': 0}})
    self.assertEqual(
        state,
        {
            'foo/flim_state/Parent_0/Child_0': {'s': 1},
            'foo/flam_state/Parent_0/Child_0': {'s': 2},
        },
    )

  def test_lift_used_inside_module(self):
    def f():
      mod = OuterHk()
      mod()

    f = transform.transform_with_state(f)
    rng = jax.random.PRNGKey(42)
    params, state = f.init(rng)
    self.assertEqual(params, {'outer_hk/outer/Parent_0/Child_0': {'w': 0}})
    self.assertEqual(
        state,
        {
            'outer_hk/outer/flim_state/Parent_0/Child_0': {'s': 1},
            'outer_hk/outer/flam_state/Parent_0/Child_0': {'s': 2},
        },
    )

  def test_lift_module_called_repeatedly(self):
    def f():
      mod = transform_flax.lift(Outer(), name='foo')
      mod()

    f = transform.transform_with_state(f)
    rng = jax.random.PRNGKey(42)
    params, state = f.init(rng)
    self.assertEqual(params, {'foo/Parent_0/Child_0': {'w': 0}})
    self.assertEqual(
        state,
        {
            'foo/flim_state/Parent_0/Child_0': {'s': 1},
            'foo/flam_state/Parent_0/Child_0': {'s': 2},
        },
    )

  def test_stateful_equivalence(self):
    def f():
      mod = transform_flax.lift(counter, name='foo')
      return mod()

    counter = Counter()
    f = transform.transform_with_state(f)

    rng = jax.random.PRNGKey(42)
    _, state = f.init(rng)
    variables = counter.init(rng)
    self.assertEqual(state, {'foo/state/~': {'count': 0}})
    self.assertEqual(variables, {'state': {'count': 0}})

    for i in range(10):
      out_hk, state = f.apply({}, state, None)
      out_flax, variables = counter.apply(variables, mutable=['state'])
      self.assertEqual(out_hk, out_flax)
      self.assertEqual(state, {'foo/state/~': {'count': i + 1}})
      self.assertEqual(variables, {'state': {'count': i + 1}})

  def test_uses_updated_params(self):
    def f():
      mod = transform_flax.lift(Child(), name='foo')
      return mod()

    f = transform.transform_with_state(f)
    rng = jax.random.PRNGKey(42)
    params, state = f.init(rng)
    # Modify the parameters from their initial value.
    params, state = jax.tree.map(lambda x: x + 1, (params, state))
    (w_out, flim_out, flam_out), _ = f.apply(params, state, None)
    w = params['foo/~']['w']
    flim = state['foo/flim_state/~']['s']
    flam = state['foo/flam_state/~']['s']
    # We want to assert that the params/state passed in are literally what is
    # returned, this ensures that initialisers did not re-run.
    self.assertIs(w, w_out)
    self.assertIs(flim, flim_out)
    self.assertIs(flam, flam_out)

  def test_with_explicit_rngs(self):
    def f():
      mod = transform_flax.lift(ParamsRNG(), name='foo')
      return mod(rngs={'params': jax.random.PRNGKey(42)})

    f = transform.transform(f)
    params = f.init(None)
    self.assertIsNotNone(f.apply(params, None))

  @parameterized.parameters(True, False)
  def test_with_keys_dict(self, with_params_rng: bool):
    def f():
      mod = transform_flax.lift(UsesRNG(), name='foo')
      if with_params_rng:
        rngs = {'params': base.next_rng_key(), 'foo': base.next_rng_key()}
      else:
        rngs = {'foo': base.next_rng_key()}
      return mod(rngs=rngs)

    f = transform.transform(f)
    rng = jax.random.PRNGKey(42)
    init_rng, apply_rng = jax.random.split(rng)
    params = f.init(init_rng)
    f.apply(params, apply_rng)  # Does not fail.

  def test_non_mapping_rngs(self):
    def f():
      mod = transform_flax.lift(ParamsRNG(), name='foo')
      return mod(rngs=jax.random.PRNGKey(42))

    f = transform.transform(f)
    with self.assertRaisesRegex(
        flax.errors.InvalidRngError, 'should be a dictionary'
    ):
      f.init(None)

  def test_lift_multiple_uses(self):
    def f(x):
      mod = transform_flax.lift(nn.Dense(1), name='mod')
      x = mod(x)
      x = mod(x)
      return x

    f = transform.transform(f)
    rng = jax.random.PRNGKey(42)
    x = jnp.ones([1, 1])
    params = f.init(rng, x)
    params = jax.tree.map(lambda x: (x.shape, x.dtype), params)
    self.assertEqual(
        params,
        {
            'mod/~': {
                'bias': ((1,), jnp.float32),
                'kernel': ((1, 1), jnp.float32),
            }
        },
    )

if __name__ == '__main__':
  absltest.main()
