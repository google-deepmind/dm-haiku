# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for haiku._src.multi_transform."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import base
from haiku._src import multi_transform
import jax
import jax.numpy as jnp
import numpy as np


def _assert_arrays_equal(x: jnp.DeviceArray, y: jnp.DeviceArray) -> None:
  np.testing.assert_almost_equal(np.array(x), np.array(y), decimal=5)


class MultiTransformTest(parameterized.TestCase):

  @parameterized.parameters(multi_transform.multi_transform,
                            multi_transform.multi_transform_with_state)
  def test_multi_transform_empty(self, mt):
    for empty_tree in ({}, [], (), AttrMap()):
      with self.subTest(type(empty_tree).__name__):
        f = mt(lambda: (lambda: None, empty_tree))  # pylint: disable=cell-var-from-loop
        f.init(None)
        self.assertEqual(f.apply, empty_tree)

  def test_custom_pytree(self):
    def f():
      init = lambda: None
      foo = lambda: 'foo'
      bar = lambda: 'bar'
      return init, AttrMap(foo=foo, bar=bar)

    f = multi_transform.multi_transform(f)
    self.assertEqual('foo', f.apply.foo({}, None))
    self.assertEqual('bar', f.apply.bar({}, None))

  def test_parameter_in_init(self):
    def f():
      w = base.get_parameter('w', [], init=jnp.zeros)
      s = base.get_state('s', [], init=jnp.zeros)
      init = lambda: None

      def add():
        s_add = base.get_state('s', [], init=jnp.zeros)
        w_add = base.get_parameter('w', [], init=jnp.zeros)
        return w, w_add, s, s_add

      def sub():
        s_sub = base.get_state('s', [], init=jnp.zeros)
        w_sub = base.get_parameter('w', [], init=jnp.zeros)
        return w, w_sub, s, s_sub

      return init, (add, sub)

    f = multi_transform.multi_transform_with_state(f)
    params, state = f.init(None)
    self.assertLen(f.apply, 2)
    for apply_fn in f.apply:
      # Check parameter and state reuse inside the transformed function.
      (w, w_apply, s, s_apply), _ = apply_fn(params, state, None)
      self.assertIs(w, w_apply)
      self.assertIs(s, s_apply)

  def test_state(self):
    def f():
      def init():
        s = base.get_state('s', [], init=jnp.zeros)
        base.set_state('s', s + 1)

      def apply():
        s = base.get_state('s')
        base.set_state('s', s + 1)
      return init, apply

    f = multi_transform.multi_transform_with_state(f)
    _, state_in = f.init(None)
    self.assertEqual(state_in, {'~': {'s': 0}})
    _, state_out = f.apply({}, state_in, None)
    self.assertEqual(state_out, {'~': {'s': 1}})


# Example custom pytree (a dict where `x.a` behaves like `x['a']`).
class AttrMap(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

jax.tree_util.register_pytree_node(AttrMap,
                                   lambda d: (list(d.values()), d.keys()),
                                   lambda k, v: AttrMap(zip(k, v)))

if __name__ == '__main__':
  absltest.main()
