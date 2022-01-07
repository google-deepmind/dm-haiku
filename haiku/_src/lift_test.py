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
from absl.testing import parameterized
from haiku._src import base
from haiku._src import lift
from haiku._src import module
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp
import numpy as np

IGNORE = lambda u: u.ignore_update()
UPDATE = lambda u: u.update({})


class Bias(module.Module):

  def __call__(self, x):
    b = base.get_parameter("b", (), init=jnp.ones)
    return x + b


class LiftTest(parameterized.TestCase):

  def test_lift_with_vmap(self):
    def inner_fn(x):
      assert x.ndim == 1
      return Bias()(x)

    def outer_fn(x):
      assert x.ndim == 2
      x = Bias()(x)
      inner = transform.without_apply_rng(transform.transform(inner_fn))
      inner_p = lift.lift(inner.init)(base.next_rng_key(), x[0])
      vmap_inner = jax.vmap(inner.apply, in_axes=(None, 0))
      return vmap_inner(inner_p, x)

    key = jax.random.PRNGKey(428)
    init_key, apply_key = jax.random.split(key)
    data = np.zeros((3, 2))

    outer = transform.transform(outer_fn)
    outer_params = outer.init(init_key, data)
    self.assertEqual(outer_params, {
        "bias": {"b": np.ones(())},
        "lifted/bias": {"b": np.ones(())},
    })

    out = outer.apply(outer_params, apply_key, data)
    np.testing.assert_equal(out, 2 * np.ones((3, 2)))

  @parameterized.parameters((lift.lift, lambda: None),
                            (lift.lift_with_state, lambda: (None, None)))
  def test_inside_transform(self, lift_fn, init_fn):
    with self.assertRaisesRegex(ValueError, "must be .* part of .*transform"):
      lift_fn(init_fn)

  @test_utils.transform_and_run
  def test_empty_lift(self):
    f = transform.transform(lambda: None)
    self.assertEmpty(lift.lift(f.init)(None))

  @parameterized.parameters(True, False)
  @test_utils.transform_and_run
  def test_empty_lift_with_state(self, ignore_update):
    f = transform.transform_with_state(lambda: None)
    init_fn, updater = lift.lift_with_state(f.init)
    params, state = init_fn(None)
    self.assertEmpty(params)
    self.assertEmpty(state)
    if ignore_update:
      updater.ignore_update()
    else:
      updater.update({})

  def test_unused_updater(self):
    def f() -> lift.LiftWithStateUpdater:
      f = transform.transform_with_state(lambda: None)
      return lift.lift_with_state(f.init)[1]

    f = transform.transform_with_state(f)

    with self.assertRaisesRegex(ValueError, "StateUpdater.*must be used"):
      f.init(None)

  @parameterized.named_parameters(("ignore then ignore", IGNORE, IGNORE),
                                  ("update then update", UPDATE, UPDATE),
                                  ("ignore then update", IGNORE, UPDATE),
                                  ("update then ignore", UPDATE, IGNORE))
  @test_utils.transform_and_run
  def test_used_multiple_times(self, update_fn1, update_fn2):
    f = transform.transform_with_state(lambda: None)
    updater = lift.lift_with_state(f.init)[1]
    update_fn1(updater)
    with self.assertRaisesRegex(ValueError, "must only be used once"):
      update_fn2(updater)

  @test_utils.transform_and_run(run_apply=False)
  def test_lift_raises_with_state(self):
    f = transform.transform_with_state(
        lambda: base.get_state("w", [], init=jnp.zeros))
    lifted = lift.lift(f.init)  # pytype: disable=wrong-arg-types
    with self.assertRaisesRegex(ValueError, "use.*lift_with_state"):
      lifted(None)

  def test_lift_with_state(self):
    def inner():
      w = base.get_state("w", [], init=jnp.zeros)
      w += 1
      base.set_state("w", w)
      return w

    inner = transform.transform_with_state(inner)

    def outer():
      lifted, updater = lift.lift_with_state(inner.init)
      params, state = lifted(None)
      self.assertEmpty(params)
      out, state = inner.apply(params, state, None)
      updater.update(state)
      return out, state

    outer = transform.transform_with_state(outer)
    params, state = outer.init(None)
    self.assertEmpty(params)
    self.assertEqual(jax.tree_map(int, state), {"lifted/~": {"w": 0}})

    for expected in (1, 2, 3):
      (w, inner_state), state = outer.apply(params, state, None)
      self.assertEqual(jax.tree_map(int, inner_state), {"~": {"w": expected}})
      self.assertEqual(w, expected)
      self.assertEmpty(params)
      self.assertEqual(state, {"lifted/~": {"w": expected}})

  @parameterized.parameters(IGNORE, UPDATE)
  def test_updater_used_in_different_inner_transform(self, updater_fn):
    def f():
      g = transform.transform_with_state(lambda: None)
      _, updater = lift.lift_with_state(g.init)
      transform.transform_with_state(lambda: updater_fn(updater)).init(None)

    f = transform.transform_with_state(f)

    with self.assertRaisesRegex(
        ValueError, "must be used within the same call to init/apply"):
      f.init(None)

  def test_transparent_lift(self):
    class OuterModule(module.Module):

      def __call__(self, x):
        x += base.get_parameter("a", shape=[10, 10], init=jnp.zeros)

        def inner_fn(x):
          return InnerModule(name="inner")(x)

        inner_transformed = transform.transform(inner_fn)
        inner_params = lift.transparent_lift(inner_transformed.init)(
            base.next_rng_key(), x)
        x = inner_transformed.apply(inner_params, base.next_rng_key(), x)
        return x

    class InnerModule(module.Module):

      def __call__(self, x):
        x += base.get_parameter("b", shape=[10, 10], init=jnp.zeros)
        return x

    @transform.transform
    def fn(x):
      return OuterModule(name="outer")(x)

    correct_weight_names = ["outer/inner", "outer"]
    rng = jax.random.PRNGKey(0)

    params = fn.init(rng, jnp.ones([10, 10]))

    self.assertCountEqual(list(params.keys()), correct_weight_names)


if __name__ == "__main__":
  absltest.main()
