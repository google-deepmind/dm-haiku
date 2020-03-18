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
"""Tests for haiku._src.moving_averages."""

from absl.testing import absltest
from haiku._src import basic
from haiku._src import moving_averages
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import tree


class MovingAveragesTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_zero_decay(self):
    ema = moving_averages.ExponentialMovingAverage(0.)
    random_input = jax.random.uniform(jax.random.PRNGKey(428), shape=(2, 3, 4))

    # The ema should be equal to the input with decay=0.
    np.testing.assert_allclose(random_input[0], ema(random_input[0]))
    np.testing.assert_allclose(random_input[1], ema(random_input[1]))

  @test_utils.transform_and_run
  def test_warmup(self):
    ema = moving_averages.ExponentialMovingAverage(
        0.5, warmup_length=2, zero_debias=False)
    random_input = jax.random.uniform(jax.random.PRNGKey(428), shape=(2, 3, 4))

    # The ema should be equal to the input for the first two calls.
    np.testing.assert_allclose(random_input[0], ema(random_input[0]))
    np.testing.assert_allclose(random_input[0], ema(random_input[0]))

    # After the warmup period, with decay = 0.5 it should be halfway between the
    # first two inputs and the new input.
    np.testing.assert_allclose(
        (random_input[0] + random_input[1]) / 2, ema(random_input[1]))

  @test_utils.transform_and_run
  def test_invalid_warmup_length(self):
    with self.assertRaises(ValueError):
      moving_averages.ExponentialMovingAverage(
          0.5, warmup_length=-1, zero_debias=False)

  @test_utils.transform_and_run
  def test_warmup_length_and_zero_debias(self):
    with self.assertRaises(ValueError):
      moving_averages.ExponentialMovingAverage(
          0.5, warmup_length=2, zero_debias=True)

  @test_utils.transform_and_run
  def test_call(self):
    ema = moving_averages.ExponentialMovingAverage(0.5)
    self.assertAlmostEqual(ema(3.), 3.)
    self.assertAlmostEqual(ema(6.), 5.)

  @test_utils.transform_and_run
  def test_fast_slow_decay(self):
    ema_fast = moving_averages.ExponentialMovingAverage(0.2)
    ema_slow = moving_averages.ExponentialMovingAverage(0.8)
    np.testing.assert_allclose(ema_fast(1.), ema_slow(1.), rtol=1e-4)
    # Expect fast decay to increase more quickly than slow.
    self.assertGreater(ema_fast(2.), ema_slow(2.))

  @test_utils.transform_and_run
  def test_fast_slow_decay_without_update(self):
    ema_fast = moving_averages.ExponentialMovingAverage(0.5)
    ema_slow = moving_averages.ExponentialMovingAverage(0.8)
    # This shouldn't have an effect.
    np.testing.assert_allclose(
        ema_fast(1., update_stats=False),
        ema_slow(1., update_stats=False),
        rtol=1e-4)
    np.testing.assert_allclose(ema_fast(1.), ema_slow(1.), rtol=1e-4)
    self.assertGreater(ema_fast(2.), ema_slow(2.))

  def test_ema_is_identity_on_unchanged_data(self):
    def f(x):
      return moving_averages.ExponentialMovingAverage(0.5)(x)

    inp_value = 1.0
    init_fn, apply_fn = transform.without_apply_rng(
        transform.transform_with_state(f))
    _, params_state = init_fn(None, inp_value)

    # The output should never change as long as the input doesn't.
    value = inp_value
    for _ in range(10):
      value, params_state = apply_fn(None, params_state, value)
      # Floating point error creeps up to 1e-7 (the default).
      np.testing.assert_allclose(inp_value, value, rtol=1e-6)


class EMAParamsTreeTest(absltest.TestCase):

  def test_ema_naming_scheme(self):
    ema_name = "this_is_a_wacky_but_valid_name"
    linear_name = "so_is_this"

    def f():
      return basic.Linear(output_size=2, name=linear_name)(jnp.zeros([6]))

    init_fn, _ = transform.transform(f)
    params = init_fn(random.PRNGKey(428))

    def g(x):
      return moving_averages.EMAParamsTree(0.2, name=ema_name)(x)
    init_fn, _ = transform.transform_with_state(g)
    _, params_state = init_fn(None, params)

    expected_ema_states = [
        "{}/{}__{}".format(ema_name, linear_name, s) for s in ["w", "b"]]
    self.assertEqual(set(expected_ema_states), set(params_state.keys()))

  def test_ema_on_changing_data(self):
    def f():
      return basic.Linear(output_size=2, b_init=jnp.ones)(jnp.zeros([6]))

    init_fn, _ = transform.transform(f)
    params = init_fn(random.PRNGKey(428))

    def g(x):
      return moving_averages.EMAParamsTree(0.2)(x)
    init_fn, apply_fn = transform.without_apply_rng(
        transform.transform_with_state(g))
    _, params_state = init_fn(None, params)
    params, params_state = apply_fn(None, params_state, params)
    # Let's modify our params.
    changed_params = tree.map_structure(lambda t: 2. * t, params)
    ema_params, params_state = apply_fn(None, params_state, changed_params)

    # ema_params should be different from changed params!
    tree.assert_same_structure(changed_params, ema_params)
    for p1, p2 in zip(tree.flatten(params), tree.flatten(ema_params)):
      self.assertEqual(p1.shape, p2.shape)
      with self.assertRaisesRegex(AssertionError, "Not equal to tolerance"):
        np.testing.assert_allclose(p1, p2, atol=1e-6)

  def test_ignore_regex(self):
    def f():
      return basic.Linear(output_size=2, b_init=jnp.ones)(jnp.zeros([6]))

    init_fn, _ = transform.transform(f)
    params = init_fn(random.PRNGKey(428))

    def g(x):
      return moving_averages.EMAParamsTree(0.2, ignore_regex=".*w")(x)
    init_fn, apply_fn = transform.without_apply_rng(
        transform.transform_with_state(g))
    _, params_state = init_fn(None, params)
    params, params_state = apply_fn(None, params_state, params)
    # Let's modify our params.
    changed_params = tree.map_structure(lambda t: 2. * t, params)
    ema_params, params_state = apply_fn(None, params_state, changed_params)

    # W should be the same!
    # ... but b should have changed!
    self.assertTrue((changed_params.linear.b != ema_params.linear.b).all())
    self.assertTrue((changed_params.linear.w == ema_params.linear.w).all())

  def test_tree_update_stats(self):
    def f():
      return basic.Linear(output_size=2, b_init=jnp.ones)(jnp.zeros([6]))

    init_fn, _ = transform.transform(f)
    params = init_fn(random.PRNGKey(428))

    def g(x):
      """This should never update internal stats."""
      return moving_averages.EMAParamsTree(0.2)(x, update_stats=False)
    init_fn, apply_fn_g = transform.without_apply_rng(
        transform.transform_with_state(g))
    _, params_state = init_fn(None, params)

    # Let's modify our params.
    changed_params = tree.map_structure(lambda t: 2. * t, params)
    ema_params, params_state = apply_fn_g(None, params_state, changed_params)
    ema_params2, params_state = apply_fn_g(None, params_state, changed_params)

    # ema_params should be the same as ema_params2 with update_stats=False!
    for p1, p2 in zip(tree.flatten(ema_params2), tree.flatten(ema_params)):
      self.assertEqual(p1.shape, p2.shape)
      np.testing.assert_allclose(p1, p2)

    def h(x):
      """This will behave like normal."""
      return moving_averages.EMAParamsTree(0.2)(x, update_stats=True)
    init_fn, apply_fn_h = transform.without_apply_rng(
        transform.transform_with_state(h))
    _, params_state = init_fn(None, params)
    params, params_state = apply_fn_h(None, params_state, params)

    # Let's modify our params.
    changed_params = tree.map_structure(lambda t: 2. * t, params)
    ema_params, params_state = apply_fn_h(None, params_state, changed_params)
    ema_params2, params_state = apply_fn_h(None, params_state, changed_params)

    # ema_params should be different as ema_params2 with update_stats=False!
    for p1, p2 in zip(tree.flatten(ema_params2), tree.flatten(ema_params)):
      self.assertEqual(p1.shape, p2.shape)
      with self.assertRaisesRegex(AssertionError, "Not equal to tolerance"):
        np.testing.assert_allclose(p1, p2, atol=1e-6)


if __name__ == "__main__":
  absltest.main()
