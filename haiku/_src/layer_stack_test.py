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
"""Tests for haiku._src.layer_stack."""

import functools
import re
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import base
from haiku._src import basic
from haiku._src import config
from haiku._src import initializers
from haiku._src import layer_stack
from haiku._src import module
from haiku._src import multi_transform
from haiku._src import transform
from haiku._src import utils
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats


# Suffixes applied by Haiku for repeated module names.
suffixes = [""] + [f"_{i}" for i in range(1, 100)]


def _slice_layers_params(layers_params):
  sliced_layers_params = {}
  for k, v in layers_params.items():
    for inner_k in v:
      for var_slice, suffix in zip(v[inner_k], suffixes):
        k_new = k.split("/")[-1] + suffix
        if k_new not in sliced_layers_params:
          sliced_layers_params[k_new] = {}
        sliced_layers_params[k_new][inner_k] = var_slice
  return sliced_layers_params


class LayerStackTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._prev_check_jax_usage = config.check_jax_usage(True)

  def tearDown(self):
    super().tearDown()
    config.check_jax_usage(self._prev_check_jax_usage)

  @parameterized.parameters([1, 2, 4])
  def test_layer_stack(self, unroll):
    """Compare layers_stack to the equivalent unrolled stack.

    Tests that the layers_stack application of a Haiku layer function is
    equivalent to repeatedly applying the layer function in an unrolled loop.

    Args:
      unroll: number of unrolled layers.
    """
    num_layers = 20

    def inner_fn(x):
      x += basic.Linear(100, name="linear1")(x)
      x += basic.Linear(100, name="linear2")(x)
      x /= jnp.mean(x)
      return x

    def outer_fn_unrolled(x):
      for _ in range(num_layers):
        x = inner_fn(x)
      return x

    def outer_fn_layer_stack(x):
      stack = layer_stack.layer_stack(num_layers, unroll=unroll)(inner_fn)
      return stack(x)

    unrolled_fn = transform.transform(outer_fn_unrolled)
    layer_stack_fn = transform.transform(outer_fn_layer_stack)

    x = jax.random.uniform(jax.random.PRNGKey(0), [10, 256, 100])

    rng_init = jax.random.PRNGKey(42)

    params = layer_stack_fn.init(rng_init, x)

    sliced_params = _slice_layers_params(params)

    unrolled_pred = unrolled_fn.apply(sliced_params, None, x)
    layer_stack_pred = layer_stack_fn.apply(params, None, x)

    np.testing.assert_allclose(unrolled_pred, layer_stack_pred, atol=1e-3)

  def test_layer_stack_multi_args(self):
    """Compare layers_stack to the equivalent unrolled stack.

    Similar to `test_layer_stack`, but use a function that takes more than one
    argument.
    """
    num_layers = 20

    def inner_fn(x, y):
      x_out = x + basic.Linear(100, name="linear1")(y)
      y_out = y + basic.Linear(100, name="linear2")(x)
      return x_out, y_out

    def outer_fn_unrolled(x, y):
      for _ in range(num_layers):
        x, y = inner_fn(x, y)
      return x, y

    def outer_fn_layer_stack(x, y):
      stack = layer_stack.layer_stack(num_layers)(inner_fn)
      return stack(x, y)

    unrolled_fn = transform.transform(outer_fn_unrolled)
    layer_stack_fn = transform.transform(outer_fn_layer_stack)

    x = jax.random.uniform(jax.random.PRNGKey(0), [10, 256, 100])
    y = jax.random.uniform(jax.random.PRNGKey(1), [10, 256, 100])

    rng_init = jax.random.PRNGKey(42)

    params = layer_stack_fn.init(rng_init, x, y)

    sliced_params = _slice_layers_params(params)

    unrolled_x, unrolled_y = unrolled_fn.apply(sliced_params, None, x, y)
    layer_stack_x, layer_stack_y = layer_stack_fn.apply(params, None, x, y)

    np.testing.assert_allclose(unrolled_x, layer_stack_x, atol=1e-3)
    np.testing.assert_allclose(unrolled_y, layer_stack_y, atol=1e-3)

  def test_layer_stack_no_varargs(self):
    """Test an error is raised when using a function with varargs."""

    class VarArgsModule(module.Module):
      """When used, this module should cause layer_stack to raise an Error."""

      def __call__(self, *args):
        return args

    class NoVarArgsModule(module.Module):
      """This module should be fine to use with layer_stack."""

      def __call__(self, x):
        return x

    def build_and_init_stack(module_class):
      def stack_fn(x):
        mod = module_class()
        return layer_stack.layer_stack(1)(mod)(x)

      stack = multi_transform.without_apply_rng(transform.transform(stack_fn))
      stack.init(jax.random.PRNGKey(1729), jnp.ones([5]))

    build_and_init_stack(NoVarArgsModule)
    with self.assertRaisesRegex(
        ValueError, "The function `f` should not have any `varargs`"):
      build_and_init_stack(VarArgsModule)

  def test_layer_stack_with_state(self):
    def outer_fn_layer_stack(x):
      def simple_stateful_layer(x):
        base.set_state("hi", x)
        return x + 1
      stack = layer_stack.layer_stack(
          5, name="with_state")(simple_stateful_layer)
      return stack(x)

    layer_stack_fn = transform.transform_with_state(outer_fn_layer_stack)

    x = jnp.ones((1,))

    params, state = layer_stack_fn.init(None, x)
    _, state = layer_stack_fn.apply(params, state, None, x)

    np.testing.assert_allclose(state["with_state/~"]["hi"],
                               np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]).T)

  @parameterized.parameters([1, 2, 4])
  def test_layer_stack_grads(self, unroll):
    """Compare layers_stack gradients to the equivalent unrolled stack.

    Tests that the layers_stack application of a Haiku layer function is
    equivalent to repeatedly applying the layer function in an unrolled loop.

    Args:
      unroll: number of unrolled layers.
    """
    num_layers = 20

    def inner_fn(x):
      x += basic.Linear(100, name="linear1")(x)
      x += basic.Linear(100, name="linear2")(x)
      x /= jnp.mean(x)
      return x

    def outer_fn_unrolled(x):
      for _ in range(num_layers):
        x = inner_fn(x)
      return x

    def outer_fn_layer_stack(x):
      stack = layer_stack.layer_stack(num_layers, unroll=unroll)(inner_fn)
      return stack(x)

    unrolled_fn = transform.transform(outer_fn_unrolled)
    layer_stack_fn = transform.transform(outer_fn_layer_stack)

    x = jax.random.uniform(jax.random.PRNGKey(0), [10, 256, 100])

    rng_init = jax.random.PRNGKey(42)

    params = layer_stack_fn.init(rng_init, x)

    sliced_params = _slice_layers_params(params)

    unrolled_grad = jax.grad(
        lambda p, x: jnp.mean(unrolled_fn.apply(p, None, x)))(sliced_params, x)
    layer_stack_grad = jax.grad(
        lambda p, x: jnp.mean(layer_stack_fn.apply(p, None, x)))(params, x)

    assert_fn = functools.partial(
        np.testing.assert_allclose, atol=1e-4, rtol=1e-4)

    jax.tree.map(
        assert_fn, unrolled_grad, _slice_layers_params(layer_stack_grad)
    )

  @unittest.skip("TODO(b/393125732): Debug why this is failing.")
  def test_random(self):
    """Random numbers should be handled correctly."""
    n = 100

    @transform.transform
    @layer_stack.layer_stack(n)
    def add_random(x):
      x = x + jax.random.normal(base.next_rng_key())
      return x

    # Evaluate a bunch of times
    key, *keys = jax.random.split(jax.random.PRNGKey(7), 1024 + 1)
    params = add_random.init(key, 0.)
    apply_fn = jax.jit(add_random.apply)
    values = [apply_fn(params, key, 0.) for key in keys]

    # Should be roughly N(0, sqrt(n))
    cdf = stats.norm(scale=np.sqrt(n)).cdf
    _, p = stats.kstest(values, cdf)  # pytype: disable=attribute-error
    self.assertLess(p, 0.1)

  def test_threading(self):
    """Test @layer_stack when the function gets per-layer inputs."""
    n = 5

    @layer_stack.layer_stack(n, with_per_layer_inputs=True)
    def f(x, y):
      x = x + y * jax.nn.one_hot(y, len(x)) / 10
      return x, 2 * y

    @multi_transform.without_apply_rng
    @transform.transform
    def g(x, ys):
      x, zs = f(x, ys)
      # Check here to catch issues at init time
      self.assertEqual(zs.shape, (n,))
      return x, zs

    rng = jax.random.PRNGKey(7)
    x = np.zeros(n)
    ys = np.arange(n).astype(np.float32)
    params = g.init(rng, x, ys)
    x, zs = g.apply(params, x, ys)
    self.assertTrue(np.allclose(x, [0, .1, .2, .3, .4]))
    self.assertTrue(np.all(zs == 2 * ys))

  def test_nested_stacks(self):
    def stack_fn(x):
      def layer_fn(x):
        return basic.Linear(100)(x)

      outer_fn = layer_stack.layer_stack(10)(layer_fn)

      layer_outer = layer_stack.layer_stack(20)(outer_fn)
      return layer_outer(x)

    hk_mod = transform.transform(stack_fn)
    apply_rng, init_rng = jax.random.split(jax.random.PRNGKey(0))

    params = hk_mod.init(init_rng, jnp.zeros([10, 100]))

    hk_mod.apply(params, apply_rng, jnp.zeros([10, 100]))

    p, = params.values()

    assert p["w"].shape == (20, 10, 100, 100), p["w"].shape
    assert p["b"].shape == (20, 10, 100), p["b"].shape

  def test_with_per_layer_inputs_multi_args(self):
    """Test layer_stack with per-layer inputs with multiple arguments."""
    width = 4
    batch_size = 5
    stack_height = 3

    def f_with_multi_args(x, a, b):
      return basic.Linear(
          width, w_init=initializers.Constant(
              jnp.eye(width)))(x) * a + b, None

    @multi_transform.without_apply_rng
    @transform.transform
    def hk_fn(x):
      return layer_stack.layer_stack(
          stack_height, with_per_layer_inputs=True)(f_with_multi_args)(
              x, jnp.full([stack_height], 2.), jnp.ones([stack_height]))

    x = jnp.zeros([batch_size, width])
    key_seq = base.PRNGSequence(19)
    params = hk_fn.init(next(key_seq), x)
    output, z = hk_fn.apply(params, x)
    self.assertIsNone(z)
    self.assertEqual(output.shape, (batch_size, width))
    np.testing.assert_equal(output, np.full([batch_size, width], 7.))

  def test_with_container_state(self):
    width = 2
    batch_size = 2
    stack_height = 3

    def f_with_container_state(x):
      hk_layer = basic.Linear(
          width, w_init=initializers.Constant(jnp.eye(width)))
      layer_output = hk_layer(x)
      layer_state = {
          "raw_output": layer_output,
          "output_projection": jnp.sum(layer_output)
      }
      return layer_output + jnp.ones_like(layer_output), layer_state

    @multi_transform.without_apply_rng
    @transform.transform
    def hk_fn(x):
      return layer_stack.layer_stack(
          stack_height,
          with_per_layer_inputs=True)(f_with_container_state)(x)

    x = jnp.zeros([batch_size, width])
    key_seq = base.PRNGSequence(19)
    params = hk_fn.init(next(key_seq), x)
    output, z = hk_fn.apply(params, x)
    self.assertEqual(z["raw_output"].shape, (stack_height, batch_size, width))
    self.assertEqual(output.shape, (batch_size, width))
    self.assertEqual(z["output_projection"].shape, (stack_height,))
    np.testing.assert_equal(np.sum(z["output_projection"]), np.array(12.))
    np.testing.assert_equal(
        np.all(z["raw_output"] == np.array([0., 1., 2.])[..., None, None]),
        np.array(True))

  @classmethod
  def _compute_weights(cls, stack_height: int, alpha: jax.Array):
    forward = [(alpha, alpha)]
    backward = [(stack_height * alpha, stack_height * alpha)]
    for i in range(2, stack_height + 1):
      a, b = forward[-1]
      forward.append((a * i * alpha, (b + 1) * i * alpha))
      j = stack_height - i + 1
      a, b = backward[-1]
      backward.append((a * j * alpha, (b + 1) * j * alpha))
    return forward, backward

  def test_reverse(self):
    # The layer stack below runs iteratively the update equation:
    # x_n = n * alpha * (x_{n-1} + 1)
    # with x_0 = 1, for n={1, ..., N}, where N = stack_height
    # The reverse layer stack as a result runs the update equation:
    # y_{n-1} = (N - n + 1) * alpha * (y_n + 1)
    # with y_N = 1, for n={N-1, ..., 0}, where N = stack_height
    width = 2
    batch_size = 3
    stack_height = 4
    alpha = jnp.power(24, - 1. / stack_height)
    forward, backward = self._compute_weights(stack_height, alpha)

    def inner_fn(x):
      # Here we initialize the layer to an identity + 1, while later we multiply
      # each parameter by the index `n`.
      return basic.Linear(
          x.shape[1],
          w_init=initializers.Constant(jnp.eye(x.shape[1])),
          b_init=initializers.Constant(1.0),
      )(x)

    @multi_transform.without_apply_rng
    @transform.transform
    def hk_fn(x, reverse=False):
      return layer_stack.layer_stack(stack_height)(inner_fn)(x, reverse=reverse)

    key_seq = base.PRNGSequence(19)
    init_value = 1 + jax.random.uniform(next(key_seq), [batch_size, width])

    def mul_by_m(x):
      m_x = jnp.arange(stack_height) + 1
      while m_x.ndim < x.ndim:
        m_x = m_x[..., None]
      return x * m_x * alpha

    params = jax.tree.map(mul_by_m, hk_fn.init(next(key_seq), init_value))

    a, b = forward[-1]
    x_n = hk_fn.apply(params, init_value)
    np.testing.assert_allclose(x_n, a * init_value + b, rtol=1e-6)

    a, b = backward[-1]
    y_0 = hk_fn.apply(params, init_value, reverse=True)
    np.testing.assert_allclose(y_0, a * init_value + b, rtol=1e-6)

  def test_reverse_with_additional_inputs(self):
    # The layer stack below runs iteratively the update equation:
    # x_n = n * alpha * (x_{n-1} + 1)
    # with x_0 = 1, for n={1, ..., N}, where N = stack_height
    # The reverse layer stack as a result runs the update equation:
    # y_{n-1} = (N - n + 1) * alpha * (y_n + 1)
    # with y_N = 1, for n={N-1, ..., 0}, where N = stack_height
    width = 2
    batch_size = 3
    stack_height = 4
    total_multiplier = 24
    alpha = jnp.power(total_multiplier, - 1. / stack_height)
    forward, backward = self._compute_weights(stack_height, alpha)

    def inner_fn(x, extra):
      # Compared to previous test we pass in the `extra` argument as an
      # additional input, in order to directly initialize the parameters to the
      # index `n` of the iteration.
      out = basic.Linear(
          x.shape[1],
          w_init=initializers.Constant(extra * jnp.eye(x.shape[1])),
          b_init=initializers.Constant(extra),
      )(x)
      return out, out

    @multi_transform.without_apply_rng
    @transform.transform
    def hk_fn(x, extra, reverse=False):
      return layer_stack.layer_stack(
          stack_height, with_per_layer_inputs=True
      )(inner_fn)(x, extra, reverse=reverse)

    extra = jnp.arange(stack_height) + 1
    extra = extra * alpha
    key_seq = base.PRNGSequence(19)
    init_value = 1 + jax.random.uniform(next(key_seq), [batch_size, width])
    params = hk_fn.init(next(key_seq), init_value, extra)

    x_n, x_all = hk_fn.apply(params, init_value, extra)
    self.assertEqual(x_all.shape[0], stack_height)
    for x_t, (a, b) in zip(x_all, forward):
      np.testing.assert_allclose(x_t, a * init_value + b, rtol=1e-6)
    np.testing.assert_allclose(x_n, x_all[-1], rtol=1e-6)

    y_0, y_all = hk_fn.apply(params, init_value, extra, reverse=True)
    self.assertEqual(y_all.shape[0], stack_height)
    for y_t, (a, b) in zip(y_all, reversed(backward)):
      np.testing.assert_allclose(y_t, a * init_value + b, rtol=1e-6)
    np.testing.assert_allclose(y_0, y_all[0], rtol=1e-6)

  def test_reverse_with_pass_reverse_to_layer_fn(self):
    # The layer stack below runs iteratively the update equation:
    # x_n = n * alpha * (x_{n-1} + 1)
    # with x_0 = 1, for n={1, ..., N}, where N = stack_height
    # The reverse layer stack as a result runs the update equation:
    # y_{n-1} = (N - n + 1) * alpha * (y_n + 1)
    # with y_N = 1, for n={N-1, ..., 0}, where N = stack_height
    # This test is equivalent to the previous one, but we nest the iterations in
    # two layer stacks.
    width = 2
    batch_size = 3
    stack_height = 4
    total_multiplier = 24
    alpha = jnp.power(total_multiplier, - 1. / stack_height)
    forward, backward = self._compute_weights(stack_height, alpha)

    def inner_fn(x, extra):
      out = basic.Linear(
          x.shape[1],
          w_init=initializers.Constant(extra * jnp.eye(x.shape[1])),
          b_init=initializers.Constant(extra),
      )(x)
      return out, out

    def outer_fn(x, extra, reverse=False):
      return layer_stack.layer_stack(
          stack_height // 2, with_per_layer_inputs=True
      )(inner_fn)(x, extra, reverse=reverse)

    @multi_transform.without_apply_rng
    @transform.transform
    def hk_fn(x, extra, reverse=False):
      return layer_stack.layer_stack(
          2, with_per_layer_inputs=True, pass_reverse_to_layer_fn=True
      )(outer_fn)(x, extra, reverse=reverse)

    extra = jnp.arange(stack_height).reshape([2, stack_height // 2]) + 1
    extra = extra * alpha
    key_seq = base.PRNGSequence(19)
    init_value = 1 + jax.random.uniform(next(key_seq), [batch_size, width])
    params = hk_fn.init(next(key_seq), init_value, extra)

    x_n, x_all = hk_fn.apply(params, init_value, extra)
    self.assertEqual(x_all.shape[:2], (2, stack_height // 2))
    x_all = x_all.reshape((stack_height, *x_all.shape[2:]))
    for x_t, (a, b) in zip(x_all, forward):
      np.testing.assert_allclose(x_t, a * init_value + b, rtol=1e-6)
    np.testing.assert_allclose(x_n, x_all[-1], rtol=1e-6)

    y_0, y_all = hk_fn.apply(params, init_value, extra, reverse=True)
    self.assertEqual(y_all.shape[:2], (2, stack_height // 2))
    y_all = y_all.reshape((stack_height, *y_all.shape[2:]))
    for y_t, (a, b) in zip(y_all, reversed(backward)):
      np.testing.assert_allclose(y_t, a * init_value + b, rtol=1e-6)
    np.testing.assert_allclose(y_0, y_all[0], rtol=1e-6)

  def test_parameter_reuse(self):

    def block(x: jax.Array) -> jax.Array:
      h = basic.Linear(output_size=x.shape[-1], with_bias=False)(x)
      h = jax.nn.relu(h)
      return h

    class MLP(basic.hk.Module):

      def __call__(self, x):
        return layer_stack.layer_stack(5)(block)(x)

    def f(x):
      mlp = MLP()
      return mlp(mlp(x))

    x = jnp.ones((2, 2))
    params = transform.transform(f).init(jax.random.PRNGKey(0), x)

    param_size = utils.tree_size(params)
    # 5 layers * (2 * 2 weights) = 20.
    np.testing.assert_equal(param_size, 20)

  def test_transparent(self):
    num_layers = 3

    class TransparencyMap(layer_stack.LayerStackTransparencyMapping):

      def stacked_to_flat(self, stacked_module_name: str, scan_idx: int) -> str:
        return stacked_module_name.replace("0", str(scan_idx))

      def flat_to_stacked(
          self, unstacked_module_name: str
      ) -> tuple[str, int] | None:
        idx = int(re.findall(r"\d+", unstacked_module_name)[0])
        return unstacked_module_name.replace(str(idx), "0"), idx

    def block(x: jax.Array, i: int) -> jax.Array:
      return basic.Linear(output_size=x.shape[-1], name=f"linear_{i}")(x)

    def looped(x: jax.Array) -> jax.Array:
      for i in range(num_layers):
        x = block(x, i)
      return x

    def stacked(x: jax.Array) -> jax.Array:
      return layer_stack.layer_stack(
          num_layers=3, transparent=True, transparency_map=TransparencyMap()
      )(lambda y: block(y, 0))(x)

    looped = transform.transform(looped)
    stacked = transform.transform(stacked)

    x = jnp.ones((2, 2))
    rng = jax.random.PRNGKey(0)
    looped_params = looped.init(rng, x)
    stacked_params = stacked.init(rng, x)

    self.assertEqual(
        jax.tree.structure(looped_params),
        jax.tree.structure(stacked_params),
    )

    # Use same set of params for both calls since stacked_params have different
    # value than looped params because differences in RNG splitting.
    np.testing.assert_allclose(
        looped.apply(looped_params, rng, x),
        stacked.apply(looped_params, rng, x),
        rtol=1e-6,
    )

  def test_layer_stack_transparent_with_custom_pytrees(self):
    class TransparencyMap(layer_stack.LayerStackTransparencyMapping):

      def stacked_to_flat(self, stacked_module_name: str, scan_idx: int) -> str:
        return stacked_module_name.replace("0", str(scan_idx))

      def flat_to_stacked(
          self, unstacked_module_name: str
      ) -> tuple[str, int] | None:
        idx = int(re.findall(r"\d+", unstacked_module_name)[0])
        return unstacked_module_name.replace(str(idx), "0"), idx

    @jax.tree_util.register_pytree_node_class
    class CustomParam:

      def __init__(self, param, name):
        self.param = param
        self.multiplier = name

      def tree_flatten(self):
        return ((self.param, self.multiplier), None)

      @classmethod
      def tree_unflatten(cls, aux, values):
        del aux
        return cls(*values)

      @property
      def shape(self) -> list[int]:
        return self.param.shape

    class CustomLinear:

      def __init__(self, *args, **kwargs):
        self.linear = basic.Linear(*args, **kwargs)

      def __call__(self, x: CustomParam) -> CustomParam:
        # Unwrap from CustomParam before invoking linear
        return CustomParam(
            self.linear(x.param * x.multiplier),
            x.multiplier,
        )

    def block(x: CustomParam, i: int) -> CustomParam:
      return CustomLinear(output_size=x.shape[-1], name=f"linear_{i}")(x)

    def looped(x: CustomParam, num_layers: int = 1) -> CustomParam:
      for i in range(num_layers):
        x = block(x, i)
      return x

    def stacked(x: CustomParam) -> CustomParam:
      return layer_stack.layer_stack(
          num_layers=1, transparent=True, transparency_map=TransparencyMap()
      )(lambda y: block(y, 0))(x)

    looped = transform.transform(looped)
    stacked = transform.transform(stacked)

    x = CustomParam(jnp.ones((2, 2)), 0.3)
    rng = jax.random.PRNGKey(0)
    looped_params = looped.init(rng, x)
    stacked_params = stacked.init(rng, x)

    self.assertEqual(
        jax.tree.structure(looped_params),
        jax.tree.structure(stacked_params),
    )

    # Use same set of params for both calls since stacked_params have different
    # value than looped params because differences in RNG splitting.
    np.testing.assert_allclose(
        looped.apply(looped_params, rng, x).param,
        stacked.apply(looped_params, rng, x).param,
        rtol=1e-6,
    )


if __name__ == "__main__":
  jax.config.update("jax_check_tracer_leaks", True)
  jax.config.update("jax_default_matmul_precision", "float32")
  absltest.main()
