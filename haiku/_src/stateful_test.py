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
"""Tests for haiku._src.stateful."""

import itertools as it

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import base
from haiku._src import base_test
from haiku._src import config
from haiku._src import initializers
from haiku._src import module
from haiku._src import stateful
from haiku._src import test_utils
from haiku._src import transform

import jax
from jax.extend import core as jax_core
import jax.numpy as jnp
import numpy as np

toggle = lambda i, a: lambda x: a(x) if base.params_frozen() else i(x)


# JAX transforms and control flow that need to be aware of Haiku internal
# state to operate unsurprisingly.
# pylint: disable=g-long-lambda
HK_OVERLOADED_JAX_PURE_EXPECTING_FNS = (
    # Just-in-time compilation.
    ("jit", stateful.jit),

    # ("make_jaxpr", stateful.make_jaxpr),
    ("eval_shape", lambda f: (lambda x: [f(x), stateful.eval_shape(f, x)])),

    # Parallelization.
    # TODO(tomhennigan): Add missing features (e.g. pjit,xmap).
    # ("pmap", lambda f: stateful.pmap(f, "i")),

    # Vectorization.
    ("vmap", lambda f: stateful.vmap(f, split_rng=False)),

    # Control flow.
    # TODO(tomhennigan): Enable for associative_scan.
    # ("associative_scan", lambda f:
    #  (lambda x: jax.lax.associative_scan(f, x))),
    ("cond", lambda f: (lambda x: stateful.cond(True, f, f, x))),
    ("fori_loop", lambda f:
     (lambda x: stateful.fori_loop(0, 1, base_test.ignore_index(f), x))),
    # ("map", lambda f: (lambda x: stateful.map(f, x))),
    ("scan", lambda f:
     (lambda x: stateful.scan(base_test.identity_carry(f), None, x))),
    ("switch", lambda f: (lambda x: stateful.switch(0, [f, f], x))),
    ("while_loop", lambda f: toggle(
        f, lambda x: stateful.while_loop(lambda xs: xs[0] == 0,
                                         lambda xs: (1, f(xs[1])),
                                         (0, x)))),

    # Automatic differentiation.
    # TODO(tomhennigan): Add missing features (e.g. custom_vjp, custom_jvp).

    ("grad", lambda f: stateful.grad(lambda x: f(x).sum())),
    ("value_and_grad", lambda f: stateful.value_and_grad(lambda x: f(x).sum())),
    ("checkpoint", stateful.remat),
)
# pylint: enable=g-long-lambda


def with_rng_reserve_size(f):
  """Run test with rng_reserve_size of 7."""
  def wrapper(*a, **kw):
    with config.context(rng_reserve_size=7):
      return f(*a, **kw)
  return wrapper


class StatefulTest(parameterized.TestCase):

  def assert_keys_equal(self, a, b):
    self.assertEqual(jax.random.key_impl(a), jax.random.key_impl(b))
    np.testing.assert_array_equal(
        jax.random.key_data(a), jax.random.key_data(b)
    )

  def assert_keys_not_equal(self, a, b):
    self.assertFalse(
        (jax.random.key_impl(a) == jax.random.key_impl(b)) and
        (jnp.all(jax.random.key_data(a) == jax.random.key_data(b))))

  @test_utils.transform_and_run
  def test_grad(self):
    x = jnp.array(3.0)
    g = stateful.grad(SquareModule())(x)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-4)

  def test_grad_no_transform(self):
    x = jnp.array(3.0)
    with self.assertRaises(ValueError, msg="Use jax.grad() instead"):
      stateful.grad(jnp.square)(x)

  @test_utils.transform_and_run
  def test_value_and_grad(self):
    x = jnp.array(2.0)
    y, g = stateful.value_and_grad(SquareModule())(x)
    self.assertEqual(y, x**2)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-4)

  def test_value_and_grad_no_transform(self):
    x = jnp.array(3.0)
    with self.assertRaises(ValueError, msg="Use jax.grad() instead"):
      stateful.value_and_grad(jnp.square)(x)

  @test_utils.transform_and_run
  def test_grad_aux(self):
    o = object()

    def f(x):
      m = SquareModule()
      return m(x), o

    x = jnp.array(3.)
    g, aux = stateful.grad(f, has_aux=True)(x)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-4)
    self.assertIs(aux, o)

  @test_utils.transform_and_run
  def test_value_and_grad_aux(self):
    o = object()

    def f(x):
      m = SquareModule()
      return m(x), o

    x = jnp.array(3.)
    (y, aux), g = stateful.value_and_grad(f, has_aux=True)(x)
    self.assertEqual(y, jnp.power(x, 2))
    np.testing.assert_allclose(g, 2 * x, rtol=1e-4)
    self.assertIs(aux, o)

  def test_grad_and_jit(self):
    def f(x):
      g = stateful.grad(SquareModule())(x)
      return g

    x = jnp.array(3.)
    f = transform.transform_with_state(f)
    params, state = jax.jit(f.init)(None, x)
    g, state = jax.jit(f.apply)(params, state, None, x)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-3)

  def test_value_and_grad_and_jit(self):
    def f(x):
      y, g = stateful.value_and_grad(SquareModule())(x)
      return y, g

    x = jnp.array(3.)
    f = transform.transform_with_state(f)
    params, state = jax.jit(f.init)(None, x)
    (y, g), state = jax.jit(f.apply)(params, state, None, x)
    np.testing.assert_allclose(y, x ** 2, rtol=1e-3)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-3)

  @test_utils.transform_and_run
  def test_jit(self):
    mod = SquareModule()
    x = jnp.array(2)
    y = stateful.jit(mod)(x)
    self.assertEqual(y, x ** 2)

  def test_jit_no_transform(self):
    x = jnp.array(2)
    with self.assertRaises(ValueError, msg="Use jax.jit() instead"):
      stateful.jit(jnp.square)(x)

  @test_utils.transform_and_run
  def test_remat(self):
    forward, backward = [], []
    callback = _callback_prim(lambda: forward.append(None),
                              lambda: backward.append(None))

    def test(remat):
      x = jnp.array(3.)
      mod = CountingModule()
      self.assertEqual(mod.count, 0)
      f = lambda x: callback(mod(x))
      if remat:
        f = stateful.remat(f)
      y, g = stateful.value_and_grad(f)(x)
      np.testing.assert_allclose(y, x ** 2, rtol=1e-3)
      np.testing.assert_allclose(g, 2 * x, rtol=1e-3)
      self.assertEqual(mod.count, 1)
      num_forward = len(forward)
      num_backward = len(backward)
      del forward[:], backward[:]
      return num_forward, num_backward

    # Sanity check.
    self.assertEqual(test(remat=True), test(remat=True))
    self.assertEqual(test(remat=False), test(remat=False))

    # NOTE: JAX does not guarantee to execute primitives once and only once for
    # a given function (we observe f=2,b=1 without remat and f=5,b=1 with
    # remat), but we do expect that JAX will execute our primitive forward at
    # least one more time with remat than without it.
    num_forward_remat, num_backward_remat = test(remat=True)
    num_forward_no_remat, num_backward_no_remat = test(remat=False)
    self.assertGreater(num_forward_remat, num_forward_no_remat)
    self.assertEqual(num_backward_remat, num_backward_no_remat)

  def test_remat_no_transform(self):
    x = jnp.array(3.)
    with self.assertRaises(ValueError, msg="Use jax.remat() instead"):
      stateful.remat(jnp.square)(x)

  @test_utils.combined_named_parameters(
      test_utils.named_bools("jax_remat"),
      test_utils.named_bools("inline_hk_remat"))
  def test_create_module_inside_remat(self, jax_remat, inline_hk_remat):
    log = []
    def forward(x):
      def create_and_use_layer(x):
        m = SquareModule(name="layer")
        log.append(m.module_name)
        return m(x)

      if not inline_hk_remat:
        create_and_use_layer = stateful.remat(create_and_use_layer)

      for _ in range(2):
        if inline_hk_remat:
          x = stateful.remat(create_and_use_layer)(x)
        else:
          x = create_and_use_layer(x)
      return x

    def reset():
      del log[:]
      self.assertEmpty(log)

    # Test forward.
    x = jnp.float32(3)
    forward = transform.transform_with_state(forward)
    params, state = forward.init(None, x)
    self.assertEqual(log, ["layer", "layer_1"])
    reset()

    # Test backward.
    for _ in range(3):
      grad_fn = jax.grad(lambda x: forward.apply(params, state, None, x)[0])
      if jax_remat:
        grad_fn = jax.remat(grad_fn)
      self.assertEqual(int(grad_fn(x)), int(4 * (x ** 3)))
      self.assertEqual(log, ["layer", "layer_1"])
      reset()

  @parameterized.parameters(True, False)
  def test_cond(self, single_arg):
    def f(x):
      mod = SquareModule()
      if single_arg:
        return stateful.cond(x == 2, mod, lambda x: mod(x + 1), x)
      else:
        return stateful.cond(x == 2, x, mod, x, lambda x: mod(x + 1))

    f = transform.transform_with_state(f)
    for x, y in ((1, 4), (2, 4), (3, 16)):
      x, y = map(jnp.array, (x, y))
      params, state = f.init(None, x)
      out, state = f.apply(params, state, None, x)
      self.assertEqual(state, {"square_module": {"y": y}})
      self.assertEqual(out, y)

  @test_utils.transform_and_run
  def test_cond_traces_branches_with_same_id_once(self):
    witness = []
    def f(x):
      witness.append(None)
      return x ** 2

    stateful.cond(False, f, f, 0)
    hk_call_count = len(witness)
    self.assertEqual(hk_call_count, 1)

    # Ensure we are in sync with JAX.
    del witness[:]
    jax.lax.cond(False, f, f, 0)
    jax_call_count = len(witness)
    self.assertEqual(hk_call_count, jax_call_count)

  @test_utils.transform_and_run
  def test_cond_no_args(self):
    x = stateful.cond(True, lambda: 5, lambda: 4)
    self.assertEqual(x, 5)

  @test_utils.transform_and_run
  def test_cond_operand_kwarg(self):
    x = stateful.cond(True, lambda x: x + 5, lambda x: x + 4, operand=1)
    self.assertEqual(x, 6)

  @test_utils.transform_and_run
  def test_cond_operand_kwarg_and_operands(self):
    with self.assertRaisesRegex(ValueError, "cannot.*pass.*positionally"):
      stateful.cond(True, lambda x: x + 5, lambda x: x + 4, 1, operand=1)

  @test_utils.transform_and_run
  def test_cond_two_args(self):
    a, b = stateful.cond(True,
                         lambda a, b: (b, a),
                         lambda a, b: (a, b),
                         2, 1)
    self.assertEqual(a, 1)
    self.assertEqual(b, 2)

  @test_utils.transform_and_run
  def test_cond_three_args(self):
    a, b, c = stateful.cond(True,
                            lambda a, b, c: (c, b, a),
                            lambda a, b, c: (a, b, c),
                            3, 2, 1)
    self.assertEqual(a, 1)
    self.assertEqual(b, 2)
    self.assertEqual(c, 3)

  def test_cond_no_transform(self):
    x = jnp.array(3.)
    with self.assertRaises(ValueError, msg="Use jax.cond() instead"):
      stateful.cond(x == 2, x, jnp.square, x, lambda x: jnp.square(x + 1))

  def test_switch(self):
    def f(i, x):
      mod = SquareModule()
      branches = [mod, lambda x: mod(x + 1), lambda x: mod(x + 2)]
      return stateful.switch(i, branches, x)

    f = transform.transform_with_state(f)
    for i, x, y in ((0, 1, 1), (1, 2, 9), (2, 3, 25)):
      i, x, y = map(jnp.array, (i, x, y))
      params, state = f.init(None, i, x)
      out, state = f.apply(params, state, None, i, x)
      self.assertEqual(state, {"square_module": {"y": y}})
      self.assertEqual(out, y)

  def test_switch_multiple_operands(self):
    def f(i, x, y, z):
      mod = SquareModule()
      branches = [lambda x, y, z: mod(x),
                  lambda y, x, z: mod(x),
                  lambda y, z, x: mod(x),
                  ]
      return stateful.switch(i, branches, x, y, z)

    f = transform.transform_with_state(f)
    xyz = (1, 3, 5)
    for i in range(3):
      params, state = f.init(None, i, *xyz)
      out, state = f.apply(params, state, None, i, *xyz)
      expected_out = xyz[i]**2
      self.assertEqual(state, {"square_module": {"y": expected_out}})
      self.assertEqual(out, expected_out)

  @test_utils.transform_and_run(run_apply=False)
  def test_cond_branch_structure_error(self):
    true_fn = lambda x: base.get_parameter("w", x.shape, x.dtype, init=jnp.ones)
    false_fn = lambda x: x
    with self.assertRaisesRegex(
        TypeError, "cond branch outputs must have the same pytree structure"
    ):
      stateful.cond(False, true_fn, false_fn, 0)

  @test_utils.transform_and_run(run_apply=False)
  def test_switch_branch_structure_error(self):
    branches = [
        lambda x: base.get_parameter("w", x.shape, x.dtype, init=jnp.ones),
        lambda x: x,
    ]
    with self.assertRaisesRegex(
        TypeError, "switch branch outputs must have the same pytree structure"
    ):
      stateful.switch(0, branches, 0)

  @parameterized.parameters(1, 2, 4, 8)
  @test_utils.transform_and_run
  def test_switch_traces_cases_with_same_id_once(self, n):
    f_witness = []
    g_witness = []

    def f(x):
      f_witness.append(None)
      return x ** 2

    def g(x):
      g_witness.append(None)
      return x ** 2

    stateful.switch(0, [f, g] * n, 2)
    f_hk_call_count = len(f_witness)
    g_hk_call_count = len(g_witness)
    self.assertEqual(f_hk_call_count, 1)
    self.assertEqual(g_hk_call_count, 1)

    # Ensure we are in sync with JAX.
    del f_witness[:], g_witness[:]
    jax.lax.switch(0, [f, g] * n, 2)
    f_jax_call_count = len(f_witness)
    g_jax_call_count = len(g_witness)
    self.assertEqual(f_hk_call_count, f_jax_call_count)
    self.assertEqual(f_hk_call_count, g_jax_call_count)

  def test_switch_no_transform(self):
    i = jnp.array(2)
    x = jnp.array(42.)
    with self.assertRaises(ValueError, msg="Use jax.switch() instead"):
      stateful.switch(i, [jnp.square] * 3, x)

  @test_utils.transform_and_run
  def test_difference_empty(self):
    before = stateful.internal_state()
    after = stateful.internal_state()
    self.assertEmpty(jax.tree.leaves(stateful.difference(before, after)))

  @parameterized.parameters(base.get_parameter, base.get_state)
  @test_utils.transform_and_run(run_apply=False)
  def test_difference_new(self, get_x):
    get_x("a", [], init=jnp.zeros)
    before = stateful.internal_state()
    b = get_x("b", [], init=jnp.zeros)
    after = stateful.internal_state()
    diff = stateful.difference(before, after)
    if get_x == base.get_state:
      self.assertEmpty(diff.params)
      self.assertEqual(diff.state, {"~": {"a": None,
                                          "b": base.StatePair(b, b)}})
    else:
      self.assertEqual(diff.params, {"~": {"a": None, "b": b}})
      self.assertEmpty(diff.state)
    self.assertIsNone(diff.rng)

  @test_utils.transform_and_run(run_apply=False)
  def test_difference_update_state(self):
    base.get_state("a", [], init=jnp.zeros)
    base.get_state("b", [], init=jnp.zeros)
    before = stateful.internal_state()
    base.set_state("b", jnp.ones([]))
    after = stateful.internal_state()
    diff = stateful.difference(before, after)
    self.assertEmpty(diff.params)
    self.assertEqual(diff.state, {"~": {"a": None,
                                        "b": base.StatePair(0., 1.)}})
    self.assertIsNone(diff.rng)

  @test_utils.transform_and_run(run_apply=False)
  def test_difference_rng(self):
    before = stateful.internal_state()
    base.next_rng_key()
    after = stateful.internal_state()
    diff = stateful.difference(before, after)
    self.assertEmpty(diff.params)
    self.assertEmpty(diff.state)
    self.assertIsNotNone(diff.rng)

  def test_scan_no_transform(self):
    xs = jnp.arange(3)
    with self.assertRaises(ValueError, msg="Use jax.lax.scan() instead"):
      stateful.scan(lambda c, x: (c, x), (), xs)

  @parameterized.parameters(0, 1, 2, 4, 8)
  def test_scan_with_state(self, unroll_length):
    def f(xs):
      m = CountingModule()
      def sf(c, x):
        self.assertEqual(c, ())
        return c, m(x)
      _, ys = stateful.scan(sf, (), xs)
      return ys

    f = transform.transform_with_state(f)
    key = jax.random.PRNGKey(42)
    init_key, apply_key = jax.random.split(key)
    xs = jnp.arange(unroll_length)
    params, state = f.init(init_key, xs)
    self.assertEqual(list(state), ["counting_module"])
    self.assertEqual(list(state["counting_module"]), ["count"])
    np.testing.assert_allclose(state["counting_module"]["count"], 0, rtol=1e-4)
    ys, state = f.apply(params, state, apply_key, xs)
    np.testing.assert_allclose(state["counting_module"]["count"], unroll_length,
                               rtol=1e-4)
    np.testing.assert_allclose(ys, xs ** 2, rtol=1e-4)

  @parameterized.parameters(0, 1, 2, 8)
  @test_utils.transform_and_run
  @with_rng_reserve_size
  def test_stateful_scan_with_rng_use(self, iteration_count):
    def body_fun(c, x):
      for _ in range(10):
        _ = base.next_rng_key()
      return c, x
    base.reserve_rng_keys(5)
    _ = stateful.scan(body_fun, (), (), length=iteration_count)

  @parameterized.parameters(0, 1, 2, 8)
  @test_utils.transform_and_run
  @with_rng_reserve_size
  def test_stateful_fori_with_rng_use(self, iteration_count):
    def body_fun(_, x):
      for _ in range(10):
        _ = base.next_rng_key()
      return x
    base.reserve_rng_keys(5)
    _ = stateful.fori_loop(0, iteration_count, body_fun, 1)

  @test_utils.transform_and_run
  @with_rng_reserve_size
  def test_stateful_cond_with_rng_use(self):
    # Test if using different amount of keys in different branches
    # results in error
    def true_branch(x):
      _ = base.next_rng_key()
      return x

    def false_branch(x):
      _ = base.next_rng_key()
      _ = base.next_rng_key()
      return x

    base.reserve_rng_keys(5)
    _ = stateful.cond(True, true_branch, false_branch, 0)
    _ = stateful.cond(False, true_branch, false_branch, 0)

  @test_utils.transform_and_run
  @with_rng_reserve_size
  def test_stateful_switch_with_rng_use(self):
    # Test if using different amount of keys in different branches
    # results in error
    def branch_f(i):
      for _ in range(i):
        _ = base.next_rng_key()
      return i

    base.reserve_rng_keys(5)
    branches = [lambda _, i=i: branch_f(i) for i in range(5)]
    self.assertEqual(stateful.switch(3, branches, None), 3)
    self.assertEqual(stateful.switch(0, branches, None), 0)

  @test_utils.transform_and_run
  def test_stateful_while_loop_with_rng_use(self):
    def body_fun(i):
      _ = base.next_rng_key()
      _ = base.next_rng_key()
      return i+1

    base.reserve_rng_keys(5)
    if transform.running_init():
      body_fun(0)
    else:
      stateful.while_loop(lambda i: i < 7, body_fun, 0)  # does not crash.

  @parameterized.parameters(*it.product((0, 1, 2, 4, 8), (1, 2, 3)))
  @test_utils.transform_and_run
  def test_fori(self, lower, n):
    upper = lower + n
    m = CountingModule()
    y = stateful.fori_loop(lower, upper, lambda i, x: m(i), 2)
    self.assertEqual(y, jnp.square(upper - 1))
    self.assertEqual(m.count, upper - lower)

  @test_utils.transform_and_run
  def test_fori_traced_length(self):
    m = CountingModule()

    def f(lower, upper):
      y = stateful.fori_loop(lower, upper, lambda i, x: m(i), 2)
      return y

    # Because of the jit, lower and upper will be tracers.
    out = stateful.jit(f)(0, 3)
    self.assertEqual(out, 4)
    self.assertEqual(m.count, 3)

  @test_utils.transform_and_run
  def test_map(self):
    x = np.zeros((10, 10), dtype=np.float32)

    def f(x):
      self.assertLen(x.shape, 1)
      return x + jax.random.uniform(base.next_rng_key())

    if transform.running_init():
      f(x[0])
    else:
      stateful.map(f, x)

  def test_vmap(self):
    def g(x):
      return CountingModule()(x)

    def f(x):
      return stateful.vmap(g, split_rng=False)(x)

    f = transform.transform_with_state(f)

    x = jnp.ones([4]) + 1
    params, state = f.init(None, x)

    # State should not be mapped.
    self.assertEmpty(params)
    (cnt,) = jax.tree.leaves(state)
    self.assertEqual(cnt.ndim, 0)
    self.assertEqual(cnt, 0)

    # The output should be mapped but state should not be.
    y, state = f.apply(params, state, None, x)
    self.assertEqual(y.shape, (4,))
    np.testing.assert_allclose(y, x ** 2)
    (cnt,) = jax.tree.leaves(state)
    self.assertEqual(cnt.ndim, 0)
    self.assertEqual(cnt, 1)

  def test_vmap_must_be_called_in_transform(self):
    f = stateful.vmap(lambda x: x, split_rng=False)
    with self.assertRaisesRegex(ValueError,
                                "must be used as part of an.*hk.transform"):
      f(0)

  @test_utils.transform_and_run
  def test_vmap_no_in_axes(self):
    def fn_name(_):
      pass
    with self.assertRaisesRegex(
        ValueError, "fn_name must have at least one non-None value in in_axes"):
      stateful.vmap(fn_name, in_axes=None, split_rng=False)

  @test_utils.transform_and_run
  def test_vmap_in_axes_different_size(self):
    x = jnp.ones([1, 2])
    with self.assertRaisesRegex(
        ValueError, "vmap got inconsistent sizes for array axes to be mapped"):
      stateful.vmap(lambda a, b: None, in_axes=(0, 1), split_rng=False)(x, x)

  @test_utils.transform_and_run
  def test_vmap_in_axes_supports_list(self):
    a = jnp.ones([4])
    b = stateful.vmap(lambda a: a * 2, in_axes=[0], split_rng=False)(a)
    np.testing.assert_array_equal(b, a * 2)

  @test_utils.transform_and_run
  def test_vmap_no_split_rng(self):
    key_before = base.next_rng_key()
    f = stateful.vmap(lambda _: base.next_rng_key(), split_rng=False)
    x = jnp.arange(4)
    k1, k2, k3, k4 = f(x)
    key_after = base.next_rng_key()
    self.assert_keys_equal(k1, k2)
    self.assert_keys_equal(k2, k3)
    self.assert_keys_equal(k3, k4)
    self.assert_keys_not_equal(key_before, k1)
    self.assert_keys_not_equal(key_after, k1)
    self.assert_keys_not_equal(key_before, key_after)

  @test_utils.transform_and_run
  def test_vmap_split_rng(self):
    key_before = base.next_rng_key()
    f = stateful.vmap(lambda _: base.next_rng_key(), split_rng=True)
    x = jnp.arange(4)
    k1, k2, k3, k4 = f(x)
    key_after = base.next_rng_key()
    # Test that none of the keys are equal.
    named_keys = (("k1", k1), ("k2", k2), ("k3", k3), ("k4", k4),
                  ("key_before", key_before), ("key_after", key_after))
    for (a_name, a), (b_name, b) in it.combinations(named_keys, 2):
      self.assertFalse(
          np.array_equal(a, b),
          msg=f"Keys should not be equal, but {a_name} == {b_name}")

  @test_utils.transform_and_run(run_apply=False)
  def test_vmap_split_rng_better_out_axes_error(self):
    def creates_params(_):
      base.get_parameter("mapped",
                         (), jnp.float32,
                         init=initializers.TruncatedNormal())
    f = stateful.vmap(creates_params, split_rng=True)
    x = jnp.arange(4)
    with self.assertRaisesRegex(ValueError,
                                "split_rng to True during initialization"):
      f(x)

  @test_utils.transform_and_run(run_apply=False)
  def test_vmap_split_rng_out_axes_error_no_split_rng(self):
    f = stateful.vmap(lambda x: x, split_rng=False, out_axes=None)
    x = jnp.arange(4)
    with self.assertRaisesRegex(ValueError, ".*vmap.*out_axes.*None.*"):
      # test our split_rng error does not clobber jax error message.
      f(x)

  def test_vmap_split_rng_out_axes_error_no_init(self):
    @transform.transform
    def g(x):
      f = stateful.vmap(lambda x: x, split_rng=True, out_axes=None)
      f(x)

    x = jnp.arange(4)
    with self.assertRaisesRegex(ValueError, ".*vmap.*out_axes.*None.*"):
      # test our split_rng error does not clobber jax error message.
      g.apply({}, jax.random.PRNGKey(42), x)

  def test_while_loop_rejected_in_init(self):
    def f():
      stateful.while_loop(lambda x: x.all(), lambda x: not x, 1)
    f = transform.transform(f)
    with self.assertRaisesRegex(
        ValueError, "hk.while_loop does not support initialization"):
      f.init(None)

  def test_updating_state_in_cond_fails(self):
    def f(x):
      m = CountingModule(op=lambda x: x + 1)
      if not base.params_frozen():
        return m(x)
      else:
        stateful.while_loop(m, lambda x: x, x)

    f = transform.transform_with_state(f)
    x = jnp.zeros([])
    params, state = f.init(None, x)
    with self.assertRaisesRegex(
        ValueError,
        "does not support.*set_state.*next_rng_key.*in.*cond_fun`"):
      f.apply(params, state, None, x)

  def test_rng_in_cond_fails(self):
    def f(x):
      m = CountingModule(op=lambda x: x + 1)
      if not base.params_frozen():
        return m(x)
      else:
        stateful.while_loop(lambda _: base.next_rng_key(), lambda x: x, x)

    f = transform.transform_with_state(f)
    x = jnp.zeros([])
    params, state = f.init(None, x)
    with self.assertRaisesRegex(
        ValueError,
        "does not support.*set_state.*next_rng_key.*in.*cond_fun`"):
      f.apply(params, state, jax.random.PRNGKey(42), x)

  @parameterized.parameters(0, 1, 2, 4, 8)
  def test_while_loop_with_state(self, iters):
    def f(x):
      m = CountingModule(op=lambda x: x + 1)
      if not base.params_frozen():
        return m(x)
      else:
        _, y = stateful.while_loop(lambda a: a[0] < iters,
                                   lambda a: (a[0] + 1, m(a[1])),
                                   (0, x))
        return y

    f = transform.transform_with_state(f)
    x = jnp.zeros([])
    params, state = f.init(None, x)
    self.assertEqual(list(state), ["counting_module"])
    self.assertEqual(list(state["counting_module"]), ["count"])
    np.testing.assert_allclose(state["counting_module"]["count"], x, rtol=1e-4)

    y, state = f.apply(params, state, None, x)
    np.testing.assert_allclose(state["counting_module"]["count"], iters,
                               rtol=1e-4)
    np.testing.assert_allclose(y, iters, rtol=1e-4)

  def test_eval_shape(self):
    def some_shape_changing_fun(x):
      return x[0, :]

    def f(x):
      m = CountingModule(op=some_shape_changing_fun)
      # state is not changed in this call
      out_shape_struct = stateful.eval_shape(m, x)
      return m(x), out_shape_struct

    f = transform.transform_with_state(f)
    key = jax.random.PRNGKey(42)
    in_shape = (10, 10)
    x = jnp.ones(in_shape)
    params, state = f.init(key, x)
    self.assertEqual(list(state), ["counting_module"])
    self.assertEqual(list(state["counting_module"]), ["count"])
    np.testing.assert_allclose(state["counting_module"]["count"], 0, rtol=1e-4)
    (out, shape_struct), state = f.apply(params, state, key, x)
    # Count is only advanced once
    np.testing.assert_allclose(state["counting_module"]["count"], 1, rtol=1e-4)
    np.testing.assert_allclose(out, some_shape_changing_fun(x), rtol=1e-4)
    self.assertEqual(shape_struct.shape, (in_shape[1],))

  def test_eval_shape_no_transform(self):
    x = jnp.array(3.)
    with self.assertRaises(ValueError, msg="Use jax.eval_shape() instead"):
      stateful.eval_shape(jnp.square)(x)

  @test_utils.transform_and_run
  def test_temporary_state_resets_names(self):
    with stateful.temporary_internal_state(stateful.internal_state()):
      mod1 = module.Module(name="foo")
    mod2 = module.Module(name="foo")
    self.assertEqual(mod1.module_name, "foo")
    self.assertEqual(mod2.module_name, "foo")

  @test_utils.transform_and_run(run_apply=False)
  def test_eval_shape_no_leaked_tracers_under_leak_checker(self):
    with jax.checking_leaks():
      stateful.eval_shape(SquareModule(), jnp.ones(()))  # does not crash

  @test_utils.combined_named_parameters(base_test.SIDE_EFFECTING_FUNCTIONS,
                                        HK_OVERLOADED_JAX_PURE_EXPECTING_FNS)
  @test_utils.transform_and_run
  @test_utils.with_guardrails
  def test_safe_use_of_jax(self, haiku_side_effect_fn, hk_jax_fn):
    if "reserve_rng_keys_while_loop" in self._testMethodName:
      self.skipTest("Expected not to work.")

    # Make `f` identify with the side effecting function included.
    f = hk_jax_fn(lambda x: [haiku_side_effect_fn(), x][1])
    x = jnp.ones([1])
    # These functions should not trigger exceptions from our guardrails.
    f(x)

  @test_utils.transform_and_run
  def test_vmap_split_rng_with_default(self):
    with self.assertRaisesRegex(TypeError,
                                "hk.vmap.require_split_rng = False"):
      # Intentionally missing split_rng arg.
      stateful.vmap(lambda: None)

    with self.subTest("require_split_rng=0"):
      stateful.vmap.require_split_rng = False
      try:
        # This call should not trigger an error, even though we are missing the
        # split_rng argument which appears required (if you look at the function
        # signature). It only works because require_split_rng is
        # propagated to vmap via a sneaky decorator. This only exists to support
        # users who import code that they cannot edit (e.g. from a read only
        # file system) that is not passing the argument.
        f = stateful.vmap(base.next_rng_key, axis_size=2)
      finally:
        stateful.vmap.require_split_rng = True

    # Check that split_rng=False was implied.
    k1, k2 = f()
    self.assertTrue((k1 == k2).all())

  @parameterized.parameters(True, False)
  @test_utils.transform_and_run
  def test_vmap_split_rng_without_default(self, require_split_rng):
    # Tests that when split_rng is passed explicitly the value of
    # require_split_rng has no impact.
    x = jnp.arange(2)
    stateful.vmap.require_split_rng = require_split_rng
    k1, k2 = stateful.vmap(lambda x: base.next_rng_key(), split_rng=True)(x)
    self.assertTrue((k1 != k2).all())
    k1, k2 = stateful.vmap(lambda x: base.next_rng_key(), split_rng=False)(x)
    self.assertTrue((k1 == k2).all())
    stateful.vmap.require_split_rng = True


def _callback_prim(forward, backward):
  def f_impl(x):
    forward()
    return x

  def b_impl(x):
    backward()
    return (x,)

  prim = jax_core.Primitive("hk_callback")
  prim.def_impl(f_impl)
  prim.def_abstract_eval(f_impl)
  jax.interpreters.ad.deflinear(prim, b_impl)
  return prim.bind


class CountingModule(module.Module):

  def __init__(self, op=jnp.square, name=None):
    super().__init__(name=name)
    self.op = op

  @property
  def count(self):
    return base.get_state("count", [], init=jnp.zeros)

  def __call__(self, x):
    y = self.op(x)
    base.set_state("count", self.count + 1)
    return y


class SquareModule(module.Module):

  def __call__(self, x):
    assert x.ndim == 0
    p = base.get_parameter("p", [], jnp.int32, init=lambda *_: jnp.array(2))
    y = x ** p
    base.set_state("y", y)
    return y


if __name__ == "__main__":
  absltest.main()
