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

import functools
import itertools as it

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import base
from haiku._src import module
from haiku._src import stateful
from haiku._src import test_utils
from haiku._src import transform

import jax
import jax.numpy as jnp
import numpy as np


class StatefulTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_grad(self):
    x = jnp.array(3.)
    g = stateful.grad(SquareModule())(x)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-4)

  def test_grad_no_transform(self):
    x = jnp.array(3.)
    with self.assertRaises(ValueError, msg="Use jax.grad() instead"):
      stateful.grad(jnp.square)(x)

  @test_utils.transform_and_run
  def test_value_and_grad(self):
    x = jnp.array(2.)
    y, g = stateful.value_and_grad(SquareModule())(x)
    self.assertEqual(y, x ** 2)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-4)

  def test_value_and_grad_no_transform(self):
    x = jnp.array(3.)
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
    self.assertEmpty(jax.tree_leaves(stateful.difference(before, after)))

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
    with self.assertRaises(ValueError, msg="Use jax.scan() instead"):
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
    xs = jnp.arange(unroll_length)
    params, state = f.init(key, xs)
    self.assertEqual(list(state), ["counting_module"])
    self.assertEqual(list(state["counting_module"]), ["count"])
    np.testing.assert_allclose(state["counting_module"]["count"], 0, rtol=1e-4)
    ys, state = f.apply(params, state, key, xs)
    np.testing.assert_allclose(state["counting_module"]["count"], unroll_length,
                               rtol=1e-4)
    np.testing.assert_allclose(ys, xs ** 2, rtol=1e-4)

  @parameterized.parameters(0, 1, 2, 8)
  @test_utils.transform_and_run
  def test_stateful_scan_with_rng_use(self, iteration_count):
    # TODO(lenamartens): remove when default changes to > 1.
    tmp_default = base.DEFAULT_PRNG_RESERVE_SIZE
    base.DEFAULT_PRNG_RESERVE_SIZE = 64
    def body_fun(c, x):
      for _ in range(10):
        _ = base.next_rng_key()
      return c, x
    base.reserve_rng_keys(5)
    _ = stateful.scan(body_fun, (), (), length=iteration_count)
    base.DEFAULT_PRNG_RESERVE_SIZE = tmp_default

  @parameterized.parameters(0, 1, 2, 8)
  @test_utils.transform_and_run
  def test_stateful_fori_with_rng_use(self, iteration_count):
    tmp_default = base.DEFAULT_PRNG_RESERVE_SIZE
    base.DEFAULT_PRNG_RESERVE_SIZE = 64
    def body_fun(_, x):
      for _ in range(10):
        _ = base.next_rng_key()
      return x
    base.reserve_rng_keys(5)
    _ = stateful.fori_loop(0, iteration_count, body_fun, 1)
    base.DEFAULT_PRNG_RESERVE_SIZE = tmp_default

  @test_utils.transform_and_run
  def test_stateful_cond_with_rng_use(self):
    tmp_default = base.DEFAULT_PRNG_RESERVE_SIZE
    base.DEFAULT_PRNG_RESERVE_SIZE = 64
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
    base.DEFAULT_PRNG_RESERVE_SIZE = tmp_default

  @test_utils.transform_and_run
  def test_stateful_switch_with_rng_use(self):
    tmp_default = base.DEFAULT_PRNG_RESERVE_SIZE
    base.DEFAULT_PRNG_RESERVE_SIZE = 64
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
    base.DEFAULT_PRNG_RESERVE_SIZE = tmp_default

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

  def test_vmap(self):
    def g(x):
      return CountingModule()(x)

    def f(x):
      return stateful.vmap(g)(x)

    f = transform.transform_with_state(f)

    x = jnp.ones([4]) + 1
    params, state = f.init(None, x)

    # State should not be mapped.
    self.assertEmpty(params)
    cnt, = jax.tree_leaves(state)
    self.assertEqual(cnt.ndim, 0)
    self.assertEqual(cnt, 0)

    # The output should be mapped but state should not be.
    y, state = f.apply(params, state, None, x)
    self.assertEqual(y.shape, (4,))
    np.testing.assert_allclose(y, x ** 2)
    cnt, = jax.tree_leaves(state)
    self.assertEqual(cnt.ndim, 0)
    self.assertEqual(cnt, 1)

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

  def test_named_call(self):
    def f(x):
      return stateful.named_call(SquareModule(), name="square")(x)

    x = jnp.array(2.)
    rng = jax.random.PRNGKey(42)
    init, apply = transform.transform_with_state(f)
    params, state = init(rng, x)
    y, state = jax.jit(apply)(params, state, rng, x)
    self.assertEqual(y, x ** 2)

  @parameterized.parameters(jax.jit, jax.grad, jax.vmap, jax.remat)
  def test_named_call_jax_transforms(self, jax_transform):
    f = jnp.sum
    x = jnp.array([1.])

    unnamed_out = jax_transform(f)(x)
    named_out = jax_transform(stateful.named_call(f, name="test"))(x)

    self.assertEqual(unnamed_out, named_out)

  def test_static_argnums_named_call(self):
    f = stateful.named_call(lambda x, y: y if x else None,
                            name="test")
    f = jax.jit(f, static_argnums=(0,))
    out = f(True, 5)
    self.assertEqual(out, 5)

  def test_named_call_non_jaxtype_arg(self):
    # For the test to fail without the invalid JaxType filter we need to pass
    # in a valid JaxType that forces the invalid Jaxtype to be raised to an
    # abstract value.
    def f(not_a_jaxtype, a_jaxtype):
      # then Jax needs to try and evaluate the abstractified non-JaxType
      if not_a_jaxtype:
        return a_jaxtype
      return 0

    f = stateful.named_call(f, name="test")
    out = jax.jit(f, static_argnums=(0,))("not a Jaxtype", 1)
    self.assertEqual(out, 1)

  @parameterized.parameters("hi", None, object(), object)
  def test_named_call_non_jaxtype_result(self, non_jaxtype):
    def fun_with_non_jaxtype_output(x, non_jaxtype):
      return x, non_jaxtype

    def jitted_fun(x, non_jaxtype):
      named_fun = stateful.named_call(fun_with_non_jaxtype_output)
      # The non-jaxtype is returned out of named_call (which is supported),
      # but is not returned out of the jit (which should not be supported).
      x, non_jaxtype_out = named_fun(x, non_jaxtype)
      self.assertEqual(non_jaxtype_out, non_jaxtype)
      return x

    jitted_fun = jax.jit(jitted_fun, static_argnums=1)
    self.assertEqual(jitted_fun(0, non_jaxtype), 0)

  def test_named_call_partial_function(self):
    f = stateful.named_call(lambda x, y: y if x else None)
    f = jax.jit(functools.partial(f, True))
    out = f(5)
    self.assertEqual(out, 5)

  def test_named_call_default_name(self):
    @stateful.named_call
    def naming_things_is_hard(x):
      return x ** 2

    @jax.jit
    def f(x):
      return naming_things_is_hard(x) + naming_things_is_hard(x)

    c = jax.xla_computation(f)(2)
    self.assertIn("naming_things_is_hard", c.as_hlo_text())

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


def _callback_prim(forward, backward):
  def f_impl(x):
    forward()
    return x

  def b_impl(x):
    backward()
    return (x,)

  prim = jax.core.Primitive("hk_callback")
  prim.def_impl(f_impl)
  prim.def_abstract_eval(f_impl)
  jax.ad.deflinear(prim, b_impl)
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
