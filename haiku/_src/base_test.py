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
"""Tests for haiku._src.base."""

import functools
import itertools as it

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import base
from haiku._src import config
from haiku._src import test_utils
import jax
import jax.numpy as jnp
import numpy as np

# TODO(tomhennigan) Improve test coverage.

custom_state_creator = functools.partial(
    base.custom_creator, params=False, state=True)

custom_state_getter = functools.partial(
    base.custom_getter, params=False, state=True)

identity_carry = lambda f: lambda carry, x: (carry, f(x))
ignore_index = lambda f: lambda i, x: f(x)


def with_rng_example():
  with base.with_rng(jax.random.PRNGKey(42)):
    pass


def replace_rng_sequence_state_example():
  base.replace_rng_sequence_state((jax.random.PRNGKey(42), tuple()))


# Methods in Haiku that mutate internal state.
SIDE_EFFECTING_FUNCTIONS = (
    ("get_parameter", lambda: base.get_parameter("w", [], init=jnp.zeros)),
    ("get_state", lambda: base.get_state("w", [], init=jnp.zeros)),
    ("set_state", lambda: base.set_state("w", 1)),
    ("next_rng_key", base.next_rng_key),
    ("next_rng_keys", lambda: base.next_rng_keys(2)),
    ("reserve_rng_keys", lambda: base.reserve_rng_keys(2)),
    ("with_rng", with_rng_example),
    (
        "replace_rng_sequence_state",
        replace_rng_sequence_state_example,
    ),
)

# JAX transforms and control flow that need to be aware of Haiku internal
# state to operate unsurprisingly.
# pylint: disable=g-long-lambda
JAX_PURE_EXPECTING_FNS = (
    # Just-in-time compilation.
    ("jit", jax.jit),
    ("make_jaxpr", jax.make_jaxpr),
    ("eval_shape", lambda f: (lambda x: jax.eval_shape(f, x))),

    # Parallelization.
    # TODO(tomhennigan): Add missing features (e.g. pjit,xmap).
    ("pmap", lambda f: jax.pmap(f, "i")),

    # Vectorization.
    ("vmap", jax.vmap),

    # Control flow.
    # TODO(tomhennigan): Enable for associative_scan.
    # ("associative_scan", lambda f:
    #  (lambda x: jax.lax.associative_scan(
    #      lambda a, b: [f(a + b), a + b][-1], jnp.stack([x, x, x, x])))),
    ("cond", lambda f: (lambda x: jax.lax.cond(True, f, f, x))),
    (
        "fori_loop",
        lambda f:
        (lambda x: jax.lax.fori_loop(0, 1, ignore_index(f), x))),
    ("map", lambda f: (lambda x: jax.lax.map(f, x))),
    ("scan", lambda f: (lambda x: jax.lax.scan(identity_carry(f), None, x))),
    ("switch", lambda f: (lambda x: jax.lax.switch(0, [f, f], x))),
    ("while_loop", lambda f: (lambda x: jax.lax.while_loop(
        lambda xs: xs[0] == 0, lambda xs: [1, f(xs[1])], (0, x)))),

    # Automatic differentiation.
    # TODO(tomhennigan): Add missing features (e.g. custom_vjp, custom_jvp).
    ("grad", lambda f: jax.grad(lambda x: f(x).sum())),
    ("value_and_grad", lambda f: jax.value_and_grad(lambda x: f(x).sum())),
    ("checkpoint", jax.checkpoint),  # aka. remat
)
# pylint: enable=g-long-lambda


class BaseTest(parameterized.TestCase):

  def assert_keys_equal(self, a, b):
    self.assertEqual(jax.random.key_impl(a), jax.random.key_impl(b))
    np.testing.assert_array_equal(
        jax.random.key_data(a), jax.random.key_data(b)
    )

  @test_utils.transform_and_run
  def test_parameter_reuse(self):
    w1 = base.get_parameter("w", [], init=jnp.zeros)
    w2 = base.get_parameter("w", [], init=jnp.zeros)
    self.assertIs(w1, w2)

  def test_params(self):
    with base.new_context() as ctx:
      w = base.get_parameter("w", [], init=jnp.zeros)
    self.assertEqual(ctx.collect_params(), {"~": {"w": w}})

  @test_utils.transform_and_run
  def test_naked_get_parameter(self):
    w1 = base.get_parameter("w", [], init=jnp.zeros)
    w2 = base.get_parameter("w", [], init=jnp.zeros)
    self.assertIs(w1, w2)

  def test_naked_parameter_in_tilde_collection(self):
    with base.new_context() as ctx:
      w1 = base.get_parameter("w1", [], init=jnp.zeros)
      w2 = base.get_parameter("w2", [], init=jnp.ones)
      self.assertIsNot(w1, w2)
    self.assertEqual(ctx.collect_params(), {"~": {"w1": w1, "w2": w2}})

  @parameterized.parameters(({},), ({"~": {}},))
  def test_parameter_in_immutable_ctx(self, params):
    with base.new_context(params=params):
      with self.assertRaisesRegex(
          ValueError, "parameters must be created as part of `init`"):
        base.get_parameter("w", [], init=jnp.zeros)

  def test_get_parameter_rng_exception(self):
    with base.new_context():
      with self.assertRaisesRegex(
          base.MissingRNGError, "pass a non-None PRNGKey to init"
      ):
        base.get_parameter(
            "w", [], init=lambda shape, dtype: base.next_rng_key()
        )

  def test_get_parameter_wrong_shape(self):
    with base.new_context():
      with self.assertRaisesRegex(ValueError, "does not match shape"):
        base.get_parameter("w", (1,), init=jnp.zeros)
        base.get_parameter("w", (2,), init=jnp.zeros)

  def test_get_parameter_no_init(self):
    with base.new_context():
      with self.assertRaisesRegex(ValueError, "Initializer must be specified."):
        base.get_parameter("w", [])

  def test_get_parameter_no_init_during_init_second_call(self):
    with base.new_context():
      w = base.get_parameter("w", [], init=jnp.zeros)
      self.assertIs(base.get_parameter("w", []), w)

  def test_get_parameter_no_init_during_apply(self):
    w = jnp.zeros([])
    with base.new_context(params={"~": {"w": w}}):
      self.assertIs(base.get_parameter("w", []), w)

  @parameterized.parameters(base.next_rng_key, lambda: base.next_rng_keys(1))
  def test_rng_no_transform(self, f):
    with self.assertRaisesRegex(ValueError,
                                "must be used as part of an `hk.transform`"):
      f()

  @test_utils.transform_and_run
  def test_rng(self):
    a = base.next_rng_key()
    b = base.next_rng_key()
    self.assertIsNot(a, b)

  @test_utils.transform_and_run
  def test_rngs(self):
    a, b = base.next_rng_keys(2)
    c, d = base.next_rng_keys(2)
    for l, r in it.permutations((a, b, c, d), 2):
      self.assertIsNot(l, r)

  @test_utils.transform_and_run(seed=None)
  def test_no_rng(self):
    with self.assertRaisesRegex(ValueError, "must pass a non-None PRNGKey"):
      base.next_rng_key()

  def test_invalid_rng(self):
    with self.assertRaisesRegex(ValueError, "not a JAX PRNGKey"):
      base.new_context(rng="nonsense")  # type: ignore

  def test_invalid_rng_none_ignored(self):
    with base.new_context(rng=None):
      pass

  def test_maybe_rng_no_transform(self):
    with self.assertRaisesRegex(ValueError,
                                "must be used as part of an `hk.transform`"):
      base.maybe_next_rng_key()

  @test_utils.transform_and_run(seed=None)
  def test_maybe_no_rng(self):
    self.assertIsNone(base.maybe_next_rng_key())

  def test_maybe_rng_vs_not(self):
    """If we have an rng, then next_rng_key() == maybe_next_rng_key()."""
    rngs = []
    maybes = []

    @test_utils.transform_and_run
    def three():
      for _ in range(3):
        rngs.append(base.next_rng_key())

    @test_utils.transform_and_run
    def maybe_three():
      for _ in range(3):
        maybes.append(base.maybe_next_rng_key())

    three()
    maybe_three()
    self.assertLen(rngs, 6)
    self.assertTrue(jnp.all(jnp.array(rngs) == jnp.array(maybes)))

  def test_maybe_get_rng_seq_state_no_transform(self):
    with self.assertRaisesRegex(
        ValueError, "must be used as part of an `hk.transform`"
    ):
      base.maybe_get_rng_sequence_state()

  @test_utils.transform_and_run(seed=None)
  def test_maybe_get_rng_seq_state_no_rng(self):
    self.assertIsNone(base.maybe_get_rng_sequence_state())

  def test_maybe_get_rng_seq_state_vs_next_rng(self):
    rngs_next = []
    rngs_state = []

    @test_utils.transform_and_run
    def next_rng_three():
      for _ in range(3):
        rngs_next.append(base.next_rng_key())

    @test_utils.transform_and_run
    def get_state_three():
      rng_state = base.maybe_get_rng_sequence_state()
      for _ in range(3):
        seq = hk.PRNGSequence(rng_state)
        rng = next(seq)
        rng_state = seq.internal_state
        rngs_state.append(rng)

    next_rng_three()
    get_state_three()
    self.assertLen(rngs_next, 6)
    self.assertTrue(jnp.all(jnp.array(rngs_next) == jnp.array(rngs_state)))

  def test_replace_rng_seq_state_no_transform(self):
    with self.assertRaisesRegex(
        ValueError, "must be used as part of an `hk.transform`"
    ):
      base.replace_rng_sequence_state((jax.random.PRNGKey(42), tuple()))

  @test_utils.transform_and_run(seed=None)
  def test_replace_rng_seq_state_no_rng(self):
    with self.assertRaisesRegex(
        base.MissingRNGError,
        "requires an RNG to be passed into the transformed function",
    ):
      base.replace_rng_sequence_state((jax.random.PRNGKey(42), tuple()))

  @test_utils.transform_and_run(seed=1)
  def test_replace_then_get_rng_seq_state(self):
    rng_state = (
        jax.random.PRNGKey(123),
        (jax.random.PRNGKey(234), jax.random.PRNGKey(345)),
    )
    base.replace_rng_sequence_state(rng_state)
    self.assertEqual(base.maybe_get_rng_sequence_state(), rng_state)

  def test_replace_get_rng_seq_state_vs_no_replace(self):
    rngs_no_replace = []
    rngs_replace = []
    seed = 123

    @test_utils.transform_and_run(seed=seed)
    def no_replace_three():
      for _ in range(3):
        rngs_no_replace.append(base.next_rng_key())

    @test_utils.transform_and_run(seed=1)
    def replace_three():
      if hk.running_init():
        replace_seed = seed
      else:
        replace_seed = seed + 1
      base.replace_rng_sequence_state(
          (jax.random.PRNGKey(replace_seed), tuple())
      )
      for _ in range(3):
        rngs_replace.append(base.next_rng_key())

    no_replace_three()
    replace_three()
    self.assertLen(rngs_no_replace, 6)
    self.assertTrue(
        jnp.all(jnp.array(rngs_no_replace) == jnp.array(rngs_replace))
    )

  @parameterized.parameters(
      (base.get_parameter, base.custom_creator, "collect_params"),
      (base.get_state, custom_state_creator, "collect_state"))
  def test_init_custom_creator(self, get_x, custom_x, collect_x):
    def zeros_creator(next_creator, shape, dtype, init, context):
      self.assertEqual(context.full_name, "~/w")
      self.assertEqual(context.module_name, "~")
      self.assertEqual(context.name, "w")
      self.assertEqual(shape, [])
      self.assertEqual(dtype, jnp.float32)
      self.assertEqual(init, jnp.ones)
      return next_creator(shape, dtype, jnp.zeros)

    with base.new_context() as ctx:
      with custom_x(zeros_creator):
        get_x("w", [], init=jnp.ones)
    self.assertEqual(getattr(ctx, collect_x)(), {"~": {"w": jnp.zeros([])}})

  @parameterized.parameters((base.get_parameter, base.custom_creator),
                            (base.get_state, custom_state_creator))
  def test_nested_creators(self, get_x, custom_x):
    log = []

    def logging_creator(log_msg):
      def _logging_creator(next_creator, shape, dtype, init, context):
        del context
        log.append(log_msg)
        return next_creator(shape, dtype, init)
      return _logging_creator

    with base.new_context():
      with custom_x(logging_creator("a")), \
           custom_x(logging_creator("b")), \
           custom_x(logging_creator("c")):
        get_x("w", [], init=jnp.ones)

    self.assertEqual(log, ["a", "b", "c"])

  @parameterized.parameters((base.get_parameter, base.custom_creator,
                             base.custom_getter, "collect_params"),
                            (base.get_state, custom_state_creator,
                             custom_state_getter, "collect_state"))
  def test_original_dtype(self, get_x, custom_create_x, custom_get_x,
                          collect_x):
    def dtype_cast_creator(next_creator, shape, dtype, init, context):
      if context.original_dtype == jnp.bfloat16:
        dtype = jnp.float32
      return next_creator(shape, dtype, init)

    def dtype_recast_getter(next_getter, value, context):
      if context.original_dtype == jnp.bfloat16:
        assert value.dtype == jnp.float32
        value = value.astype(jnp.bfloat16)
      return next_getter(value)

    with base.new_context() as ctx:
      with custom_create_x(dtype_cast_creator), \
           custom_get_x(dtype_recast_getter):
        value = get_x("w", [], jnp.bfloat16, jnp.ones)
        orig_value = jax.tree.leaves(getattr(ctx, collect_x)())[0]

        assert value.dtype == jnp.bfloat16
        assert orig_value.dtype == jnp.float32

  @parameterized.parameters((base.get_parameter, base.custom_creator),
                            (base.get_state, custom_state_creator))
  def test_original_shape(self, get_x, custom_x):

    def new_shape_creator(next_creator, shape, dtype, init, context):
      del shape
      del context
      new_shape = (1, 2, 3)
      return next_creator(new_shape, dtype, init)

    def original_shape_restorer(next_creator, shape, dtype, init, context):
      assert shape == (1, 2, 3)
      return next_creator(context.original_shape, dtype, init)

    with base.new_context():
      with custom_x(new_shape_creator):
        with custom_x(original_shape_restorer):
          value = get_x("w", [5], jnp.bfloat16, jnp.ones)
          assert value.shape == (5,)

  @parameterized.parameters(
      (base.get_parameter, base.custom_getter, "collect_params"),
      (base.get_state, custom_state_getter, "collect_state"))
  def test_custom_getter_bf16(self, get_x, custom_x, collect_x):
    def bf16_getter(next_getter, value, context):
      del context
      if value.dtype == jnp.float32:
        value = value.astype(jnp.bfloat16)
      return next_getter(value)

    with base.new_context() as ctx:
      with custom_x(bf16_getter):
        f = get_x("f", [], jnp.float32, init=jnp.ones)
        i = get_x("i", [], jnp.int32, init=jnp.ones)

    collection = getattr(ctx, collect_x)()
    self.assertEqual(collection["~"]["f"].dtype, jnp.float32)
    self.assertEqual(f.dtype, jnp.bfloat16)
    self.assertEqual(collection["~"]["i"].dtype, jnp.int32)
    self.assertEqual(i.dtype, jnp.int32)

  @parameterized.parameters((base.get_parameter, base.custom_getter),
                            (base.get_state, custom_state_getter))
  def test_nested_getters(self, get_x, custom_x):
    log = []

    def logging_getter(log_msg, dtype_in, dtype_out):
      def _logging_getter(next_getter, value, context):
        del context
        log.append(log_msg)
        self.assertEqual(value.dtype, dtype_in)
        value = value.astype(dtype_out)
        return next_getter(value)
      return _logging_getter

    with base.new_context():
      with custom_x(logging_getter("a", jnp.float32, jnp.bfloat16)), \
           custom_x(logging_getter("b", jnp.bfloat16, jnp.int32)), \
           custom_x(logging_getter("c", jnp.int32, jnp.int8)):
        w = get_x("w", [], init=jnp.ones)

    self.assertEqual(w.dtype, jnp.int8)
    self.assertEqual(log, ["a", "b", "c"])

  @parameterized.parameters(*it.permutations([True, False], 2))
  def test_creator_types(self, params, state):
    log = []
    def logging_creator(next_creator, shape, dtype, init, context):
      log.append(context.full_name)
      return next_creator(shape, dtype, init)

    with base.new_context():
      with base.custom_creator(logging_creator, params=params, state=state):
        base.get_parameter("params", [], init=jnp.zeros)
        base.get_state("state", [], init=jnp.zeros)

    self.assertLen(log, int(params) + int(state))
    if params:
      self.assertIn("~/params", log)
    if state:
      self.assertIn("~/state", log)

  @parameterized.parameters(*it.permutations([True, False], 2))
  def test_getter_types(self, params, state):
    log = []
    def logging_getter(next_getter, value, context):
      log.append(context.full_name)
      return next_getter(value)

    with base.new_context():
      with base.custom_getter(logging_getter, params=params, state=state):
        base.get_parameter("params", [], init=jnp.zeros)
        base.get_state("state", [], init=jnp.zeros)

    self.assertLen(log, int(params) + int(state))
    if params:
      self.assertIn("~/params", log)
    if state:
      self.assertIn("~/state", log)

  @parameterized.parameters(base.custom_creator, custom_state_creator)
  def test_creator_requires_context(self, custom_x):
    def my_creator(next_creator, shape, dtype, init, context):
      del context
      return next_creator(shape, dtype, init)

    with self.assertRaisesRegex(ValueError,
                                "must be used as part of an `hk.transform`"):
      with custom_x(my_creator):
        pass

  @parameterized.parameters(base.custom_getter, custom_state_getter)
  def test_getter_requires_context(self, custom_x):
    def my_getter(next_getter, value, context):
      del context
      return next_getter(value)

    with self.assertRaisesRegex(ValueError,
                                "must be used as part of an `hk.transform`"):
      with custom_x(my_getter):
        pass

  def test_setter_requires_context(self):
    def my_setter(next_setter, value, context):
      del context
      return next_setter(value)

    with self.assertRaisesRegex(ValueError,
                                "must be used as part of an `hk.transform`"):
      with base.custom_setter(my_setter):
        pass

  def test_setter_array(self):
    witness = []
    x = jnp.ones([])
    y = x + 1

    def my_setter(next_setter, value, context):
      self.assertIs(value, x)
      self.assertEqual(context.original_shape, value.shape)
      self.assertEqual(context.original_dtype, value.dtype)
      self.assertEqual(context.full_name, "~/x")
      self.assertEqual(context.name, "x")
      self.assertIsNone(context.module)
      witness.append(None)
      del next_setter
      return y

    with base.new_context():
      with base.custom_setter(my_setter):
        base.set_state("x", x)
        x = base.get_state("x")
        self.assertIs(x, y)

    self.assertNotEmpty(witness)

  def test_setter_tree(self):
    witness = []
    x = {"a": jnp.ones([]), "b": jnp.zeros([123])}
    y = jax.tree.map(lambda x: x + 1, x)

    def my_setter(next_setter, value, ctx):
      self.assertIs(value, x)
      self.assertEqual(ctx.original_shape, {"a": (), "b": (123,)})
      self.assertEqual(ctx.original_dtype, {"a": jnp.float32, "b": jnp.float32})
      self.assertEqual(ctx.full_name, "~/x")
      self.assertEqual(ctx.name, "x")
      self.assertIsNone(ctx.module)
      witness.append(None)
      del next_setter
      return y

    with base.new_context():
      with base.custom_setter(my_setter):
        base.set_state("x", x)
        x = base.get_state("x")
        self.assertIs(x, y)

    self.assertNotEmpty(witness)

  def test_get_state_no_init_raises(self):
    with base.new_context():
      with self.assertRaisesRegex(ValueError, "set an init function"):
        base.get_state("i")

    with base.new_context(state={"~": {}}):
      with self.assertRaisesRegex(ValueError, "set an init function"):
        base.get_state("i")

  def test_get_state_no_shape_raises(self):
    with base.new_context():
      with self.assertRaisesRegex(ValueError, "provide shape and dtype"):
        base.get_state("i", init=jnp.zeros)

    with base.new_context(state={"~": {}}):
      with self.assertRaisesRegex(ValueError, "provide shape and dtype"):
        base.get_state("i", init=jnp.zeros)

  def test_set_then_get(self):
    with base.new_context() as ctx:
      base.set_state("i", 1)
      base.get_state("i")

    self.assertEqual(ctx.collect_initial_state(), {"~": {"i": 1}})

    for _ in range(10):
      with ctx:
        base.set_state("i", 1)
        y = base.get_state("i")
        self.assertEqual(y, 1)
      self.assertEqual(ctx.collect_initial_state(), {"~": {"i": 1}})

  def test_stateful(self):
    with base.new_context() as ctx:
      for _ in range(10):
        count = base.get_state("count", (), jnp.int32, jnp.zeros)
        base.set_state("count", count + 1)

    self.assertEqual(ctx.collect_initial_state(), {"~": {"count": 0}})
    self.assertEqual(ctx.collect_state(), {"~": {"count": 10}})

  def test_new_state_in_apply(self):
    with base.new_context(params={}, state={}) as ctx:
      base.set_state("count", 1)

    self.assertEqual(ctx.collect_initial_state(), {"~": {"count": 1}})
    self.assertEqual(ctx.collect_state(), {"~": {"count": 1}})

  @parameterized.product(
      seed=[42, 28], wrap_seed=[True, False], jitted=[True, False])
  def test_prng_sequence(self, seed, wrap_seed, jitted):
    def create_random_values(key_or_seed):
      key_seq = base.PRNGSequence(key_or_seed)
      return (jax.random.normal(next(key_seq), []),
              jax.random.normal(next(key_seq), []))
    # Values using our sequence.
    key_or_seed = jax.random.PRNGKey(seed) if wrap_seed else seed
    seq_v1, seq_v2 = (jax.jit(create_random_values)(key_or_seed)
                      if jitted else create_random_values(key_or_seed))
    # Generate values using manual splitting.
    key = jax.random.PRNGKey(seed)
    key, temp_key = jax.random.split(key)
    raw_v1 = jax.random.normal(temp_key, [])
    _, temp_key = jax.random.split(key)
    raw_v2 = jax.random.normal(temp_key, [])
    self.assertEqual(raw_v1, seq_v1)
    self.assertEqual(raw_v2, seq_v2)

  def test_prng_sequence_invalid_input(self):
    with self.assertRaisesRegex(ValueError, "not a JAX PRNGKey"):
      base.PRNGSequence("nonsense")  # type: ignore

  def test_prng_sequence_wrong_shape(self):
    with self.assertRaisesRegex(ValueError,
                                "key did not have expected shape and/or dtype"):
      base.PRNGSequence(jax.random.split(jax.random.PRNGKey(42), 2))

  def test_prng_sequence_wrong_shape_custom_prng(self):
    with self.assertRaisesRegex(ValueError,
                                "key did not have expected shape and/or dtype"):
      with jax.enable_custom_prng():
        base.PRNGSequence(jax.random.split(jax.random.PRNGKey(42), 2))

  def test_prng_reserve(self):
    k = jax.random.PRNGKey(42)
    s = base.PRNGSequence(k)
    s.reserve(10)
    hk_keys = tuple(next(s) for _ in range(10))
    jax_keys = tuple(jax.random.split(test_utils.clone(k), num=11)[1:])
    jax.tree.map(self.assert_keys_equal, hk_keys, jax_keys)

  def test_prng_reserve_twice(self):
    k = jax.random.PRNGKey(42)
    s = base.PRNGSequence(k)
    s.reserve(2)
    s.reserve(2)
    hk_keys = tuple(next(s) for _ in range(4))
    k, subkey1, subkey2 = tuple(jax.random.split(test_utils.clone(k), num=3))
    _, subkey3, subkey4 = tuple(jax.random.split(k, num=3))
    jax_keys = (subkey1, subkey2, subkey3, subkey4)
    jax.tree.map(self.assert_keys_equal, hk_keys, jax_keys)

  def test_prng_sequence_split(self):
    k = jax.random.PRNGKey(42)
    s = base.PRNGSequence(k)
    hk_keys = s.take(10)
    jax_keys = tuple(jax.random.split(test_utils.clone(k), num=11)[1:])
    jax.tree.map(self.assert_keys_equal, hk_keys, jax_keys)

  @parameterized.parameters(42, 28)
  def test_with_rng(self, seed):
    ctx_key = jax.random.PRNGKey(seed * 2 + 1)
    key = jax.random.PRNGKey(seed)
    _, next_key = jax.random.split(key)
    expected_output = jax.random.uniform(next_key, ())

    with base.new_context(rng=ctx_key):
      without_decorator_out = jax.random.uniform(base.next_rng_key(), ()).item()

    with base.new_context(rng=test_utils.clone(ctx_key)):
      with base.with_rng(test_utils.clone(key)):
        with_decorator_out = jax.random.uniform(base.next_rng_key(), ()).item()

    self.assertNotEqual(without_decorator_out, expected_output)
    self.assertEqual(with_decorator_out, expected_output)

  def test_with_rng_no_transform(self):
    with self.assertRaisesRegex(ValueError,
                                "must be used as part of an `hk.transform`"):
      with base.with_rng(jax.random.PRNGKey(428)):
        pass

  def test_new_context(self):
    with base.new_context() as ctx:
      pass
    self.assertEmpty(ctx.collect_params())
    self.assertEmpty(ctx.collect_initial_state())
    self.assertEmpty(ctx.collect_state())

  def test_context_copies_input(self):
    before = {"~": {"w": jnp.array(1.)}}
    with base.new_context(params=before, state=before) as ctx:
      base.get_parameter("w", [], init=jnp.ones)
      base.set_state("w", jnp.array(2.))
    self.assertEqual(ctx.collect_params(), {"~": {"w": jnp.array(1.)}})
    self.assertIsNot(ctx.collect_initial_state(), before)
    self.assertEqual(ctx.collect_initial_state(), before)
    self.assertEqual(ctx.collect_state(), {"~": {"w": jnp.array(2.)}})
    self.assertEqual(before, {"~": {"w": jnp.array(1.)}})

  def test_assert_no_new_parameters(self):
    with base.new_context():
      base.get_parameter("w", [], init=jnp.zeros)
      with base.assert_no_new_parameters():
        # Should not raise, "w" already exists.
        base.get_parameter("w", [], init=jnp.zeros)

      with self.assertRaisesRegex(AssertionError,
                                  "New parameters were created: .*x"):
        with base.assert_no_new_parameters():
          # Should raise, "x" does not exist.
          base.get_parameter("x", [], init=jnp.zeros)

  def test_context_cleanup_after_error(self):
    with base.new_context():
      with self.assertRaisesRegex(ValueError, "expected"):
        raise ValueError("expected")
    self.assertEmpty(base.frame_stack)

  @test_utils.combined_named_parameters(SIDE_EFFECTING_FUNCTIONS,
                                        JAX_PURE_EXPECTING_FNS)
  @test_utils.transform_and_run
  @test_utils.with_guardrails
  def test_unsafe_use_of_jax(self, haiku_side_effect_fn, jax_fn):
    # Make `f` identify with the side effecting function included.
    f = jax_fn(lambda x: [haiku_side_effect_fn(), x][1])
    x = jnp.ones([1])
    with self.assertRaises(base.JaxUsageError):
      f(x)

  def test_do_not_store(self):
    def my_creator(next_creator, shape, dtype, init, context):
      del next_creator, shape, dtype, init, context
      return base.DO_NOT_STORE

    def my_getter(next_getter, value, context):
      assert value is base.DO_NOT_STORE
      return next_getter(
          context.original_init(context.original_shape, context.original_dtype))

    def my_setter(next_setter, value, context):
      del next_setter, value, context
      return base.DO_NOT_STORE

    with base.new_context() as ctx:
      with base.custom_creator(my_creator, state=True), \
           base.custom_getter(my_getter, state=True), \
           base.custom_setter(my_setter):
        self.assertEqual(base.get_parameter("w", [], init=jnp.ones), 1)
        self.assertEqual(base.get_state("s1", [], init=jnp.ones), 1)
        base.set_state("s2", jnp.ones([]))

    self.assertEmpty(ctx.collect_params())
    self.assertEmpty(ctx.collect_state())

  def test_do_not_store_array_like(self):
    with self.assertRaises(ValueError):
      base.DO_NOT_STORE.shape  # pylint: disable=pointless-statement # pytype: disable=attribute-error
    with self.assertRaises(ValueError):
      base.DO_NOT_STORE.dtype  # pylint: disable=pointless-statement # pytype: disable=attribute-error

  def test_current_name_no_transform(self):
    with self.assertRaisesRegex(ValueError,
                                "must be used as part of an `hk.transform`"):
      base.current_name()

  @test_utils.transform_and_run(seed=123, run_apply=False)
  def test_rng_reserve_size(self):
    size = 5

    with config.context(rng_reserve_size=size):
      split_key = jax.random.PRNGKey(123)
      for _ in range(2):
        split_key, *expected_keys = jax.random.split(split_key, size+1)
        hk_keys = hk.next_rng_keys(size)
        jax.tree.map(self.assert_keys_equal, list(hk_keys), expected_keys)

  @parameterized.parameters(
      base.get_params, base.get_current_state, base.get_initial_state
  )
  def test_get_params_or_state_must_be_inside_transform(self, f):
    with self.assertRaisesRegex(ValueError,
                                "must be used as part of an `hk.transform`"):
      f()

  def test_get_params_or_state_empty(self):
    def f():
      self.assertEmpty(base.get_params())
      self.assertEmpty(base.get_initial_state())
      self.assertEmpty(base.get_current_state())

    test_utils.transform_and_run(f)

  def test_get_params_or_state(self):
    sidechannel = [({}, {}, {}), ({}, {}, {})]

    def f():
      sidechannel[0] = (
          base.get_params(),
          base.get_initial_state(),
          base.get_current_state(),
      )
      base.get_parameter("w", [], init=jnp.ones)
      x = base.get_state("x", [], init=jnp.zeros)
      base.set_state("x", x + 1)
      sidechannel[1] = (
          base.get_params(),
          base.get_initial_state(),
          base.get_current_state(),
      )

    f = test_utils.transform.transform_with_state(f)

    params, state = f.init(None)
    (
        (params_before, initial_state_before, current_state_before),
        (params_after, initial_state_after, current_state_after),
    ) = sidechannel
    # Initially params/state are empty.
    self.assertEmpty(params_before)
    self.assertEmpty(initial_state_before)
    self.assertEmpty(current_state_before)
    # At the end of the function the params and initial state should match the
    # output of the init function.
    self.assertEqual(params, params_after)
    self.assertEqual(state, initial_state_after)
    # The current state at the end of the function should have advanced.
    self.assertEqual(current_state_after, {"~": {"x": 1}})
    # The arrays at the leaves of the various dicts should alias.
    self.assertIs(params["~"]["w"], params_after["~"]["w"])
    self.assertIs(state["~"]["x"], initial_state_after["~"]["x"])
    # But the dicts themselves should be different.
    self.assertIsNot(params, params_after)
    self.assertIsNot(params_before, params_after)
    self.assertIsNot(params["~"], params_after["~"])

    _, state = f.apply(params, state, None)
    (
        (params_before, initial_state_before, current_state_before),
        (params_after, initial_state_after, current_state_after),
    ) = sidechannel
    # The params should always match the parameters passed into the apply
    # function.
    self.assertEqual(params_before, params_after)
    self.assertEqual(params, params_after)
    # Initial state should not change during the apply function.
    self.assertEqual(initial_state_before, initial_state_after)
    self.assertEqual(initial_state_before, {"~": {"x": 0}})
    # The current state at the end of the function should match the output of
    # apply.
    self.assertEqual(state, current_state_after)
    # The arrays at the leaves of the various dicts should alias.
    self.assertIs(params_before["~"]["w"], params_after["~"]["w"])
    self.assertIs(params["~"]["w"], params_after["~"]["w"])
    self.assertIs(state["~"]["x"], current_state_after["~"]["x"])
    # But the dicts themselves should be different.
    self.assertIsNot(params, params_after)
    self.assertIsNot(params_before, params_after)
    self.assertIsNot(params["~"], params_after["~"])
    self.assertIsNot(params_before["~"], params_after["~"])

if __name__ == "__main__":
  absltest.main()
