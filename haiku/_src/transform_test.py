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
"""Tests for haiku._src.transform."""

import inspect

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import base
from haiku._src import data_structures
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp

# TODO(tomhennigan) Improve test coverage.


class TransformTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_parameter_reuse(self):
    w1 = base.get_parameter("w", [], init=jnp.zeros)
    w2 = base.get_parameter("w", [], init=jnp.zeros)
    self.assertIs(w1, w2)

  def test_params(self):
    def f():
      w = base.get_parameter("w", [], init=jnp.zeros)
      return w

    init_fn, _ = transform.transform(f)
    params = init_fn(None)
    self.assertEqual(params, {"~": {"w": jnp.zeros([])}})

  @test_utils.transform_and_run
  def test_naked_get_parameter(self):
    w1 = base.get_parameter("w", [], init=jnp.zeros)
    w2 = base.get_parameter("w", [], init=jnp.zeros)
    self.assertIs(w1, w2)

  def test_naked_parameter_in_tilde_collection(self):
    def net():
      w1 = base.get_parameter("w1", [], init=jnp.zeros)
      w2 = base.get_parameter("w2", [], init=jnp.ones)
      self.assertIsNot(w1, w2)

    init_fn, _ = transform.transform(net)
    params = init_fn(None)
    self.assertEqual(params,
                     {"~": {"w1": jnp.zeros([]), "w2": jnp.ones([])}})

  @parameterized.parameters((None,), ({},), ({"~": {}},))
  def test_parameter_in_apply(self, params):
    _, apply_fn = transform.transform(
        lambda: base.get_parameter("w", [], init=jnp.zeros))

    with self.assertRaisesRegex(
        ValueError, "parameters must be created as part of `init`"):
      apply_fn(params, None)

  @test_utils.transform_and_run(seed=None)
  def test_no_rng(self):
    with self.assertRaisesRegex(ValueError, "must pass a non-None PRNGKey"):
      base.next_rng_key()

  def test_invalid_rng(self):
    f = transform.transform(lambda: None)
    with self.assertRaisesRegex(
        ValueError, "Init must be called with an RNG as the first argument"):
      f.init("nonsense")
    with self.assertRaisesRegex(
        ValueError, "Apply must be called with an RNG as the second argument"):
      f.apply({}, "nonsense")

  def test_invalid_rng_state(self):
    f = transform.transform_with_state(lambda: None)
    with self.assertRaisesRegex(
        ValueError, "Init must be called with an RNG as the first argument"):
      f.init("nonsense")
    with self.assertRaisesRegex(
        ValueError, "Apply must be called with an RNG as the third argument"):
      f.apply({}, {"x": {}}, "nonsense")

  @parameterized.parameters(transform.transform,
                            transform.transform_with_state)
  def test_invalid_rng_none_ignored(self, transform_fn):
    f = transform_fn(lambda: None)
    args = f.init(None)
    if not isinstance(args, tuple):
      args = (args,)
    f.apply(*args, None)

  def test_invalid_params(self):
    f = transform.transform_with_state(lambda: None)
    with self.assertRaisesRegex(TypeError,
                                "params argument does not appear valid"):
      f.apply("z", {}, None)

  def test_invalid_state(self):
    f = transform.transform_with_state(lambda: None)
    with self.assertRaisesRegex(TypeError,
                                "state argument does not appear valid"):
      f.apply({}, "z", None)

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

  def test_init_custom_creator(self):
    def zeros_creator(next_creator, shape, dtype, init, context):
      self.assertEqual(context.full_name, "~/w")
      self.assertEqual(shape, [])
      self.assertEqual(dtype, jnp.float32)
      self.assertEqual(init, jnp.ones)
      return next_creator(shape, dtype, jnp.zeros)

    def f():
      with base.custom_creator(zeros_creator):
        return base.get_parameter("w", [], init=jnp.ones)

    params = transform.transform(f).init(None)
    self.assertEqual(params, {"~": {"w": jnp.zeros([])}})

  def test_not_triggered_in_apply(self):
    log = []

    def counting_creator(next_creator, shape, dtype, init, context):
      log.append(context.full_name)
      return next_creator(shape, dtype, init)

    def net():
      with base.custom_creator(counting_creator):
        for i in range(4):
          base.get_parameter("w{}".format(i), [], init=jnp.zeros)

    init_fn, apply_fn = transform.transform(net)

    params = init_fn(None)
    self.assertEqual(log, ["~/w0", "~/w1", "~/w2", "~/w3"])

    del log[:]
    apply_fn(params, None)
    self.assertEmpty(log)

  def test_nested_creators(self):
    log = []

    def logging_creator(log_msg):
      def _logging_creator(next_creator, shape, dtype, init, context):
        del context
        log.append(log_msg)
        return next_creator(shape, dtype, init)
      return _logging_creator

    def f():
      a, b, c = map(logging_creator, ["a", "b", "c"])
      with base.custom_creator(a), \
           base.custom_creator(b), \
           base.custom_creator(c):
        return base.get_parameter("w", [], init=jnp.ones)

    transform.transform(f).init(None)
    self.assertEqual(log, ["a", "b", "c"])

  def test_argspec(self):
    init_fn, apply_fn = transform.transform_with_state(lambda: None)
    init_fn_spec = inspect.getfullargspec(init_fn)
    apply_fn_spec = inspect.getfullargspec(apply_fn)

    self.assertEqual(init_fn_spec.args, ["rng"])
    self.assertEqual(apply_fn_spec.args, ["params", "state", "rng"])

  def test_get_state_no_init_raises(self):
    init_fn, apply_fn = transform.transform_with_state(
        lambda: base.get_state("i"))
    with self.assertRaisesRegex(ValueError, "set an init function"):
      init_fn(None)
    state = params = {"~": {}}
    with self.assertRaisesRegex(ValueError, "set an init function"):
      apply_fn(params, state, None)

  def test_get_state_no_shape_raises(self):
    init_fn, apply_fn = transform.transform_with_state(
        lambda: base.get_state("i", init=jnp.zeros))
    with self.assertRaisesRegex(ValueError, "provide shape and dtype"):
      init_fn(None)
    state = params = {"~": {}}
    with self.assertRaisesRegex(ValueError, "provide shape and dtype"):
      apply_fn(params, state, None)

  def test_get_state_no_init(self):
    _, apply_fn = transform.transform_with_state(lambda: base.get_state("i"))
    for i in range(10):
      state_in = {"~": {"i": i}}
      _, state_out = apply_fn({}, state_in, None)
      self.assertEqual(state_in, state_out)

  def test_set_then_get(self):
    def net():
      base.set_state("i", 1)
      return base.get_state("i")

    init_fn, apply_fn = transform.transform_with_state(net)
    params, state = init_fn(None)
    self.assertEqual(state, {"~": {"i": 1}})

    for i in range(10):
      state_in = {"~": {"i": i}}
      y, state_out = apply_fn(params, state_in, None)
      self.assertEqual(y, 1)
      self.assertEqual(state_out, {"~": {"i": 1}})

  def test_stateful(self):
    def f():
      for _ in range(10):
        count = base.get_state("count", (), jnp.int32, jnp.zeros)
        base.set_state("count", count + 1)
      return count

    init_fn, apply_fn = transform.transform_with_state(f)
    params, state = init_fn(None)
    self.assertEqual(state, {"~": {"count": 0}})
    _, state = apply_fn(params, state, None)
    self.assertEqual(state, {"~": {"count": 10}})

  def test_without_state(self):
    def f():
      w = base.get_parameter("w", [], init=jnp.zeros)
      return w

    init_fn, apply_fn = transform.without_state(
        transform.transform_with_state(f))
    params = init_fn(None)
    out = apply_fn(params, None)
    self.assertEqual(out, 0)

  def test_without_state_raises_if_state_used(self):
    def f():
      for _ in range(10):
        count = base.get_state("count", (), jnp.int32, jnp.zeros)
        base.set_state("count", count + 1)
      return count

    init_fn, _ = transform.without_state(transform.transform_with_state(f))

    with self.assertRaisesRegex(ValueError, "use.*transform_with_state"):
      init_fn(None)

  def test_with_empty_state(self):
    def f():
      w = base.get_parameter("w", [], init=jnp.zeros)
      return w

    init_fn, apply_fn = transform.with_empty_state(
        transform.transform(f))
    params, state = init_fn(None)
    self.assertEmpty(state)
    out, state = apply_fn(params, state, None)
    self.assertEqual(out, 0)
    self.assertEmpty(state)

  def test_inline_use(self):
    def f():
      w = base.get_parameter("w", [], init=jnp.zeros)
      return w

    f = transform.transform(f)

    rng = jax.random.PRNGKey(42)
    params = f.init(rng)
    w = f.apply(params, None)
    self.assertEqual(w, 0)

  def test_method(self):
    obj = ObjectWithTransform()
    x = jnp.ones([])
    params = obj.forward.init(None, x)
    obj_out, y = obj.forward.apply(params, None, x)
    self.assertEqual(y, 1)
    self.assertIs(obj, obj_out)
    params = jax.tree_map(lambda v: v + 1, params)
    obj_out, y = obj.forward.apply(params, None, x)
    self.assertEqual(y, 2)
    self.assertIs(obj, obj_out)

  def test_trampoline(self):
    obj = ObjectWithTransform()
    x = jnp.ones([])
    params = obj.trampoline.init(None, x)
    obj_out, y = obj.trampoline.apply(params, None, x)
    self.assertEqual(y, 1)
    self.assertIs(obj, obj_out)

  @parameterized.parameters((42, True), (42, False),
                            (28, True), (28, False))
  def test_prng_sequence(self, seed, wrap_seed):
    # Values using our sequence.
    key_or_seed = jax.random.PRNGKey(seed) if wrap_seed else seed
    key_seq = base.PRNGSequence(key_or_seed)
    seq_v1 = jax.random.normal(next(key_seq), [])
    seq_v2 = jax.random.normal(next(key_seq), [])
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
      base.PRNGSequence("nonsense")

  def test_prng_sequence_wrong_shape(self):
    with self.assertRaisesRegex(ValueError,
                                "key did not have expected shape and/or dtype"):
      base.PRNGSequence(jax.random.split(jax.random.PRNGKey(42), 2))

  @parameterized.parameters(42, 28)
  def test_with_rng(self, seed):
    key = jax.random.PRNGKey(seed)
    unrelated_key = jax.random.PRNGKey(seed * 2 + 1)
    _, next_key = jax.random.split(key)
    expected_output = jax.random.uniform(next_key, ())

    def without_decorator():
      return jax.random.uniform(base.next_rng_key(), ())
    without_decorator = transform.transform(without_decorator)
    without_decorator_out = without_decorator.apply(None, unrelated_key).item()

    def with_decorator():
      with base.with_rng(key):
        return jax.random.uniform(base.next_rng_key(), ())

    with_decorator = transform.transform(with_decorator)
    with_decorator_out = with_decorator.apply(None, unrelated_key).item()

    self.assertNotEqual(without_decorator_out, expected_output)
    self.assertEqual(with_decorator_out, expected_output)

  def test_without_apply_rng_output_type(self):
    def f():
      w = base.get_parameter("w", [], init=jnp.zeros)
      return w

    f = transform.without_apply_rng(transform.transform_with_state(f))
    self.assertIsInstance(f, transform.TransformedWithState)

    f = transform.without_apply_rng(transform.transform(f))
    self.assertIsInstance(f, transform.Transformed)

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

  def test_without_state_raises_if_state_used_on_apply(self):
    f = lambda: base.set_state("~", 1)
    f = transform.without_state(transform.transform_with_state(f))
    rng = jax.random.PRNGKey(42)
    with self.assertRaisesRegex(ValueError, "use.*transform_with_state"):
      params = f.init(rng)
      f.apply(params, rng)

  def test_running_init(self):
    l = []
    f = transform.transform(lambda: l.append(transform.running_init()))
    f.init(None)
    f.apply({}, None)
    init_value, apply_value = l  # pylint: disable=unbalanced-tuple-unpacking
    self.assertEqual(init_value, True)
    self.assertEqual(apply_value, False)

  def test_running_init_outside_transform(self):
    with self.assertRaisesRegex(ValueError,
                                "running_init.*used as part of.*transform"):
      transform.running_init()

  @parameterized.parameters(
      None,
      transform.without_apply_rng,
      transform.without_state,
      lambda f: transform.without_state(transform.without_apply_rng(f)))
  def test_persists_original_fn(self, without):
    orig_f = lambda: None
    f = transform.transform(orig_f)
    if without is not None:
      f = without(f)
    self.assertPersistsOriginal(f, orig_f)

  @parameterized.parameters(
      None,
      lambda f: transform.with_empty_state(transform.without_state(f)))
  def test_persists_original_fn_transform_with_state(self, without):
    orig_f = lambda: None
    f = transform.transform_with_state(orig_f)
    if without is not None:
      f = without(f)
    self.assertPersistsOriginal(f, orig_f)

  def assertPersistsOriginal(self, f, orig_f):
    self.assertIs(transform.get_original_fn(f), orig_f)
    self.assertIs(transform.get_original_fn(f.init), orig_f)
    self.assertIs(transform.get_original_fn(f.apply), orig_f)

  @parameterized.parameters(
      transform.transform,
      lambda f: transform.without_state(transform.transform_with_state(f)))
  def test_calling_with_duplicate_state_kwarg(self, transform_fn):
    def f(state):
      del state
    self.assert_raises_by_name_error(transform_fn(f))

  @parameterized.parameters(transform.transform, transform.transform_with_state)
  def test_calling_with_duplicate_rng_kwarg(self, transform_fn):
    def f(rng):
      del rng
    self.assert_raises_by_name_error(
        transform.without_apply_rng(transform_fn(f)))

  def assert_raises_by_name_error(self, f):
    with self.assertRaisesRegex(TypeError, "pass them positionally"):
      f.apply(params=None, state=None, rng=None)

  @test_utils.with_environ("HAIKU_FLATMAPPING", None)
  def test_output_type_default(self):
    self.assert_output_type(data_structures.FlatMap)

  @test_utils.with_environ("HAIKU_FLATMAPPING", "0")
  def test_output_type_env_var_0(self):
    self.assert_output_type(dict)

  @test_utils.with_environ("HAIKU_FLATMAPPING", "1")
  def test_output_type_env_var_1(self):
    self.assert_output_type(data_structures.FlatMap)

  def assert_output_type(self, cls):
    def f():
      base.get_parameter("w", [], init=jnp.zeros)
      base.get_state("w", [], init=jnp.zeros)

    init, apply = transform.transform_with_state(f)
    params, state_in = init(None)
    _, state_out = apply(params, state_in, None)
    self.assertLen(params, 1)
    self.assertLen(state_in, 1)
    self.assertLen(state_out, 1)
    self.assertEqual(type(params), cls)
    self.assertEqual(type(params["~"]), cls)
    self.assertEqual(type(state_in["~"]), cls)
    self.assertEqual(type(state_out["~"]), cls)

  def test_unexpected_tracer_error_hint(self):
    def leaks_and_uses_tracer():
      jax.jit(base.next_rng_key)()
      base.next_rng_key()

    init, _ = transform.transform(leaks_and_uses_tracer)
    with self.assertRaisesRegex(jax.errors.UnexpectedTracerError,
                                "want to use the Haiku version"):
      init(jax.random.PRNGKey(42))


class ObjectWithTransform:

  def __init__(self):
    self.trampoline = transform.transform(self._trampoline)
    self.forward = transform.transform(self._forward)

  def _trampoline(self, x):
    return self._forward(x)

  def _forward(self, x):
    w = base.get_parameter("w", [], init=jnp.zeros)
    return self, x + w

if __name__ == "__main__":
  absltest.main()
