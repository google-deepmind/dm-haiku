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
"""Tests for haiku._src.base."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import base
from haiku._src import test_utils
import jax
import jax.numpy as jnp

# TODO(tomhennigan) Improve test coverage.


class BaseTest(parameterized.TestCase):

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

  def test_rng_no_transform(self):
    with self.assertRaisesRegex(ValueError,
                                "must be used as part of an `hk.transform`"):
      base.next_rng_key()

  @test_utils.transform_and_run
  def test_rng(self):
    a = base.next_rng_key()
    b = base.next_rng_key()
    self.assertIsNot(a, b)

  @test_utils.transform_and_run(seed=None)
  def test_no_rng(self):
    with self.assertRaisesRegex(ValueError, "must pass a non-None PRNGKey"):
      base.next_rng_key()

  def test_invalid_rng(self):
    with self.assertRaisesRegex(ValueError, "not a JAX PRNGKey"):
      base.new_context(rng="nonsense")

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

  def test_init_custom_creator(self):
    def zeros_creator(next_creator, name, shape, dtype, init):
      self.assertEqual(name, "~/w")
      self.assertEqual(shape, [])
      self.assertEqual(dtype, jnp.float32)
      self.assertEqual(init, jnp.ones)
      return next_creator(name, shape, dtype, jnp.zeros)

    with base.new_context() as ctx:
      with base.custom_creator(zeros_creator):
        base.get_parameter("w", [], init=jnp.ones)

    self.assertEqual(ctx.collect_params(), {"~": {"w": jnp.zeros([])}})

  def test_unable_to_mutate_name(self):
    def mutates_name(next_creator, name, shape, dtype, init):
      next_creator(name + "_foo", shape, dtype, init)

    with base.new_context(), base.custom_creator(mutates_name):
      with self.assertRaisesRegex(ValueError,
                                  "Modifying .*name.* not supported"):
        base.get_parameter("w", [], init=jnp.ones)

  def test_nested_creators(self):
    log = []

    def logging_creator(log_msg):
      def _logging_creator(next_creator, name, shape, dtype, init):
        log.append(log_msg)
        return next_creator(name, shape, dtype, init)
      return _logging_creator

    with base.new_context():
      with base.custom_creator(logging_creator("a")), \
           base.custom_creator(logging_creator("b")), \
           base.custom_creator(logging_creator("c")):
        base.get_parameter("w", [], init=jnp.ones)

    self.assertEqual(log, ["a", "b", "c"])

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
    ctx_key = jax.random.PRNGKey(seed * 2 + 1)
    key = jax.random.PRNGKey(seed)
    _, next_key = jax.random.split(key)
    expected_output = jax.random.uniform(next_key, ())

    with base.new_context(rng=ctx_key):
      without_decorator_out = jax.random.uniform(base.next_rng_key(), ()).item()

    with base.new_context(rng=ctx_key):
      with base.with_rng(key):
        with_decorator_out = jax.random.uniform(base.next_rng_key(), ()).item()

    self.assertNotEqual(without_decorator_out, expected_output)
    self.assertEqual(with_decorator_out, expected_output)

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

if __name__ == "__main__":
  absltest.main()
