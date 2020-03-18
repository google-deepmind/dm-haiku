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
"""Tests for haiku._src.module."""

import contextlib
from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import base
from haiku._src import module
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp


# TODO(tomhennigan) Improve test coverage.
class ModuleTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_module_naming_default(self):
    mod1 = EmptyModule()
    mod2 = EmptyModule()
    self.assertEqual(mod1.module_name, "empty_module")
    self.assertEqual(mod2.module_name, "empty_module_1")

  @test_utils.transform_and_run
  def test_module_naming_custom(self):
    mod1 = EmptyModule(name="custom_name")
    mod2 = EmptyModule(name="custom_name")
    self.assertEqual(mod1.module_name, "custom_name")
    self.assertEqual(mod2.module_name, "custom_name_1")

  @parameterized.parameters(1, 2, 3)
  @test_utils.transform_and_run
  def test_module_naming_explicit_numbering(self, step):
    for n in range(0, step * 10, step):
      module_name = f"custom_name_{n}"
      self.assertEqual(EmptyModule(name=module_name).module_name, module_name)

  @parameterized.parameters(1, 2, 3)
  @test_utils.transform_and_run
  def test_module_naming_explicit_reverse_numbering(self, step):
    total = step * 10
    for n in range(0, total, step):
      n = total - n
      module_name = f"custom_name_{n}"
      self.assertEqual(EmptyModule(name=module_name).module_name, module_name)

    self.assertEqual(EmptyModule(name="custom_name").module_name,
                     f"custom_name_{total + 1}")

  @test_utils.transform_and_run
  def test_module_naming_explicit_numbering_collision(self):
    self.assertEqual(EmptyModule(name="custom_name").module_name, "custom_name")
    self.assertEqual(EmptyModule(name="custom_name").module_name,
                     "custom_name_1")
    with self.assertRaisesRegex(
        ValueError, "Module name 'custom_name_1' is not unique"):
      EmptyModule(name="custom_name_1")

  @test_utils.transform_and_run
  def test_module_naming_explicit_numbering_out_of_order(self):
    for n in (1, 3, 2, 4):
      self.assertEqual(
          EmptyModule(name=f"custom_name_{n}").module_name, f"custom_name_{n}")
    with self.assertRaisesRegex(
        ValueError, "Module name 'custom_name_4' is not unique"):
      EmptyModule(name="custom_name_4")

  @test_utils.transform_and_run
  def test_flatten_invalid_name(self):
    with self.assertRaisesRegex(ValueError, "is not a valid module name"):
      EmptyModule(name="1bad-name")

  @test_utils.transform_and_run
  def test_parameter_reuse(self):
    mod = ScalarModule()
    w1 = mod()
    w2 = mod()
    self.assertIs(w1, w2)

  @test_utils.transform_and_run
  def test_multiple_forward_methods(self):
    mod = MultipleForwardMethods(name="outer")
    mod()
    self.assertEqual(mod.ctor_mod.module_name, "outer/~/scalar_module")
    self.assertEqual(mod.call_mod.module_name, "outer/scalar_module")
    self.assertEqual(mod.encode_mod.module_name, "outer/~encode/scalar_module")
    self.assertEqual(mod.decode_mod.module_name, "outer/~decode/scalar_module")

  @test_utils.transform_and_run
  def test_nesting(self):
    mod = ParentModule()
    self.assertEqual(mod.module_name, "parent_module")
    self.assertEqual(mod.child1.module_name, "parent_module/~/child_module")
    self.assertEqual(mod.child2.module_name, "parent_module/~/child_module_1")

  def test_outside_transform_exception(self):
    with self.assertRaisesRegex(ValueError,
                                "initialized inside an `hk.transform`"):
      EmptyModule()

  def test_params(self):
    init_fn, _ = transform.transform(lambda: ScalarModule()())  # pylint: disable=unnecessary-lambda
    params = init_fn(None)
    self.assertEqual(params, {"scalar_module": {"w": jnp.zeros([])}})

  def test_params_nested(self):
    init_fn, _ = transform.transform(
        lambda: MultipleForwardMethods(name="outer")())  # pylint: disable=unnecessary-lambda
    params = init_fn(None)
    self.assertEqual(params,
                     {"outer/~/scalar_module": {"w": jnp.zeros([])},
                      "outer/scalar_module": {"w": jnp.zeros([])},
                      "outer/~encode/scalar_module": {"w": jnp.zeros([])},
                      "outer/~decode/scalar_module": {"w": jnp.zeros([])}})

  def test_used_inside_transform(self):
    log = []

    def counting_creator(next_creator, name, shape, dtype, init):
      log.append(name)
      return next_creator(name, shape, dtype, init)

    def net():
      with base.custom_creator(counting_creator):
        return MultipleForwardMethods()()

    init_fn, apply_fn = transform.transform(net)

    params = init_fn(None)
    self.assertEqual(log, [
        "multiple_forward_methods/~/scalar_module/w",        # __init__
        "multiple_forward_methods/scalar_module/w",          # __call__
        "multiple_forward_methods/~encode/scalar_module/w",  # encode
        "multiple_forward_methods/~decode/scalar_module/w",  # decode
    ])

    del log[:]
    apply_fn(params)
    self.assertEmpty(log)

  def test_stateful_module(self):
    init_fn, apply_fn = transform.transform_with_state(
        lambda: CountingModule()())  # pylint: disable=unnecessary-lambda
    params, state = init_fn(None)
    self.assertEqual(state, {"counting_module": {"count": 0}})
    _, state = apply_fn(params, state, None)
    self.assertEqual(state, {"counting_module": {"count": 10}})

  def test_without_state(self):
    init_fn, apply_fn = transform.without_state(
        transform.transform_with_state(lambda: ScalarModule()()))  # pylint: disable=unnecessary-lambda
    params = init_fn(None)
    out = apply_fn(params, None)
    self.assertEqual(out, 0)

  def test_without_state_raises_if_state_used(self):
    init_fn, _ = transform.without_state(
        transform.transform_with_state(lambda: CountingModule()()))  # pylint: disable=unnecessary-lambda
    with self.assertRaisesRegex(ValueError, "use.*transform_with_state"):
      init_fn(None)

  @test_utils.transform_and_run
  def test_params_dict(self):
    mods = [ScalarModule() for _ in range(5)]
    for i, mod in enumerate(mods):
      w = mod()
      if i:
        self.assertEqual(mod.params_dict(), {"scalar_module_{}/w".format(i): w})
      else:
        self.assertEqual(mod.params_dict(), {"scalar_module/w": w})

  @test_utils.transform_and_run
  def test_params_dict_captured(self):
    mod = CapturesModule(ScalarModule())
    w = mod()
    self.assertEqual(mod.params_dict(), {"scalar_module/w": w})

  @test_utils.transform_and_run
  def test_params_dict_captured_lambda(self):
    mod = CapturesModule(lambda: ScalarModule()())  # pylint: disable=unnecessary-lambda
    w = mod()
    self.assertIs(w, mod())
    self.assertEqual(mod.params_dict(), {"captures_module/scalar_module/w": w})

  def test_inline_use(self):
    def f():
      return ScalarModule()()

    f = transform.transform(f)

    rng = jax.random.PRNGKey(42)
    params = f.init(rng)
    w = f.apply(params)
    self.assertEqual(w, 0)

  def test_transparent(self):
    init_fn, _ = transform.transform(lambda: TransparentModule()())  # pylint: disable=unnecessary-lambda
    params = init_fn(None)
    self.assertEqual(params, {"scalar_module": {"w": jnp.zeros([])}})

  @test_utils.transform_and_run
  def test_method_hook(self):
    events = []
    @contextlib.contextmanager
    def method_hook(mod, method_name):
      events.append(("enter", method_name, getattr(mod, "module_name", None)))
      yield
      events.append(("exit", method_name, mod.module_name))

    # Test __init__.
    with module.hook_methods(method_hook):
      m = EmptyModule()
      self.assertIsNotNone(m)
      self.assertEqual(events, [("enter", "__init__", None),
                                ("exit", "__init__", "empty_module")])

    # Test __call__.
    del events[:]
    m = CapturesModule(ScalarModule())
    with module.hook_methods(method_hook):
      m()
    self.assertEqual(events, [("enter", "__call__", "captures_module"),
                              ("enter", "__call__", "scalar_module"),
                              ("exit", "__call__", "scalar_module"),
                              ("exit", "__call__", "captures_module")])

  @test_utils.transform_and_run
  def test_callback_runs_after_submodules_updated(self):
    params = []
    @contextlib.contextmanager
    def method_hook(mod, method_name):
      yield
      if method_name != "params_dict":
        params.append((mod.module_name, method_name, tuple(mod.params_dict())))

    m = CapturesModule(ScalarModule())
    with module.hook_methods(method_hook):
      m()
    self.assertEqual(params,
                     [("scalar_module", "__call__", ("scalar_module/w",)),
                      ("captures_module", "__call__", ("scalar_module/w",))])

  def test_context_reuse_same_instance(self):
    params = {"parent_module/~/child_module": {"w": jnp.array(2.)},
              "parent_module/~/child_module_1": {"w": jnp.array(3.)},
              "parent_module_1/~/child_module": {"w": jnp.array(4.)},
              "parent_module_1/~/child_module_1": {"w": jnp.array(5.)}}

    with base.new_context(params=params) as ctx:
      mod1 = ParentModule()
      mod2 = ParentModule()
      self.assertEqual(mod1.module_name, "parent_module")
      self.assertEqual(mod2.module_name, "parent_module_1")
      for parent, (c1, c2) in ((mod1, (2., 3.)), (mod2, (4., 5.))):
        self.assertEqual(parent.child1(), c1)
        self.assertEqual(parent.child2(), c2)

    with ctx:
      for parent, (c1, c2) in ((mod1, (2., 3.)), (mod2, (4., 5.))):
        self.assertEqual(parent.child1(), c1)
        self.assertEqual(parent.child2(), c2)

    # Creating a new context should not be a problem.
    with base.new_context(params=ctx.collect_params()) as ctx:
      mod1 = ParentModule()
      mod2 = ParentModule()
      self.assertEqual(mod1.module_name, "parent_module")
      self.assertEqual(mod2.module_name, "parent_module_1")
      for parent, (c1, c2) in ((mod1, (2., 3.)), (mod2, (4., 5.))):
        self.assertEqual(parent.child1(), c1)
        self.assertEqual(parent.child2(), c2)


class CapturesModule(module.Module):

  def __init__(self, mod):
    super(CapturesModule, self).__init__()
    self._mod = mod

  def __call__(self):
    return self._mod()


class EmptyModule(module.Module):
  pass


class ScalarModule(module.Module):

  def __call__(self):
    return base.get_parameter("w", [], init=jnp.zeros)


class ParentModule(module.Module):

  def __init__(self):
    super(ParentModule, self).__init__()
    self.child1 = ScalarModule(name="child_module")
    self.child2 = ScalarModule(name="child_module")


class MultipleForwardMethods(module.Module):

  def __init__(self, name=None):
    super(MultipleForwardMethods, self).__init__(name=name)
    s = ScalarModule()
    s()
    self.ctor_mod = s

  def __call__(self):
    s = ScalarModule()
    self.call_mod = s
    x = s()
    x += self.autoencode()
    return x

  def autoencode(self):
    x = self.encode()
    x += self.decode()
    return x

  def encode(self):
    s = ScalarModule()
    self.encode_mod = s
    return s()

  def decode(self):
    s = ScalarModule()
    self.decode_mod = s
    return s()


class CountingModule(module.Module):

  def __call__(self):
    for _ in range(10):
      count = base.get_state("count", (), jnp.int32, jnp.zeros)
      base.set_state("count", count + 1)
    return count


class TransparentModule(module.Module):

  @module.transparent
  def __call__(self):
    return ScalarModule()()


if __name__ == "__main__":
  absltest.main()
