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

import abc
import contextlib
import sys
from typing import Callable, Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
import dataclasses
from haiku._src import base
from haiku._src import module
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp

# pylint: disable=g-import-not-at-top,g-multiple-import
if sys.version_info < (3, 8):
  from typing_extensions import Protocol, runtime_checkable
else:
  from typing import Protocol, runtime_checkable
# pylint: enable=g-import-not-at-top,g-multiple-import


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

  @test_utils.transform_and_run
  def test_supports_arg_named_module(self):

    class MyModule(module.Module):

      def __init__(self, module):  # pylint: disable=redefined-outer-name
        del module
        super().__init__()

    self.assertIsNotNone(MyModule(module=None))

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
    name_log = []
    module_log = []

    def counting_creator(next_creator, shape, dtype, init, context):
      name_log.append(context.full_name)
      mod = context.module
      module_log.append((type(mod), mod.module_name))
      return next_creator(shape, dtype, init)

    def net():
      with base.custom_creator(counting_creator):
        return MultipleForwardMethods()()

    init_fn, apply_fn = transform.transform(net)

    params = init_fn(None)
    self.assertEqual(name_log, [
        "multiple_forward_methods/~/scalar_module/w",        # __init__
        "multiple_forward_methods/scalar_module/w",          # __call__
        "multiple_forward_methods/~encode/scalar_module/w",  # encode
        "multiple_forward_methods/~decode/scalar_module/w",  # decode
    ])

    self.assertEqual(module_log, [
        (ScalarModule, "multiple_forward_methods/~/scalar_module"),
        (ScalarModule, "multiple_forward_methods/scalar_module"),
        (ScalarModule, "multiple_forward_methods/~encode/scalar_module"),
        (ScalarModule, "multiple_forward_methods/~decode/scalar_module"),
    ])

    del name_log[:]
    apply_fn(params, None)
    self.assertEmpty(name_log)

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

  @test_utils.transform_and_run
  def test_state_dict(self):
    mods = [ScalarStateModule() for _ in range(5)]
    for i, mod in enumerate(mods):
      w = mod()
      if i:
        self.assertEqual(mod.state_dict(),
                         {"scalar_state_module_{}/w".format(i): w})
      else:
        self.assertEqual(mod.state_dict(), {"scalar_state_module/w": w})

  @test_utils.transform_and_run
  def test_state_dict_captured(self):
    mod = CapturesModule(ScalarStateModule())
    w = mod()
    self.assertEqual(mod.state_dict(), {"scalar_state_module/w": w})

  @test_utils.transform_and_run
  def test_state_dict_captured_lambda(self):
    mod = CapturesModule(lambda: ScalarStateModule()())  # pylint: disable=unnecessary-lambda
    w = mod()
    self.assertIs(w, mod())
    self.assertEqual(mod.state_dict(),
                     {"captures_module/scalar_state_module/w": w})

  def test_inline_use(self):
    def f():
      return ScalarModule()()

    f = transform.transform(f)

    rng = jax.random.PRNGKey(42)
    params = f.init(rng)
    w = f.apply(params, None)
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

  @test_utils.transform_and_run
  def test_submodules_in_ctor_tracked(self):
    m = CreatesSubmoduleWithCtorParam(name="parent")
    self.assertEqual(m._submodules, {m.child.module_name})

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

  @parameterized.parameters(None, "mlp")
  def test_dataclass(self, name):
    with base.new_context() as ctx:
      output_sizes = [300, 100, 10]
      if name is None:
        mlp = DataMLP(output_sizes)
      else:
        mlp = DataMLP(output_sizes, name="mlp")
      mlp(jnp.ones([1, 28 * 28]))
      params = ctx.collect_params()
      if name is None:
        module_names = ["data_mlp/linear", "data_mlp/linear_1",
                        "data_mlp/linear_2"]
      else:
        module_names = ["mlp/linear", "mlp/linear_1", "mlp/linear_2"]
      self.assertEqual(list(params.keys()), module_names)
      for module_name, output_size in zip(module_names, output_sizes):
        self.assertEqual(params[module_name]["w"].shape[-1], output_size)
        self.assertEqual(params[module_name]["b"].shape[-1], output_size)

  @test_utils.transform_and_run
  def test_intercept_method(self):
    mod = IdentityModule()
    x = jnp.ones([])
    call_count = []

    def add_one_interceptor(f, args, kwargs, context):
      call_count.append(None)
      self.assertLen(context, 3)
      self.assertIs(context.module, mod)
      self.assertEqual(context.method_name, "__call__")
      self.assertEqual(context.orig_method(2), 2)
      self.assertEqual(args, (x,))
      self.assertEmpty(kwargs)
      y = f(*args, **kwargs)
      return y + 1

    y1 = mod(x)
    with module.intercept_methods(add_one_interceptor):
      y2 = mod(x)
    y3 = mod(x)

    self.assertLen(call_count, 1)
    self.assertEqual(y1, 1)
    self.assertEqual(y2, 2)
    self.assertEqual(y3, 1)

  @test_utils.transform_and_run
  def test_intercept_methods_calling_underlying_optional(self):
    def do_nothing_interceptor(f, args, kwargs, context):
      del f, context
      self.assertEmpty(args)
      self.assertEmpty(kwargs)

    m = RaisesModule()
    with module.intercept_methods(do_nothing_interceptor):
      m()

    with self.assertRaises(AssertionError):
      m()  # Without the interceptor we expect an error.

    # The previous error should not stop us from re-applying.
    with module.intercept_methods(do_nothing_interceptor):
      m()

  @test_utils.transform_and_run
  def test_intercept_methods_run_in_lifo_order(self):
    def op_interceptor(op):
      def _interceptor(f, args, kwargs, context):
        del context
        y = f(*args, **kwargs)
        return op(y)
      return _interceptor

    mod = IdentityModule()
    x = 7
    with module.intercept_methods(op_interceptor(lambda a: a + 1)), \
         module.intercept_methods(op_interceptor(lambda a: a ** 2)):
      y = mod(x)
    self.assertEqual(y, (x ** 2) + 1)

    with module.intercept_methods(op_interceptor(lambda a: a ** 2)), \
         module.intercept_methods(op_interceptor(lambda a: a + 1)):
      y = mod(x)
    self.assertEqual(y, (x + 1) ** 2)

  @test_utils.transform_and_run
  def test_name_scope_trivial(self):
    with module.name_scope("foo"):
      mod1 = module.Module(name="bar")
      mod2 = module.Module(name="bar")
    self.assertEqual(mod1.module_name, "foo/bar")
    self.assertEqual(mod2.module_name, "foo/bar_1")

  @test_utils.transform_and_run
  def test_name_scope_inside_module(self):
    mod = NameScopeModule(name="module")
    w, w_foo = mod()
    self.assertIsNot(w, w_foo)
    params = mod.params_dict()
    self.assertLen(params, 2)
    self.assertIs(params["module/w"], w)
    self.assertIs(params["module/foo/w"], w_foo)

  @test_utils.transform_and_run
  def test_name_scope_slash_delimited(self):
    with module.name_scope("foo/bar"):
      mod = module.Module(name="baz")
    self.assertEqual(mod.module_name, "foo/bar/baz")

  @test_utils.transform_and_run
  def test_name_scope_nesting(self):
    with module.name_scope("foo"):
      with module.name_scope("bar"):
        mod = module.Module(name="baz")
    self.assertEqual(mod.module_name, "foo/bar/baz")

  @test_utils.transform_and_run
  def test_name_scope_duplicate_name(self):
    with module.name_scope("foo"):
      mod1 = module.Module(name="bar")
    with module.name_scope("foo"):
      mod2 = module.Module(name="bar")
    self.assertEqual(mod1.module_name, "foo/bar")
    self.assertEqual(mod2.module_name, "foo_1/bar")

  @test_utils.transform_and_run
  def test_name_scope_reuse(self):
    # NOTE: If you are considering lifting this restriction, please think
    # carefully about the following case:
    #
    #     def f(x):
    #       foo_scope = name_scope("foo")
    #       with foo_scope: x = BarModule()(x)  # name: foo/bar_module
    #       with foo_scope: x = BarModule()(x)  # name: foo/bar_module
    #       return x
    #
    # We believe that the name reuse (when the scope is reused) will surprise
    # users and lead to bugs. This behaviour does match what would happen if you
    # put the body of the context manager into a method and called that method
    # twice.

    scope = module.name_scope("foo")
    with scope:
      pass
    with self.assertRaisesRegex(ValueError, "name_scope is not reusable"):
      with scope:
        pass

  @test_utils.transform_and_run
  def test_name_scope_reuse_after_error(self):
    scope = module.name_scope("foo")
    with self.assertRaisesRegex(AssertionError, "expected"):
      with scope:
        assert False, "expected"

    with self.assertRaisesRegex(ValueError, "name_scope is not reusable"):
      with scope:
        pass

  @test_utils.transform_and_run
  def test_name_scope_leading_slash(self):
    with self.assertRaisesRegex(ValueError,
                                "Name scopes must not start with /"):
      module.name_scope("/foo")

  def test_name_scope_outside_transform(self):
    with self.assertRaisesRegex(
        ValueError, "name_scope.*must be used as part of an `hk.transform`"):
      module.name_scope("foo")

  @test_utils.transform_and_run
  def test_is_protocol(self):
    self.assertFalse(getattr(module.Module, "_is_protocol"))
    self.assertFalse(getattr(ConcreteProtocolModule, "_is_protocol"))
    # NOTE: Technically this bit is set wrong (ProtocolModule) is a protocol.
    self.assertFalse(getattr(ProtocolModule, "_is_protocol"))

  @test_utils.transform_and_run
  def test_instance_checks(self):
    self.assertIsInstance(ConcreteProtocolModule(), module.Module)
    self.assertIsInstance(ConcreteProtocolModule(), SupportsFoo)
    self.assertIsInstance(ConcreteProtocolModule(), ProtocolModule)
    self.assertNotIsInstance(module.Module(), SupportsFoo)
    self.assertNotIsInstance(module.Module(), ProtocolModule)

  @test_utils.transform_and_run
  def test_name_like(self):
    m = ModuleWithCustomName(name="parent")
    m.foo()  # foo pretends to be __call__.
    m.bar()  # bar pretends to be baz.
    # baz and call are happy to be themselves.
    m.baz()
    m()
    self.assertEqual(m.init_module.module_name, "parent/~/child")
    self.assertEqual(m.foo_module.module_name, "parent/child")
    self.assertEqual(m.bar_module.module_name, "parent/~baz/child")
    self.assertEqual(m.baz_module.module_name, "parent/~baz/child")
    self.assertEqual(m.call_module.module_name, "parent/child")

  @test_utils.transform_and_run
  def test_name_like_aliasing(self):
    m = ModuleWithDoubleCall(name="parent")
    m()
    self.assertEqual(m.foo_module.module_name, "parent/child")  # pytype: disable=attribute-error
    self.assertEqual(m.call_module.module_name, "parent/child")

  @test_utils.transform_and_run
  def test_name_like_on_call(self):
    m = ModuleWithCustomNameOnCall(name="parent")
    m.foo()
    m()  # Call pretends to be foo.
    self.assertEqual(m.init_module.module_name, "parent/~/child")
    self.assertEqual(m.foo_module.module_name, "parent/~foo/child")
    self.assertEqual(m.call_module.module_name, "parent/~foo/child")

  @test_utils.transform_and_run
  def test_name_like_on_init(self):
    m = ModuleWithCustomNameOnInit(name="parent")  # init pretends to be call.
    m()
    self.assertEqual(m.init_module.module_name, "parent/child")
    self.assertEqual(m.call_module.module_name, "parent/child")

  @test_utils.transform_and_run
  def test_name_like_interceptor_method_names_unchanged(self):
    log = []
    def log_parent_methods(f, args, kwargs, context: module.MethodContext):
      if isinstance(context.module, ModuleWithCustomName):
        log.append(context.method_name)
      return f(*args, **kwargs)

    with module.intercept_methods(log_parent_methods):
      m = ModuleWithCustomName(name="parent")
      m.foo()  # foo pretends to be __call__.
      m.bar()  # bar pretends to be baz.
      # baz and call are happy to be themselves.
      m.baz()
      m()

    self.assertEqual(log, ["__init__", "foo", "bar", "baz", "__call__"])


class IdentityModule(module.Module):

  def __call__(self, x):
    return x


class RaisesModule(module.Module):

  def __call__(self):
    assert False


class CapturesModule(module.Module):

  def __init__(self, mod):
    super().__init__()
    self._mod = mod

  def __call__(self):
    return self._mod()


class CreatesSubmoduleWithCtorParam(module.Module):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.child = HasConstructorParam(name="child")


class HasConstructorParam(module.Module):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.w = base.get_parameter("w", [], init=jnp.zeros)


class EmptyModule(module.Module):
  pass


class ScalarModule(module.Module):

  def __call__(self):
    return base.get_parameter("w", [], init=jnp.zeros)


class ScalarStateModule(module.Module):

  def __call__(self):
    return base.get_state("w", [], init=jnp.zeros)


class ParentModule(module.Module):

  def __init__(self):
    super().__init__()
    self.child1 = ScalarModule(name="child_module")
    self.child2 = ScalarModule(name="child_module")


class MultipleForwardMethods(module.Module):

  def __init__(self, name=None):
    super().__init__(name=name)
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


@dataclasses.dataclass
class DataLinear(module.Module):

  output_size: int
  name: Optional[str] = None

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    w = base.get_parameter("w", [j, k], init=jnp.ones)
    b = base.get_parameter("b", [k], init=jnp.zeros)
    return x @ w + b


@dataclasses.dataclass
class DataMLP(module.Module):

  output_sizes: Sequence[int]
  activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu
  name: Optional[str] = None

  def __call__(self, x):
    for i, output_size in enumerate(self.output_sizes):
      if i > 0:
        x = self.activation(x)
      x = DataLinear(output_size, name="linear")(x)
    return x


class NameScopeModule(module.Module):

  def __call__(self):
    w = base.get_parameter("w", [], init=jnp.zeros)
    with module.name_scope("foo"):
      w_foo = base.get_parameter("w", [], init=jnp.zeros)
    return w, w_foo


@runtime_checkable
class SupportsFoo(Protocol):

  @abc.abstractmethod
  def foo(self) -> int:
    ...


# Check that we can declare a module that also inherits from a Protocol without
# encountering a metaclass conflict.
class ProtocolModule(module.Module, SupportsFoo):

  # We should also be able to add new abstractmethods to the derived class,
  # since its metaclass is a subclass of ABCMeta.
  @abc.abstractmethod
  def bar(self) -> str:
    ...


class ConcreteProtocolModule(ProtocolModule):

  def foo(self):
    return 0

  def bar(self):
    return ""


class ModuleWithCustomName(module.Module):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.init_module = module.Module(name="child")

  @module.name_like("__call__")
  def foo(self):
    self.foo_module = module.Module(name="child")

  @module.name_like("baz")
  def bar(self):
    self.bar_module = module.Module(name="child")

  def baz(self):
    self.baz_module = module.Module(name="child")

  def __call__(self):
    self.call_module = module.Module(name="child")


class ModuleWithCustomNameOnCall(module.Module):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.init_module = module.Module(name="child")

  def foo(self):
    self.foo_module = module.Module(name="child")

  @module.name_like("foo")
  def __call__(self):
    self.call_module = module.Module(name="child")


class ModuleWithCustomNameOnInit(module.Module):

  @module.name_like("__call__")
  def __init__(self, name=None):
    super().__init__(name=name)
    self.init_module = module.Module(name="child")

  def __call__(self):
    self.call_module = module.Module(name="child")


class ModuleWithDoubleCall(module.Module):

  @module.name_like("__call__")
  def foo(self):
    self.foo_module = module.Module(name="child")

  def __call__(self):
    self.foo()
    self.call_module = module.Module(name="child")

if __name__ == "__main__":
  absltest.main()
