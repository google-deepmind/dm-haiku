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
from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import inspect
from typing import Optional, Protocol, TypeVar, runtime_checkable

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import base
from haiku._src import config
from haiku._src import module
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp

ModuleT = TypeVar("ModuleT", bound=module.Module)


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
  def test_module_naming_explicit_numbering_zero_padded(self):
    self.assertEqual(
        EmptyModule(name="custom_name_000").module_name, "custom_name_000")
    self.assertEqual(
        EmptyModule(name="custom_name_001").module_name, "custom_name_001")
    self.assertEqual(
        EmptyModule(name="custom_name_002").module_name, "custom_name_002")
    self.assertEqual(
        EmptyModule(name="custom_name_007").module_name, "custom_name_007")

  @test_utils.transform_and_run
  def test_module_naming_explicit_numbering_zero_padded_reuse(self):
    self.assertEqual(
        EmptyModule(name="custom_name_007").module_name, "custom_name_007")
    self.assertEqual(
        EmptyModule(name="custom_name_007").module_name, "custom_name_007_1")

  @test_utils.transform_and_run
  def test_module_naming_explicit_numbering_zero_padded_vs_no_pad(self):
    m1 = ScalarModule(name="scalar_module_1")
    self.assertEqual(m1.module_name, "scalar_module_1")
    m2 = ScalarModule(name="scalar_module_001")
    self.assertEqual(m2.module_name, "scalar_module_001")
    self.assertIsNot(m1(), m2())  # No parameter sharing.

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
        self.assertEqual(mod.params_dict(), {f"scalar_module_{i}/w": w})
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
        self.assertEqual(mod.state_dict(), {f"scalar_state_module_{i}/w": w})
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
      self.assertLen(context, 4)
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
  def test_intercept_methods_orig_class(self):
    class A(module.Module):
      def __call__(self):
        pass

    class B(A):
      def __call__(self):  # pylint: disable=useless-parent-delegation
        return super().__call__()

    class C(B):
      def __init__(self, name=None):
        super().__init__(name=name)

    log = []

    def log_orig_class(f, args, kwargs, context):
      log.append(
          (type(context.module), context.orig_class, context.method_name))
      return f(*args, **kwargs)

    with module.intercept_methods(log_orig_class):
      B()()
      C()()

    self.assertEqual(log, [
        # b = B()
        (B, B, "__init__"),
        # b()
        (B, B, "__call__"), (B, A, "__call__"),
        # c = C()
        (C, C, "__init__"),  # NOTE: No entry for `(module.Module, __init__)`.
        # c()
        (C, B, "__call__"), (C, A, "__call__")])

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
  def test_name_scope_method_name(self):
    with module.name_scope("a", method_name="bar"):
      self.assertEqual(module.Module().module_name, "a/~bar/module")
    with module.name_scope("b", method_name="__init__"):
      self.assertEqual(module.Module().module_name, "b/~/module")
    with module.name_scope("c", method_name="__call__"):
      self.assertEqual(module.Module().module_name, "c/module")

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

  @test_utils.transform_and_run
  def test_auto_repr(self):
    m = IdentityModule()
    self.assertEqual(str(m), "IdentityModule()")

  @test_utils.transform_and_run
  def test_repr_during_ctor(self):
    # See https://github.com/google-deepmind/dm-haiku/issues/428 for other ways
    # this can get triggered.

    test = self

    class MyModule(module.Module):
      def __init__(self):
        super().__init__()
        test.assertEqual(repr(self), "MyModule()")

    MyModule()  # Does not fail.

  def test_signature(self):
    captures_expected = inspect.Signature(
        parameters=(
            inspect.Parameter(
                name="mod", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD
            ),
        )
    )
    self.assertEqual(inspect.signature(CapturesModule), captures_expected)
    datalinear_expected = inspect.Signature(
        parameters=(
            inspect.Parameter(
                name="output_size",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=int,
            ),
            inspect.Parameter(
                name="name",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=Optional[str],
            ),
        ),
        return_annotation=None,
    )
    self.assertEqual(inspect.signature(DataLinear), datalinear_expected)

  @test_utils.transform_and_run
  @config.with_config(module_auto_repr=False)
  def test_config_disable_auto_repr(self):
    self.assertRegex(str(IdentityModule()),
                     "<.*.IdentityModule object at .*>")

  @test_utils.transform_and_run
  def test_attr_disable_auto_repr(self):
    self.assertTrue(config.get_config().module_auto_repr)
    self.assertRegex(str(NoAutoReprModule()),
                     "<.*.NoAutoReprModule object at .*>")

  @parameterized.parameters("foo", "foo/bar")
  @test_utils.transform_and_run
  def test_force_name_naming(self, name):
    m0 = create_module_from_qualified_name(name)
    m1 = module.Module(name=module.force_name(name))
    m2 = module.Module(name=module.force_name(name))
    self.assertEqual(m0.name, m1.name)
    self.assertEqual(m0.module_name, m1.module_name)
    self.assertEqual(m1.name, m2.name)
    self.assertEqual(m1.module_name, m2.module_name)

  @test_utils.transform_and_run
  def test_force_name_reserves_name(self):
    m0 = module.Module(name=module.force_name("foo"))
    m1 = module.Module(name="foo")
    self.assertEqual(m0.module_name, "foo")
    self.assertEqual(m1.module_name, "foo_1")

  @parameterized.parameters("foo", "foo/bar")
  @test_utils.transform_and_run
  def test_force_name_inside_module(self, name):
    class CreatesInnerModule(module.Module):

      def __call__(self):
        return module.Module(name=module.force_name(name))

    m0 = create_module_from_qualified_name(name)
    m1 = CreatesInnerModule()()
    m2 = module.Module(name=module.force_name(name))
    self.assertEqual(m0.module_name, m1.module_name)
    self.assertEqual(m1.module_name, m2.module_name)

  @test_utils.transform_and_run
  def test_force_name_inside_name_scope(self):
    m0 = module.Module(name="foo")
    with module.name_scope("bar"):
      m1 = module.Module(name=module.force_name("foo"))
    m2 = module.Module(name=module.force_name("foo"))
    self.assertEqual(m0.module_name, m1.module_name)
    self.assertEqual(m1.module_name, m2.module_name)

  @parameterized.parameters("foo", "foo/bar")
  @test_utils.transform_and_run
  def test_force_name_parameter_reuse(self, name):
    m0 = create_module_from_qualified_name(name=name, cls=ScalarModule)
    m1 = ScalarModule(name=module.force_name(name))
    self.assertIs(m0(), m1())

  @test_utils.transform_and_run
  def test_force_name_parameter_reuse_name_scope(self):
    m0 = create_module_from_qualified_name(name="foo/bar/baz", cls=ScalarModule)
    w0 = m0()
    with module.name_scope(module.force_name("foo/bar/baz")):
      w1 = base.get_parameter("w", [], init=jnp.zeros)
    self.assertIs(w0, w1)

  @test_utils.transform_and_run
  def test_force_name_intercept_methods(self):
    def change_prefix(old, new):
      def my_interceptor(next_f, args, kwargs, context: module.MethodContext):
        if type(context.module).__name__ == "NameScopeModule":
          # Avoid infinite recursion for modules introduced by name_scope.
          return next_f(*args, **kwargs)

        name = context.module.module_name

        # We expect all usages in the test to have this prefix. If you are
        # forking this code you can probably remove this line.
        self.assertStartsWith(name, old)

        if name.startswith(old):
          name = name.replace(old, new, 1)

        with module.name_scope(module.force_name(name),
                               method_name=context.method_name):
          return next_f(*args, **kwargs)

      return module.intercept_methods(my_interceptor)

    with module.name_scope("outer"):
      m1 = ParentModule()
    with module.name_scope("inner"):
      m2 = ParentModule()

    m1()
    with change_prefix("inner", "outer"):
      m2()
    self.assertIs(m1.child1.w, m2.child1.w)
    self.assertIs(m1.child2.w, m2.child2.w)


class NoAutoReprModule(module.Module):
  AUTO_REPR = False


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
    self.w = base.get_parameter("w", [], init=jnp.zeros)
    return self.w


class ScalarStateModule(module.Module):

  def __call__(self):
    return base.get_state("w", [], init=jnp.zeros)


class ParentModule(module.Module):

  def __init__(self):
    super().__init__()
    self.child1 = ScalarModule(name="child_module")
    self.child2 = ScalarModule(name="child_module")

  def __call__(self):
    self.child1()
    self.child2()


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
  name: str | None = None

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    w = base.get_parameter("w", [j, k], init=jnp.ones)
    b = base.get_parameter("b", [k], init=jnp.zeros)
    return x @ w + b


@dataclasses.dataclass
class DataMLP(module.Module):

  output_sizes: Sequence[int]
  activation: Callable[[jax.Array], jax.Array] = jax.nn.relu
  name: str | None = None

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


def create_module_from_qualified_name(
    name: str,
    *,
    cls: type[ModuleT] = module.Module,
) -> ModuleT:
  if "/" in name:
    prefix, suffix = name.rsplit("/", 1)
    with module.name_scope(prefix):
      return cls(name=suffix)
  else:
    return cls(name=name)


if __name__ == "__main__":
  absltest.main()
