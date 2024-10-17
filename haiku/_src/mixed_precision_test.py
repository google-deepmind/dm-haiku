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
"""Tests for haiku._src.mixed_precision."""

import importlib

from absl.testing import absltest
from haiku._src import base
from haiku._src import conv
from haiku._src import mixed_precision
from haiku._src import module
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp
import jmp


def with_policy(cls: type[module.Module], policy: jmp.Policy | None):
  def decorator(f):
    def wrapper(*args, **kwargs):
      with mixed_precision.push_policy(cls, policy):
        return f(*args, **kwargs)
    return wrapper
  return decorator


class OuterModule(module.Module):

  def __call__(self):
    self.w = base.get_parameter('w', [], jnp.bfloat16, init=jnp.ones)
    self.inner = InnerModule()
    self.inner_ret = self.inner()
    return jnp.ones([], dtype=jnp.bfloat16)


class InnerModule(module.Module):

  def __call__(self):
    self.w = base.get_parameter('w', [], jnp.bfloat16, init=jnp.ones)
    return jnp.ones([], dtype=jnp.bfloat16)

  class InnerInnerModule(module.Module):

    def __call__(self):
      self.w = base.get_parameter('w', [], jnp.bfloat16, init=jnp.ones)
      return jnp.ones([], dtype=jnp.bfloat16)


def transform_and_run_once(f, *args, **kwargs):
  f = transform.transform(f)
  def g(*args, **kwargs):
    rng = jax.random.PRNGKey(28)
    params = f.init(rng, *args, **kwargs)
    out = f.apply(params, None, *args, **kwargs)
    return params, out
  return jax.tree.map(lambda x: x.dtype, jax.eval_shape(g, *args, **kwargs))


class MixedPrecisionTest(absltest.TestCase):

  def test_get_policy(self):
    self.assertIsNone(mixed_precision.get_policy(InnerModule))
    policy = jmp.get_policy('p=f16,c=f32,o=f16')
    mixed_precision.set_policy(InnerModule, policy)
    self.assertEqual(mixed_precision.get_policy(InnerModule), policy)
    mixed_precision.clear_policy(InnerModule)
    self.assertIsNone(mixed_precision.get_policy(InnerModule))

  @test_utils.transform_and_run
  def test_current_policy(self):
    policy = jmp.get_policy('p=f16,c=f32,o=f16')
    test = self

    class Foo(module.Module):

      def __call__(self):
        test.assertEqual(mixed_precision.current_policy(), policy)

    class Bar(module.Module):

      def __call__(self):
        test.assertEqual(mixed_precision.current_policy(), policy)
        Foo()()
        test.assertEqual(mixed_precision.current_policy(), policy)

    class Baz(module.Module):

      def __call__(self):
        test.assertIsNone(mixed_precision.current_policy())
        Bar()()
        test.assertIsNone(mixed_precision.current_policy())

    mixed_precision.set_policy(Bar, policy)
    Baz()()

  def test_set_global_policy(self):
    self.assertGlobalPolicy(InnerModule)

  def test_set_global_policy_inner_class(self):
    self.assertGlobalPolicy(InnerModule.InnerInnerModule)

  def test_set_global_policy_local_class(self):
    class LocalModule(InnerModule):
      pass

    self.assertGlobalPolicy(LocalModule)

  def assertGlobalPolicy(self, cls):
    policy = jmp.get_policy('p=f16,c=f32,o=f16')
    with_policy(cls, policy)(self.assertGlobalPolicy_inner)(cls)

  def assertGlobalPolicy_inner(self, cls):
    def f():
      mod = cls(name='inner_module')
      return mod(), mod.w

    params, (ret, w) = transform_and_run_once(f)

    self.assertEqual(ret, jnp.float16)
    self.assertEqual(w, jnp.float32)
    self.assertEqual(params['inner_module'], {'w': jnp.float16})

  @test_utils.transform_and_run
  def test_set_policy_factory(self):
    def factory():
      class MyModule(module.Module):

        def __call__(self, x):
          return x

      return MyModule

    cls1 = factory()
    cls2 = factory()

    mixed_precision.set_policy(cls1, jmp.get_policy('o=f16'))
    mixed_precision.set_policy(cls2, jmp.get_policy('o=bf16'))
    x = jnp.ones([])
    self.assertEqual(cls1()(x).dtype, jnp.float16)
    self.assertEqual(cls2()(x).dtype, jnp.bfloat16)

  @test_utils.transform_and_run
  def test_push_policy(self):
    policy = jmp.get_policy('o=f16')
    test = self

    class FooModule(module.Module):

      def __call__(self):
        test.assertEqual(mixed_precision.current_policy(), policy)

    mod = FooModule()
    with mixed_precision.push_policy(FooModule, policy):
      self.assertEqual(mixed_precision.get_policy(FooModule), policy)
      mod()

    self.assertIsNone(mixed_precision.get_policy(FooModule))

  @test_utils.transform_and_run
  def test_push_policy_maintains_old_policy(self):
    old_policy = jmp.get_policy('o=f16')
    new_policy = jmp.get_policy('o=f64')
    self.assertIsNone(mixed_precision.get_policy(InnerModule))
    mixed_precision.set_policy(InnerModule, old_policy)
    with mixed_precision.push_policy(InnerModule, new_policy):
      self.assertEqual(mixed_precision.get_policy(InnerModule), new_policy)
    self.assertEqual(mixed_precision.get_policy(InnerModule), old_policy)
    mixed_precision.clear_policy(InnerModule)

  @test_utils.transform_and_run
  def test_push_policy_not_allowed_in_method_of_same_class(self):
    any_policy = jmp.get_policy('o=f16')

    class PushesInMethod(module.Module):

      def __call__(self):
        with mixed_precision.push_policy(PushesInMethod, any_policy):
          pass

    mod = PushesInMethod()
    with self.assertRaisesRegex(ValueError, 'same class is not supported'):
      mod()

  @with_policy(InnerModule, jmp.get_policy('p=f16,c=f32,o=f16'))
  def test_clear_global_policy(self):
    def f():
      mod = InnerModule()
      return mod(), mod.w

    mixed_precision.clear_policy(InnerModule)

    params, (ret, w) = transform_and_run_once(f)

    self.assertEqual(ret, jnp.bfloat16)
    self.assertEqual(w, jnp.bfloat16)
    self.assertEqual(params['inner_module'], {'w': jnp.bfloat16})

  @with_policy(OuterModule, jmp.get_policy('p=f32,c=f16,o=f32'))
  @with_policy(InnerModule, jmp.get_policy('p=f16,c=f32,o=f32'))
  def test_set_global_policy_nested(self):
    def f():
      outer = OuterModule()
      outer_ret = outer()
      return outer_ret, outer.inner_ret, outer.w, outer.inner.w

    params, (outer_ret, inner_ret, outer_w, inner_w) = transform_and_run_once(f)

    # The return type of the modules should use the output type of the module.
    self.assertEqual(outer_ret, jnp.float32)
    self.assertEqual(inner_ret, jnp.float32)
    # Inside the module we should use the compute type of the policy.
    self.assertEqual(outer_w, jnp.float16)
    self.assertEqual(inner_w, jnp.float32)
    # The parameters returned from init should use the param type of the policy.
    self.assertEqual(params['outer_module'], {'w': jnp.float32})
    self.assertEqual(params['outer_module/inner_module'], {'w': jnp.float16})

  def test_policy_for_reloaded_class(self):
    conv_local = conv

    policy = jmp.get_policy('p=f16,c=f32,o=f16')
    mixed_precision.set_policy(conv_local.ConvND, policy)
    conv_local = importlib.reload(conv)

    params, y = transform_and_run_once(
        lambda: conv_local.ConvND(2, 1, 1)(jnp.ones([1, 1, 1, 1])))

    jax.tree.map(lambda p: self.assertEqual(p, jnp.float16), params)
    self.assertEqual(y, jnp.float16)

  @test_utils.transform_and_run
  def test_policy_with_interceptor(self):
    sidechannel = []
    def my_interceptor(next_f, args, kwargs, context):
      sidechannel.append(context)
      return next_f(*args, **kwargs)

    # We need this to make sure that the mixed precision interceptor is
    # installed when we call set_policy (this only happens the first call).
    mixed_precision.reset_thread_local_state_for_test()

    policy = jmp.get_policy('p=f16,c=f32,o=f16')
    with module.intercept_methods(my_interceptor):
      mixed_precision.set_policy(OuterModule, policy)
      x = OuterModule()()
      self.assertEqual(x.dtype, jnp.float16)
    # Outer.init, Outer.call, Inner.init, Inner.call
    self.assertLen(sidechannel, 4)

if __name__ == '__main__':
  absltest.main()
