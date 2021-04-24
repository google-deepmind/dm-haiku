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

from typing import Optional, Type

from absl.testing import absltest
from haiku._src import base
from haiku._src import mixed_precision
from haiku._src import module
from haiku._src import transform
import jax
import jax.numpy as jnp
import jmp


def with_policy(cls: Type[module.Module], policy: Optional[jmp.Policy]):
  def decorator(f):
    def wrapper(*args, **kwargs):
      mixed_precision.set_policy(cls, policy)
      try:
        return f(*args, **kwargs)
      finally:
        mixed_precision.clear_policy(cls)
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


def transform_and_run_once(f, *args, **kwargs):
  f = transform.transform(f)
  def g(*args, **kwargs):
    params = f.init(None, *args, **kwargs)
    out = f.apply(params, None, *args, **kwargs)
    return params, out
  return jax.tree_map(lambda x: x.dtype, jax.eval_shape(g, *args, **kwargs))


class MixedPrecisionTest(absltest.TestCase):

  @with_policy(InnerModule, jmp.get_policy('p=f16,c=f32,o=f16'))
  def test_set_global_policy(self):
    def f():
      mod = InnerModule()
      return mod(), mod.w

    params, (ret, w) = transform_and_run_once(f)

    self.assertEqual(ret, jnp.float16)
    self.assertEqual(w, jnp.float32)
    self.assertEqual(params['inner_module'], {'w': jnp.float16})

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

if __name__ == '__main__':
  absltest.main()
