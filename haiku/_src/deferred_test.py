# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for haiku._src.deferred."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import deferred
from haiku._src import module
from haiku._src import test_utils
import jax.numpy as jnp


class DeferredTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_target(self):
    target = ExampleModule()
    mod = deferred.Deferred(lambda: target)
    self.assertIs(mod.target, target)

  @test_utils.transform_and_run
  def test_only_computes_target_once(self):
    target = ExampleModule()
    targets = [target]
    mod = deferred.Deferred(targets.pop)  # pytype: disable=wrong-arg-types
    for _ in range(10):
      # If target was recomputed more than once pop should fail.
      self.assertIs(mod.target, target)
      self.assertEmpty(targets)

  @test_utils.transform_and_run
  def test_attr_forwarding_fails_before_construction(self):
    mod = deferred.Deferred(ExampleModule)
    with self.assertRaises(AttributeError):
      getattr(mod, "foo")

  @test_utils.transform_and_run
  def test_getattr(self):
    mod = deferred.Deferred(ExampleModule)
    mod()
    self.assertIs(mod.w, mod.target.w)  # pytype: disable=attribute-error

  @test_utils.transform_and_run
  def test_setattr(self):
    mod = deferred.Deferred(ExampleModule)
    mod()
    new_w = jnp.ones_like(mod.w)
    mod.w = new_w
    self.assertIs(mod.w, new_w)
    self.assertIs(mod.target.w, new_w)  # pytype: disable=attribute-error

  @test_utils.transform_and_run
  def test_setattr_on_target(self):
    mod = deferred.Deferred(ExampleModule)
    mod()
    w = jnp.ones_like(mod.w)
    mod.w = None
    # Assigning to the target directly should reflect in the parent.
    mod.target.w = w
    self.assertIs(mod.w, w)
    self.assertIs(mod.target.w, w)

  @test_utils.transform_and_run
  def test_delattr(self):
    mod = deferred.Deferred(ExampleModule)
    mod()
    self.assertTrue(hasattr(mod.target, "w"))
    del mod.w
    self.assertFalse(hasattr(mod.target, "w"))

  @test_utils.transform_and_run
  def test_alternative_forward(self):
    mod = deferred.Deferred(AlternativeForwardModule, call_methods=("forward",))
    self.assertEqual(mod.forward(), 42)

  @test_utils.transform_and_run
  def test_alternative_forward_call_type_error(self):
    mod = deferred.Deferred(AlternativeForwardModule, call_methods=("forward",))
    msg = "'AlternativeForwardModule' object is not callable"
    with self.assertRaisesRegex(TypeError, msg):
      mod()

  @test_utils.transform_and_run
  def test_str(self):
    m = ExampleModule()
    d = deferred.Deferred(lambda: m)
    self.assertEqual("Deferred(%s)" % m, str(d))

  @test_utils.transform_and_run
  def test_repr(self):
    m = ExampleModule()
    d = deferred.Deferred(lambda: m)
    self.assertEqual("Deferred(%r)" % m, repr(d))

  @test_utils.transform_and_run
  def test_deferred_naming_name_scope(self):
    with module.name_scope("foo"):
      d = deferred.Deferred(ExampleModule)
    mod = d.target
    self.assertEqual(mod.module_name, "foo/example_module")

  @parameterized.parameters(True, False)
  @test_utils.transform_and_run
  def test_deferred_naming_outer_module(self, call_module):
    outer = OuterModule()
    if call_module:
      outer()
    mod = outer.deferred.target
    self.assertEqual(mod.module_name, "outer/~/example_module")


class OuterModule(module.Module):

  def __init__(self, name="outer"):
    super().__init__(name=name)
    self.deferred = deferred.Deferred(ExampleModule)

  def __call__(self):
    return self.deferred()


class ExampleModule(module.Module):

  def __init__(self):
    super().__init__()
    self.w = jnp.ones([])

  def __str__(self):
    return "ExampleModuleStr"

  def __repr__(self):
    return "ExampleModuleRepr"

  def __call__(self):
    return self.w


class AlternativeForwardModule(module.Module):

  def forward(self):
    return 42


if __name__ == "__main__":
  absltest.main()
