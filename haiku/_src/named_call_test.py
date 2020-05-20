# Lint as: python3
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
"""Tests for haiku._src.named_call."""
import functools

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import named_call
import jax


class NamedCallTest(parameterized.TestCase):

  @parameterized.parameters(jax.jit, jax.grad, jax.vmap, jax.remat)
  def test_jax_transforms(self, transform):
    f = jax.numpy.sum
    x = jax.numpy.array([1.])

    unnamed_out = transform(f)(x)
    named_out = transform(named_call.stateful_named_call(f, name='test'))(x)

    self.assertEqual(unnamed_out, named_out)

  def test_static_argnums(self):
    f = named_call.stateful_named_call(lambda x, y: y if x else None,
                                       name='test')
    f = jax.jit(f, static_argnums=(0,))
    out = f(True, 5)
    self.assertEqual(out, 5)

  def test_partial_eval(self):
    f = named_call.stateful_named_call(lambda x, y: y if x else None,
                                       name='test')
    f = jax.jit(functools.partial(f, True))
    out = f(5)
    self.assertEqual(out, 5)

  def test_non_jaxtype_arg(self):
    # For the test to fail without the invalid JaxType filter we need to pass
    # in a valid JaxType that forces the invalid Jaxtype to be raised to an
    # abstract value.
    def f(not_a_jaxtype, a_jaxtype):
      # then Jax needs to try and evaluate the abstractified non-JaxType
      if not_a_jaxtype:
        return a_jaxtype
      return 0

    f = named_call.stateful_named_call(f, name='test')
    out = jax.jit(f, static_argnums=(0,))('not a Jaxtype', 1)
    self.assertEqual(out, 1)

if __name__ == '__main__':
  absltest.main()
