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
"""Tests for haiku._src.stateful."""

from absl.testing import absltest
from haiku._src import base
from haiku._src import module
from haiku._src import stateful
from haiku._src import test_utils
from haiku._src import transform

import jax
import jax.numpy as jnp
import numpy as np


class StatefulTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_grad(self):
    x = jnp.array(3.)
    g = stateful.grad(SquareModule())(x)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-4)

  def test_grad_no_transform(self):
    x = jnp.array(3.)
    with self.assertRaises(ValueError, msg="Use jax.grad() instead"):
      stateful.grad(lambda x: x**2)(x)

  @test_utils.transform_and_run
  def test_value_and_grad(self):
    x = jnp.array(2.)
    y, g = stateful.value_and_grad(SquareModule())(x)
    self.assertEqual(y, x ** 2)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-4)

  def test_value_and_grad_no_transform(self):
    x = jnp.array(3.)
    with self.assertRaises(ValueError, msg="Use jax.grad() instead"):
      stateful.value_and_grad(lambda x: x**2)(x)

  @test_utils.transform_and_run
  def test_grad_aux(self):
    o = object()

    def f(x):
      m = SquareModule()
      return m(x), o

    x = jnp.array(3.)
    g, aux = stateful.grad(f, has_aux=True)(x)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-4)
    self.assertIs(aux, o)

  @test_utils.transform_and_run
  def test_value_and_grad_aux(self):
    o = object()

    def f(x):
      m = SquareModule()
      return m(x), o

    x = jnp.array(3.)
    (y, aux), g = stateful.value_and_grad(f, has_aux=True)(x)
    self.assertEqual(y, x ** 2)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-4)
    self.assertIs(aux, o)

  def test_grad_and_jit(self):
    def f(x):
      g = stateful.grad(SquareModule())(x)
      return g

    x = jnp.array(3.)
    f = transform.transform_with_state(f)
    params, state = jax.jit(f.init)(None, x)
    g, state = jax.jit(f.apply)(params, state, None, x)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-3)

  def test_value_and_grad_and_jit(self):
    def f(x):
      y, g = stateful.value_and_grad(SquareModule())(x)
      return y, g

    x = jnp.array(3.)
    f = transform.transform_with_state(f)
    params, state = jax.jit(f.init)(None, x)
    (y, g), state = jax.jit(f.apply)(params, state, None, x)
    np.testing.assert_allclose(y, x ** 2, rtol=1e-3)
    np.testing.assert_allclose(g, 2 * x, rtol=1e-3)

  @test_utils.transform_and_run
  def test_jit(self):
    mod = SquareModule()
    x = jnp.array(2)
    y = stateful.jit(mod)(x)
    self.assertEqual(y, x ** 2)

  def test_jit_no_transform(self):
    x = jnp.array(2)
    with self.assertRaises(ValueError, msg="Use jax.jit() instead"):
      stateful.jit(lambda x: x**2)(x)

  @test_utils.transform_and_run
  def test_remat(self):
    forward, backward = [], []
    callback = _callback_prim(lambda: forward.append(None),
                              lambda: backward.append(None))

    def test(remat):
      x = jnp.array(3.)
      mod = CountingModule()
      self.assertEqual(mod.count, 0)
      f = lambda x: callback(mod(x))
      if remat:
        f = stateful.remat(f)
      y, g = stateful.value_and_grad(f)(x)
      np.testing.assert_allclose(y, x ** 2, rtol=1e-3)
      np.testing.assert_allclose(g, 2 * x, rtol=1e-3)
      self.assertEqual(mod.count, 1)
      num_forward = len(forward)
      num_backward = len(backward)
      del forward[:], backward[:]
      return num_forward, num_backward

    # Sanity check.
    self.assertEqual(test(remat=True), test(remat=True))
    self.assertEqual(test(remat=False), test(remat=False))

    # NOTE: JAX does not guarantee to execute primitives once and only once for
    # a given function (we observe f=2,b=1 without remat and f=5,b=1 with
    # remat), but we do expect that JAX will execute our primitive forward at
    # least one more time with remat than without it.
    num_forward_remat, num_backward_remat = test(remat=True)
    num_forward_no_remat, num_backward_no_remat = test(remat=False)
    self.assertGreater(num_forward_remat, num_forward_no_remat)
    self.assertEqual(num_backward_remat, num_backward_no_remat)

  def test_remat_no_transform(self):
    x = jnp.array(3.)
    with self.assertRaises(ValueError, msg="Use jax.remat() instead"):
      stateful.remat(lambda x: x**2)(x)

  def test_cond(self):
    def f(x):
      mod = SquareModule()
      return stateful.cond(x == 2, x, mod, x, lambda x: mod(x + 1))
    f = transform.transform_with_state(f)
    for x, y in ((1, 4), (2, 4), (3, 16)):
      x, y = map(jnp.array, (x, y))
      params, state = f.init(None, x)
      out, state = f.apply(params, state, None, x)
      self.assertEqual(state, {"square_module": {"y": y}})
      self.assertEqual(out, y)

  def test_cond_no_transform(self):
    x = jnp.array(3.)
    with self.assertRaises(ValueError, msg="Use jax.cond() instead"):
      stateful.cond(x == 2, x, lambda x: x**2, x, lambda x: (x + 1)**2)


def _callback_prim(forward, backward):
  def f_impl(x):
    forward()
    return x

  def b_impl(x):
    backward()
    return (x,)

  prim = jax.core.Primitive("hk_callback")
  prim.def_impl(f_impl)
  prim.def_abstract_eval(f_impl)
  jax.ad.deflinear(prim, b_impl)
  return prim.bind


class CountingModule(module.Module):

  @property
  def count(self):
    return base.get_state("count", [], init=jnp.zeros)

  def __call__(self, x):
    y = x ** 2
    base.set_state("count", self.count + 1)
    return y


class SquareModule(module.Module):

  def __call__(self, x):
    assert x.ndim == 0
    p = base.get_parameter("p", [], jnp.int32, init=lambda *_: jnp.array(2))
    y = x ** p
    base.set_state("y", y)
    return y


if __name__ == "__main__":
  absltest.main()
