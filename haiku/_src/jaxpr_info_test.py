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
"""Tests for jaxpr_info."""

from absl import logging
from absl.testing import absltest
from haiku._src import conv
from haiku._src import jaxpr_info
from haiku._src import module
from haiku._src import transform
import jax
from jax.extend import core as jax_core
import jax.numpy as jnp
import numpy as np


class MyModel(module.Module):

  def __init__(self, name: str | None = None):
    super().__init__(name=name)

  def __call__(self, x: jax.Array):
    return conv.Conv2D(16, 3)(x)


class JaxprInfoTest(absltest.TestCase):

  def test_simple_expression(self):

    def add(x, y):
      return jnp.sign(x) + jnp.cos(y)

    a = jnp.zeros((12, 7))
    mod = jaxpr_info.make_model_info(add)(a, a)
    if jax.__version_info__ < (0, 4, 24):
      expected = """
add
  sign
    sign in f32[12,7], out f32[12,7]
  cos in f32[12,7], out f32[12,7]
  add in f32[12,7], f32[12,7], out f32[12,7]
"""
    else:
      expected = """
add
  sign in f32[12,7], out f32[12,7]
  cos in f32[12,7], out f32[12,7]
  add in f32[12,7], f32[12,7], out f32[12,7]
"""
    self.assertContentsEqual(jaxpr_info.format_module(mod), expected)

  def test_compute_flops(self):

    def _compute_flops(eqn: jax_core.JaxprEqn,
                       expression: jaxpr_info.Expression) -> int:
      del expression
      return max(np.prod(var.aval.shape) for var in eqn.invars)  # pytype: disable=attribute-error

    def add(x, y):
      return jnp.sign(x) + jnp.cos(y)

    a = jnp.zeros((12, 7))
    mod = jaxpr_info.make_model_info(add, compute_flops=_compute_flops)(a, a)
    # jnp.sign implementation changed in jax v0.4.24
    if jax.__version_info__ < (0, 4, 24):
      expected = """
add 252 flops
  sign 84 flops
    sign 84 flops in f32[12,7], out f32[12,7]
  cos 84 flops in f32[12,7], out f32[12,7]
  add 84 flops in f32[12,7], f32[12,7], out f32[12,7]
"""
    else:
      expected = """
add 252 flops
  sign 84 flops in f32[12,7], out f32[12,7]
  cos 84 flops in f32[12,7], out f32[12,7]
  add 84 flops in f32[12,7], f32[12,7], out f32[12,7]
"""
    self.assertContentsEqual(jaxpr_info.format_module(mod), expected)

  def test_haiku_module(self):

    def forward(x):
      return MyModel()(x)

    forward_t = transform.transform_with_state(forward)

    rng = jax.random.PRNGKey(42)
    x = jnp.zeros((16, 8, 8, 32))
    params, state = forward_t.init(rng, x)

    mod = jaxpr_info.make_model_info(forward_t.apply)(params, state, rng, x)
    self.assertContentsEqual(
        jaxpr_info.format_module(mod), """
apply_fn
  my_model 4.624 kparams
    conv2_d 4.624 kparams
      conv_general_dilated in f32[16,8,8,32], f32[3,3,32,16], out f32[16,8,8,16]
      broadcast_in_dim in f32[16], out f32[16,8,8,16]
      add in f32[16,8,8,16], f32[16,8,8,16], out f32[16,8,8,16]
""")

  def test_haiku_module_loss(self):

    def forward(x):
      return MyModel()(x)

    forward_t = transform.transform_with_state(forward)

    rng = jax.random.PRNGKey(42)
    x = jnp.zeros((16, 8, 8, 32))
    params, state = forward_t.init(rng, x)

    def loss(params, state, rng, x):
      loss = jnp.sum(forward_t.apply(params, state, rng, x)[0])
      return loss, loss

    grad = jax.grad(loss, has_aux=True)
    mod = jaxpr_info.make_model_info(grad)(params, state, rng, x)
    formatted_mod = jaxpr_info.format_module(mod)

    # Support old JAX versions on GitHub presubmits.
    formatted_mod = formatted_mod.replace("transpose(jvp(conv2_d))",
                                          "conv2_d").replace(
                                              "jvp(conv2_d)", "conv2_d")

    self.assertContentsEqual(
        formatted_mod, """
loss
  jvp(my_model)
    conv2_d
      conv_general_dilated in f32[16,8,8,32], f32[3,3,32,16], out f32[16,8,8,16]
      broadcast_in_dim in f32[16], out f32[16,8,8,16]
      add in f32[16,8,8,16], f32[16,8,8,16], out f32[16,8,8,16]
  reduce_sum in f32[16,8,8,16], out f32[]
  broadcast_in_dim in f32[], out f32[16,8,8,16]
  transpose(jvp(my_model))
    conv2_d
      reduce_sum in f32[16,8,8,16], out f32[16]
      conv_general_dilated in f32[16,8,8,32], f32[16,8,8,16], out f32[3,3,32,16]
""".strip())

  def assertContentsEqual(self, a: str, b: str):
    a, b = a.strip(), b.strip()
    logging.info("a:\n%s", a)
    logging.info("b:\n%s", b)
    self.assertEqual(a, b)

if __name__ == "__main__":
  absltest.main()
