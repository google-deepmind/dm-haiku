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
from typing import Optional

from absl.testing import absltest
from haiku._src import conv
from haiku._src import jaxpr_info
from haiku._src import module
from haiku._src import transform
import jax
import jax.numpy as jnp
import numpy as np


class MyModel(module.Module):

  def __init__(self, name: Optional[str] = None):
    super().__init__(name=name)

  def __call__(self, x: jnp.ndarray):
    return conv.Conv2D(16, 3)(x)


class JaxprInfoTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.prev_profiler_name_scopes = module.profiler_name_scopes(enabled=True)

  def tearDown(self):
    super().tearDown()
    module.profiler_name_scopes(self.prev_profiler_name_scopes)

  def test_simple_expression(self):

    def add(x, y):
      return jnp.sign(x) + jnp.cos(y)

    a = jnp.zeros((12, 7))
    mod = jaxpr_info.make_model_info(add)(a, a)
    self.assertEqual(
        jaxpr_info.format_module(mod).strip(), """
add
  sign
    sign in f32[12,7], out f32[12,7]
  cos in f32[12,7], out f32[12,7]
  add in f32[12,7], f32[12,7], out f32[12,7]
""".strip())

  def test_compute_flops(self):

    def _compute_flops(eqn: jax.core.JaxprEqn,
                       expression: jaxpr_info.Expression) -> int:
      del expression
      return max(np.prod(var.aval.shape) for var in eqn.invars)  # pytype: disable=attribute-error

    def add(x, y):
      return jnp.sign(x) + jnp.cos(y)

    a = jnp.zeros((12, 7))
    mod = jaxpr_info.make_model_info(add, compute_flops=_compute_flops)(a, a)
    self.assertEqual(
        jaxpr_info.format_module(mod).strip(), """
add 252 flops
  sign 84 flops
    sign 84 flops in f32[12,7], out f32[12,7]
  cos 84 flops in f32[12,7], out f32[12,7]
  add 84 flops in f32[12,7], f32[12,7], out f32[12,7]
""".strip())

  def test_haiku_module(self):

    def forward(x):
      return MyModel()(x)

    forward_t = transform.transform_with_state(forward)

    rng = jax.random.PRNGKey(42)
    x = jnp.zeros((16, 8, 8, 32))
    params, state = forward_t.init(rng, x)

    mod = jaxpr_info.make_model_info(forward_t.apply)(params, state, rng, x)
    self.assertEqual(
        jaxpr_info.format_module(mod).strip(), """
apply_fn
  my_model 4.624 kparams
    conv2_d 4.624 kparams
      conv_general_dilated in f32[16,8,8,32], f32[3,3,32,16], out f32[16,8,8,16]
      broadcast_in_dim in f32[16], out f32[16,8,8,16]
      add in f32[16,8,8,16], f32[16,8,8,16], out f32[16,8,8,16]
""".strip())


if __name__ == '__main__':
  absltest.main()
