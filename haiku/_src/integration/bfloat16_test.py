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
"""Tests for haiku._src.conformance.bfloat16_test."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
from haiku._src.typing import DType, Shape  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp


class Bfloat16Test(parameterized.TestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_bfloat16(
      self,
      module_fn: descriptors.ModuleFn,
      shape: Shape,
      dtype: DType,
  ):
    if jax.local_devices()[0].platform != 'tpu':
      self.skipTest('bfloat16 only supported on TPU')

    if dtype != jnp.float32:
      self.skipTest('Skipping module without float32 input')

    rng = jax.random.PRNGKey(42)

    def g(x):
      mod = module_fn()
      return mod(x)

    init_fn, apply_fn = hk.transform_with_state(g)

    # Create state in f32 to start.
    # NOTE: We need to do this since some initializers (e.g. random.uniform) do
    # not support <32bit dtypes.
    x = jax.random.uniform(rng, shape)
    params, state = init_fn(rng, x)

    # Cast f32 to bf16.
    f32_to_bf16 = (
        lambda v: v.astype(jnp.bfloat16) if v.dtype == jnp.float32 else v)
    params, state = jax.tree_map(f32_to_bf16, (params, state))

    # bf16 in should result in bf16 out.
    x = x.astype(jnp.bfloat16)

    for _ in range(2):
      y, state = apply_fn(params, state, rng, x)

      def assert_bf16(v):
        if v.dtype != jnp.int32:
          self.assertEqual(v.dtype, jnp.bfloat16)

      jax.tree_map(assert_bf16, y)
      jax.tree_map(assert_bf16, state)

if __name__ == '__main__':
  absltest.main()
