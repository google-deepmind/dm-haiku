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
"""Common utilities."""

from absl.testing import parameterized
import haiku as hk
from haiku._src.integration import descriptors
from haiku._src.typing import DType, Shape  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp
import tree


class DTypeTestCase(parameterized.TestCase):
  """Common base class for dtype tests."""

  def assert_dtype(
      self,
      test_dtype: DType,
      module_fn: descriptors.ModuleFn,
      shape: Shape,
      input_dtype: DType,
  ):
    """Checks that modules accepting float32 input_dtype output test_dtype."""
    if jax.local_devices()[0].platform != 'tpu':
      self.skipTest('bfloat16 only supported on TPU')

    if input_dtype != jnp.float32:
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
    params, state = jax.eval_shape(init_fn, rng, x)

    # Cast f32 to test_dtype.
    def make_param(v):
      dtype = test_dtype if v.dtype == jnp.float32 else v.dtype
      return jnp.ones(v.shape, dtype)
    params, state = jax.tree_map(make_param, (params, state))

    # test_dtype in should result in test_dtype out.
    x = x.astype(test_dtype)

    for _ in range(2):
      y, state = jax.eval_shape(apply_fn, params, state, rng, x)

      def assert_dtype(path, v):
        if v.dtype != jnp.int32:
          self.assertEqual(v.dtype, test_dtype, msg=path)

      tree.map_structure_with_path(assert_dtype, y)
      tree.map_structure_with_path(assert_dtype, state)
