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
import jax
import jax.numpy as jnp
import numpy as np
import tree

ModuleFn = descriptors.ModuleFn


class DTypeTestCase(parameterized.TestCase):
  """Common base class for dtype tests."""

  def assert_dtype(self, test_dtype, module_fn: ModuleFn, shape, input_dtype):
    """Checks that modules accepting float32 input_dtype output test_dtype."""

    if input_dtype != jnp.float32:
      self.skipTest('Skipping module with non-f32 input')

    def ones_creator(next_creator, shape, dtype, init, context):
      if context.full_name == 'vector_quantizer/embeddings':
        # NOTE: vector_quantizer/embeddings is created using a ctor argument
        # so dtype is not expected to follow input to __call__.
        dtype = test_dtype
      else:
        self.assertEqual(dtype, test_dtype, msg=context.full_name)

      # NOTE: We need to do this since some initializers (e.g. random.uniform)
      # do not support <32bit dtypes. This also makes the test run a bit faster.
      init = jnp.ones
      return next_creator(shape, dtype, init)

    def g(x):
      with hk.experimental.custom_creator(ones_creator):
        mod = module_fn()
        return mod(x)

    g = hk.transform_with_state(g)

    # No custom creator for state so we need to do this manually.
    def cast_if_floating(x):
      if jnp.issubdtype(x.dtype, jnp.floating):
        x = x.astype(test_dtype)
      return x

    def init_fn(rng, x):
      params, state = g.init(rng, x)
      state = jax.tree_map(cast_if_floating, state)
      return params, state

    x = np.ones(shape, test_dtype)
    rng = jax.random.PRNGKey(42)
    params, state = jax.eval_shape(init_fn, rng, x)

    for _ in range(2):
      y, state = jax.eval_shape(g.apply, params, state, rng, x)

      def assert_dtype(path, v):
        if jnp.issubdtype(v.dtype, jnp.floating):
          self.assertEqual(v.dtype, test_dtype, msg=path)

      tree.map_structure_with_path(assert_dtype, y)
      tree.map_structure_with_path(assert_dtype, state)
