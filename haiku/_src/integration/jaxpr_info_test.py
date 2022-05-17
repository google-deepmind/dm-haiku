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
"""Tests to ensure all modules work with jaxpr_info."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
import jax
import jax.numpy as jnp

ModuleFn = descriptors.ModuleFn
jaxpr_info = hk.experimental.jaxpr_info


class JaxprInfoTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_info_and_html(self, module_fn: ModuleFn, shape, dtype):
    x = jnp.ones(shape, dtype)
    f = hk.transform_with_state(lambda: module_fn()(x))
    rng = jax.random.PRNGKey(42)
    params, state = f.init(rng)
    info = jaxpr_info.make_model_info(f.apply)(params, state, rng)
    if descriptors.module_type(module_fn).__name__ != 'Sequential':
      self.assertNotEmpty(info.expressions)
    self.assertIsNotNone(jaxpr_info.as_html_page(info))

if __name__ == '__main__':
  absltest.main()
