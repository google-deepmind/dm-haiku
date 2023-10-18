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
"""Running haiku._src.integration.common.DTypeTestCase with float64.

This is put into a separate file since it requires enabling float64 right after
loading jax.
"""

from absl.testing import absltest
from haiku._src import test_utils
from haiku._src.integration import common
from haiku._src.integration import descriptors
from jax import config
import jax.numpy as jnp

ModuleFn = descriptors.ModuleFn


class Float64Test(common.DTypeTestCase):

  def setUp(self):
    super().setUp()
    config.update("jax_enable_x64", True)

  def tearDown(self):
    super().tearDown()
    config.update("jax_enable_x64", False)

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_float32(self, module_fn: ModuleFn, shape, dtype):
    self.assertTrue(config.read("jax_enable_x64"))
    self.assert_dtype(jnp.float32, module_fn, shape, dtype)

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_float64(self, module_fn: ModuleFn, shape, dtype):
    self.assertTrue(config.read("jax_enable_x64"))
    self.assert_dtype(jnp.float64, module_fn, shape, dtype)

if __name__ == "__main__":
  absltest.main()
