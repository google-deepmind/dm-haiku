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
"""Tests Haiku modules in half precision."""

from absl.testing import absltest
from haiku._src import test_utils
from haiku._src.integration import common
from haiku._src.integration import descriptors
import jax.numpy as jnp

ModuleFn = descriptors.ModuleFn


class Bfloat16Test(common.DTypeTestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_float16(self, module_fn: descriptors.ModuleFn, shape, dtype):
    self.assert_dtype(jnp.float16, module_fn, shape, dtype)

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_bfloat16(self, module_fn: descriptors.ModuleFn, shape, dtype):
    self.assert_dtype(jnp.bfloat16, module_fn, shape, dtype)

if __name__ == '__main__':
  absltest.main()
