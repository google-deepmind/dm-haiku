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
"""Tests for haiku._src.conformance.float64_test."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors
from haiku._src.integration.bfloat16_test import DTypeTestCase
from haiku._src.typing import DType, Shape  # pylint: disable=g-multiple-import
import jax
from jax.config import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)


class Float64Test(DTypeTestCase):
  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_float64(
      self,
      module_fn: descriptors.ModuleFn,
      shape: Shape,
      dtype: DType,
  ):
    self.assert_dtype(jnp.float64, module_fn, shape, dtype)

if __name__ == '__main__':
  absltest.main()
