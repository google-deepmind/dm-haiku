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
"""Tests for haiku._src.initializers."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import initializers
from haiku._src import test_utils
import jax.numpy as jnp
import numpy as np


class InitializersTest(parameterized.TestCase):

  @parameterized.parameters(np.float32, jnp.float32)
  @test_utils.transform_and_run
  def test_initializers(self, dtype):
    # This just makes sure we can call the initializers in accordance to the
    # API and get the right shapes and dtypes out.
    inits = [
        initializers.Constant(42.0),
        initializers.RandomNormal(),
        initializers.RandomNormal(2.0),
        initializers.RandomUniform(),
        initializers.RandomUniform(3.0),
        initializers.VarianceScaling(),
        initializers.VarianceScaling(2.0),
        initializers.VarianceScaling(2.0, mode="fan_in"),
        initializers.VarianceScaling(2.0, mode="fan_out"),
        initializers.VarianceScaling(2.0, mode="fan_avg"),
        initializers.VarianceScaling(2.0, distribution="truncated_normal"),
        initializers.VarianceScaling(2.0, distribution="normal"),
        initializers.VarianceScaling(2.0, distribution="uniform"),
        initializers.UniformScaling(),
        initializers.UniformScaling(2.0),
        initializers.TruncatedNormal(),

        # Users are supposed to be able to use these.
        jnp.zeros,
        jnp.ones,
    ]

    # TODO(ibab): Test other shapes as well.
    shape = (20, 42)

    for init in inits:
      generated = init(shape, dtype)
      self.assertEqual(generated.shape, shape)
      self.assertEqual(generated.dtype, dtype)

if __name__ == "__main__":
  absltest.main()
