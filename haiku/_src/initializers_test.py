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
from haiku._src import initializers
from haiku._src import test_utils
import jax.numpy as jnp


class InitializersTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_initializers(self):
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
        initializers.Orthogonal(),

        # Users are supposed to be able to use these.
        jnp.zeros,
        jnp.ones,
    ]

    # TODO(ibab): Test other shapes as well.
    shape = (20, 42)

    dtype = jnp.float32
    for init in inits:
      generated = init(shape, dtype)
      self.assertEqual(generated.shape, shape)
      self.assertEqual(generated.dtype, dtype)

  @test_utils.transform_and_run
  def test_invalid_variance_scale(self):

    with self.assertRaisesRegex(ValueError, "scale.*must be a positive float"):
      initializers.VarianceScaling(scale=-1.0)

    with self.assertRaisesRegex(ValueError, "Invalid `mode` argument*"):
      initializers.VarianceScaling(mode="foo")

    with self.assertRaisesRegex(ValueError, "Invalid `distribution` argument*"):
      initializers.VarianceScaling(distribution="bar")

  @test_utils.transform_and_run
  def test_compute_fans(self):
    fan_in_out1 = initializers._compute_fans([])
    self.assertEqual(fan_in_out1, (1, 1))
    fan_in_out2 = initializers._compute_fans([2])
    self.assertEqual(fan_in_out2, (2, 2))
    fan_in_out3 = initializers._compute_fans([3, 4])
    self.assertEqual(fan_in_out3, (3, 4))
    fan_in_out4 = initializers._compute_fans([1, 2, 3, 4])
    self.assertEqual(fan_in_out4, (6, 8))

  @test_utils.transform_and_run
  def test_orthogonal_invalid_shape(self):
    init = initializers.Orthogonal()
    shape = (20,)
    with self.assertRaisesRegex(
        ValueError, "Orthogonal initializer requires at least a 2D shape."):
      init(shape, jnp.float32)

  @test_utils.transform_and_run
  def test_orthogonal_orthogonal(self):
    init = initializers.Orthogonal()
    shape = (42, 20)
    generated = init(shape, jnp.float32)
    self.assertEqual(generated.shape, shape)
    self.assertEqual(generated.dtype, jnp.float32)


if __name__ == "__main__":
  absltest.main()
