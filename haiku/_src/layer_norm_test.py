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
"""Tests for haiku._src.layer_norm."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import initializers
from haiku._src import layer_norm
from haiku._src import test_utils
import jax.numpy as jnp
import numpy as np


class LayerNormTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_connection(self):
    data = jnp.zeros([2, 3, 4, 5])
    norms = []
    for axis in range(4):
      norms.append(layer_norm.LayerNorm(axis=axis, create_scale=True,
                                        create_offset=True)(data))

    norms.append(layer_norm.LayerNorm(axis=slice(1, None), create_scale=True,
                                      create_offset=True)(data))
    norms.append(layer_norm.LayerNorm(axis=slice(2, None), create_scale=True,
                                      create_offset=True)(data))
    norms.append(layer_norm.LayerNorm(axis=slice(1, -1), create_scale=True,
                                      create_offset=True)(data))

    return norms

  @test_utils.transform_and_run
  def test_simple_case(self):
    layer = layer_norm.LayerNorm([1, 2],
                                 create_scale=False,
                                 create_offset=False)
    inputs = np.ones([2, 3, 3, 5])

    outputs = layer(inputs)
    for x in np.nditer(outputs):
      self.assertEqual(x, 0.0)

  @test_utils.transform_and_run
  def test_simple_case_var(self):
    layer = layer_norm.LayerNorm([1, 2],
                                 create_scale=True,
                                 create_offset=True,
                                 scale_init=initializers.Constant(0.5),
                                 offset_init=initializers.Constant(2.0))

    inputs = np.ones([2, 3, 3, 5])

    outputs = layer(inputs)
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  @test_utils.transform_and_run
  def test_simple_case_tensor(self):
    layer = layer_norm.LayerNorm([1, 2],
                                 create_scale=False,
                                 create_offset=False)

    inputs = np.ones([2, 3, 3, 5])
    scale = np.full((5,), 0.5)
    offset = np.full((5,), 2.0)

    outputs = layer(inputs, scale, offset)
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  @parameterized.named_parameters(("String", "foo"), ("ListString", ["foo"]))
  @test_utils.transform_and_run
  def test_invalid_axis(self, axis):
    with self.assertRaisesRegex(
        ValueError, "`axis` should be an int, slice or iterable of ints."):
      layer_norm.LayerNorm(axis, create_scale=False, create_offset=False)

  @test_utils.transform_and_run
  def test_no_scale_and_init_provided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `scale_init` if `create_scale=False`."):
      layer_norm.LayerNorm(
          3, create_scale=False, create_offset=True, scale_init=np.ones)

  @test_utils.transform_and_run
  def test_no_offset_beta_init_provided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `offset_init` if `create_offset=False`."):
      layer_norm.LayerNorm(
          3, create_scale=True, create_offset=False, offset_init=np.zeros)

  @test_utils.transform_and_run
  def test_create_scale_and_scale_provided(self):
    layer = layer_norm.LayerNorm([2], create_scale=True, create_offset=False)

    with self.assertRaisesRegex(
        ValueError, "Cannot pass `scale` at call time if `create_scale=True`."):
      layer(np.ones([2, 3, 4]), scale=np.ones([4]))

  @test_utils.transform_and_run
  def test_create_offset_and_offset_provided(self):
    layer = layer_norm.LayerNorm([2], create_offset=True, create_scale=False)

    with self.assertRaisesRegex(
        ValueError,
        "Cannot pass `offset` at call time if `create_offset=True`."):
      layer(np.ones([2, 3, 4]), offset=np.ones([4]))

  @test_utils.transform_and_run
  def test_slice_axis(self):
    slice_layer = layer_norm.LayerNorm(
        slice(1, -1), create_scale=False, create_offset=False)
    axis_layer = layer_norm.LayerNorm((1, 2),
                                      create_scale=False,
                                      create_offset=False)

    inputs = np.random.uniform(size=[3, 4, 4, 5], low=0, high=10)
    scale = np.random.normal(size=(5,), loc=1.0)
    offset = np.random.normal(size=(5,))

    slice_outputs = slice_layer(inputs, scale, offset)
    axis_outputs = axis_layer(inputs, scale, offset)

    np.testing.assert_array_equal(slice_outputs, axis_outputs)

  @test_utils.transform_and_run
  def test_connection_instance_norm(self):
    layer = layer_norm.InstanceNorm(create_scale=True, create_offset=True)

    inputs = np.ones([3, 4, 5, 6])
    result = layer(inputs)

    self.assertEqual(result.shape, (3, 4, 5, 6))


if __name__ == "__main__":
  absltest.main()
