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
"""Tests for haiku._src.group_norm."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import group_norm
from haiku._src import initializers
from haiku._src import test_utils
import jax.numpy as jnp
import numpy as np


def constant(fill_value, *, shape):
  return np.full(shape, fill_value, np.float32)


class GroupNormTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_simple_case(self):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    inputs = jnp.ones([2, 3, 3, 10])

    outputs = layer(inputs)
    for x in np.nditer(outputs):
      self.assertEqual(x, 0.0)

  @test_utils.transform_and_run
  def test_simple_case_var(self):
    layer = group_norm.GroupNorm(
        groups=5,
        create_scale=True,
        create_offset=True,
        scale_init=initializers.Constant(0.5),
        offset_init=initializers.Constant(2.0))

    inputs = jnp.ones([2, 3, 3, 10])

    outputs = layer(inputs)
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  @test_utils.transform_and_run
  def test_simple_case_nchwvar(self):
    layer = group_norm.GroupNorm(
        groups=5,
        create_scale=True,
        create_offset=True,
        scale_init=initializers.Constant(0.5),
        offset_init=initializers.Constant(2.0),
        data_format="NCHW")

    inputs = jnp.ones([2, 10, 3, 3])

    outputs = layer(inputs)
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  @test_utils.transform_and_run
  def test_data_format_agnostic_var(self):
    c_last_layer = group_norm.GroupNorm(
        groups=5, create_scale=True, create_offset=True)
    c_first_layer = group_norm.GroupNorm(
        groups=5, create_scale=True, create_offset=True, data_format="NCHW")

    inputs = np.random.uniform(0, 10, [3, 4, 4, 10]).astype(np.float32)

    c_last_output = c_last_layer(inputs)
    inputs = jnp.transpose(inputs, [0, 3, 1, 2])
    c_first_output = c_first_layer(inputs)
    c_first_output = jnp.transpose(c_first_output, [0, 2, 3, 1])

    self.assertAllClose(c_last_output, c_first_output)

  @test_utils.transform_and_run
  def test_simple_case_tensor(self):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)

    inputs = jnp.ones([2, 3, 3, 10])
    scale = constant(0.5, shape=(10,))
    offset = constant(2.0, shape=(10,))

    outputs = layer(inputs, scale, offset)
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  @test_utils.transform_and_run
  def test_simple_case_nchwtensor(self):
    layer = group_norm.GroupNorm(
        groups=5, data_format="NCHW", create_scale=False, create_offset=False)

    inputs = jnp.ones([2, 10, 3, 3])
    scale = constant(0.5, shape=(10, 1, 1))
    offset = constant(2.0, shape=(10, 1, 1))

    outputs = layer(inputs, scale, offset)
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  @test_utils.transform_and_run
  def test_data_format_agnostic_tensor(self):
    c_last = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    c_first = group_norm.GroupNorm(
        groups=5, data_format="NCHW", create_scale=False, create_offset=False)

    inputs = np.random.uniform(0, 10, [3, 4, 4, 10]).astype(np.float32)
    scale = np.random.normal(size=(10,), loc=1.0)
    offset = np.random.normal(size=(10,))

    c_last_output = c_last(inputs, scale, offset)
    inputs = jnp.transpose(inputs, [0, 3, 1, 2])
    scale = jnp.reshape(scale, (10, 1, 1))
    offset = jnp.reshape(offset, (10, 1, 1))
    c_first_output = c_first(inputs, scale, offset)
    c_first_output = jnp.transpose(c_first_output, [0, 2, 3, 1])

    self.assertAllClose(c_last_output, c_first_output, rtol=1e-5)

  @parameterized.parameters("NHW", "HWC", "channel_last")
  @test_utils.transform_and_run
  def test_invalid_data_format(self, data_format):
    with self.assertRaisesRegex(
        ValueError,
        "Unable to extract channel information from '{}'.".format(data_format)):
      group_norm.GroupNorm(
          groups=5,
          data_format=data_format,
          create_scale=False,
          create_offset=False)

  @parameterized.parameters("NCHW", "NCW", "channels_first")
  @test_utils.transform_and_run
  def test_valid_data_format_channels_first(self, data_format):
    test = group_norm.GroupNorm(
        groups=5,
        data_format=data_format,
        create_scale=False,
        create_offset=False)

    self.assertEqual(test.channel_index, 1)

  @parameterized.parameters("NHWC", "NWC", "channels_last")
  @test_utils.transform_and_run
  def test_valid_data_format_channels_last(self, data_format):
    test = group_norm.GroupNorm(
        groups=5,
        data_format=data_format,
        create_scale=False,
        create_offset=False)

    self.assertEqual(test.channel_index, -1)

  @parameterized.named_parameters(("String", "foo"), ("ListString", ["foo"]))
  @test_utils.transform_and_run
  def test_invalid_axis(self, axis):
    with self.assertRaisesRegex(
        ValueError, "`axis` should be an int, slice or iterable of ints."):
      group_norm.GroupNorm(
          groups=5, axis=axis, create_scale=False, create_offset=False)

  @test_utils.transform_and_run
  def test_no_scale_and_init_provided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `scale_init` if `create_scale=False`."):
      group_norm.GroupNorm(
          groups=5,
          create_scale=False,
          create_offset=True,
          scale_init=jnp.ones)

  @test_utils.transform_and_run
  def test_no_offset_beta_init_provided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `offset_init` if `create_offset=False`."):
      group_norm.GroupNorm(
          groups=5,
          create_scale=True,
          create_offset=False,
          offset_init=jnp.zeros)

  @test_utils.transform_and_run
  def test_create_scale_and_scale_provided(self):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=True, create_offset=False)

    with self.assertRaisesRegex(
        ValueError, "Cannot pass `scale` at call time if `create_scale=True`."):
      layer(jnp.ones([2, 3, 5]), scale=jnp.ones([4]))

  @test_utils.transform_and_run
  def test_create_offset_and_offset_provided(self):
    layer = group_norm.GroupNorm(
        groups=5, create_offset=True, create_scale=False)

    with self.assertRaisesRegex(
        ValueError,
        "Cannot pass `offset` at call time if `create_offset=True`."):
      layer(jnp.ones([2, 3, 5]), offset=jnp.ones([4]))

  @test_utils.transform_and_run
  def test_slice_axis(self):
    slice_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    axis_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)

    inputs = np.random.uniform(0, 10, [3, 4, 4, 5]).astype(np.float32)
    scale = np.random.normal(size=(5,), loc=1.0)
    offset = np.random.normal(size=(5,))

    slice_outputs = slice_layer(inputs, scale, offset)
    axis_outputs = axis_layer(inputs, scale, offset)

    self.assertAllClose(slice_outputs, axis_outputs)

  @test_utils.transform_and_run
  def test_rank_changes(self):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)

    inputs = jnp.ones([2, 3, 3, 5])
    scale = constant(0.5, shape=(5,))
    offset = constant(2.0, shape=(5,))

    layer(inputs, scale, offset)

    with self.assertRaisesRegex(
        ValueError,
        "The rank of the inputs cannot change between calls, the original"):
      layer(jnp.ones([2, 3, 3, 4, 5]), scale, offset)

  @parameterized.named_parameters(("Small", (2, 4, 4)), ("Bigger", (2, 3, 8)))
  @test_utils.transform_and_run
  def test_incompatible_groups_and_tensor(self, shape):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)

    inputs = jnp.ones(shape)

    with self.assertRaisesRegex(
        ValueError,
        "The number of channels must be divisible by the number of groups"):
      layer(inputs)

  @test_utils.transform_and_run
  def test5ddata_format_agnostic(self):
    c_last_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    c_first_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False, data_format="NCDHW")

    inputs = np.random.uniform(0, 10, [3, 4, 4, 4, 10]).astype(np.float32)
    scale = np.random.normal(size=(10,), loc=1.0)
    offset = np.random.normal(size=(10,))

    c_last_output = c_last_layer(inputs, scale, offset)
    inputs = jnp.transpose(inputs, [0, 4, 1, 2, 3])
    scale = jnp.reshape(scale, [-1, 1, 1, 1])
    offset = jnp.reshape(offset, [-1, 1, 1, 1])
    c_first_output = c_first_layer(inputs, scale, offset)
    c_first_output = jnp.transpose(c_first_output, [0, 2, 3, 4, 1])

    self.assertAllClose(
        c_last_output, c_first_output, atol=1e-5, rtol=1e-5)

  @test_utils.transform_and_run
  def test3ddata_format_agnostic(self):
    c_last_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    c_first_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False, data_format="NCW")

    inputs = np.random.uniform(0, 10, [3, 4, 10]).astype(np.float32)
    scale = np.random.normal(size=(10,), loc=1.0)
    offset = np.random.normal(size=(10,))

    c_last_output = c_last_layer(inputs, scale, offset)
    inputs = jnp.transpose(inputs, [0, 2, 1])
    scale = jnp.reshape(scale, [-1, 1])
    offset = jnp.reshape(offset, [-1, 1])
    c_first_output = c_first_layer(inputs, scale, offset)
    c_first_output = jnp.transpose(c_first_output, [0, 2, 1])

    self.assertAllClose(
        c_last_output, c_first_output, atol=1e-5, rtol=1e-5)

  def assertAllClose(self, actual, desired, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(actual, desired, atol=atol, rtol=rtol)

if __name__ == "__main__":
  absltest.main()
