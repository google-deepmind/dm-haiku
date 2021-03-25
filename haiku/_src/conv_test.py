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
"""Tests for haiku._src.conv."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import conv
from haiku._src import initializers
from haiku._src import test_utils
from haiku._src import transform
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def create_constant_initializers(w, b, with_bias):
  if with_bias:
    return {
        "w_init": initializers.Constant(w),
        "b_init": initializers.Constant(b)
    }
  else:
    return {"w_init": initializers.Constant(w)}


class ConvTest(parameterized.TestCase):

  @parameterized.parameters(0, -2)
  def testIncorrectN(self, n):
    init_fn, _ = transform.transform(
        lambda: conv.ConvND(n, output_channels=1, kernel_shape=3))
    with self.assertRaisesRegex(
        ValueError,
        "convolution operations for `num_spatial_dims` greater than 0"):
      init_fn(None)

  @parameterized.parameters(1, 2, 3)
  def test_connect_conv_same(self, n):
    input_shape = [2] + [16]*n + [4]

    def f():
      data = jnp.zeros(input_shape)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3,
                        padding="SAME")
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2,) + (16,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(1, 2, 3)
  def test_connect_conv_valid(self, n):
    input_shape = [2] + [16]*n + [4]

    def f():
      data = jnp.zeros(input_shape)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3,
                        padding="VALID")
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2,) + (14,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(1, 2, 3)
  def test_strided_conv(self, n):
    input_shape = [2] + [16]*n + [4]

    def f():
      data = jnp.zeros(input_shape)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3, stride=3)
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2,) + (6,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(1, 2, 3)
  def test_diluted_conv(self, n):
    input_shape = [2] + [16]*n + [4]

    def f():
      data = jnp.zeros(input_shape)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3, rate=3)
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2,) + (16,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(1, 2, 3)
  def test_connect_conv_channels_first(self, n):
    input_shape = [2, 4] + [16]*n

    def f():
      data = jnp.zeros(input_shape)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3,
                        data_format="channels_first")
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2, 3) + (16,)*n
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(1, 2, 3)
  def test_unbatched(self, n):
    input_shape = [2] + [16]*n + [4]

    def f():
      data = jnp.zeros(input_shape)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3)
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2,) + (16,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(1, 2, 3)
  def test_connect_conv_padding_function_valid(self, n):
    reached = [0]

    def foo(ks):  # pylint: disable=unused-argument
      reached[0] += 1
      return (0, 0)

    input_shape = [2] + [16]*n + [4]

    def f():
      data = jnp.zeros(input_shape)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3,
                        padding=foo)
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2,) + (14,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)
    self.assertEqual(reached[0], n*2)

  @parameterized.parameters(1, 2, 3)
  def test_connect_conv_padding_function_same(self, n):
    reached = [0]

    def foo(ks):
      reached[0] += 1
      return ((ks-1)//2, ks//2)

    input_shape = [2] + [16]*n + [4]

    def f():
      data = jnp.zeros(input_shape)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3,
                        padding=foo)
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2,) + (16,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)
    self.assertEqual(reached[0], n*2)

  @test_utils.transform_and_run(run_apply=False)
  def test_invalid_input_shape(self):
    n = 1
    input_shape = [2, 4] + [16]*n

    with self.assertRaisesRegex(
        ValueError, r"Input to ConvND needs to have rank in \[2, 3\]"):
      data = jnp.zeros(input_shape * 2)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3,
                        data_format="channels_first")
      net(data)

  @test_utils.transform_and_run(run_apply=False)
  def test_invalid_mask_shape(self):
    n = 1
    input_shape = [2, 4] + [16]*n

    with self.assertRaisesRegex(ValueError, "Mask needs to have the same "
                                            "shape as weights. Shapes are:"):
      data = jnp.zeros(input_shape)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3,
                        data_format="channels_first", mask=jnp.ones([1, 5, 1]))
      net(data)

  @test_utils.transform_and_run
  def test_valid_mask_shape(self):
    n = 2
    input_shape = [2, 4] + [16]*n
    data = jnp.zeros(input_shape)
    net = conv.ConvND(n, output_channels=3, kernel_shape=3,
                      data_format="channels_first",
                      mask=jnp.ones([3, 3, 4, 3]))
    out = net(data)
    expected_output_shape = (2, 3) + (16,)*n
    self.assertEqual(out.shape, expected_output_shape)

  @test_utils.transform_and_run
  def test_group_conv(self):
    batch_size = 3
    seqlen = 12
    hidden = 32
    hidden_out = 64
    feature_group_count = 2
    inputs = np.zeros((batch_size, seqlen, hidden))
    inputs[0, 0, :2] = 1.0
    inputs[0, 5, 24] = 1.0
    inputs[0, 7, 28:32] = 1.0

    data = jnp.asarray(inputs)
    net = conv.Conv1D(
        output_channels=hidden_out,
        kernel_shape=1,
        with_bias=False,
        feature_group_count=feature_group_count)
    out = net(data)

    expected_output_shape = (batch_size, seqlen, hidden_out)
    self.assertEqual(out.shape, expected_output_shape)

    # Make sure changing first half in time step 0 did affect exactly
    # all first half elements in the output:
    self.assertTrue((out[0, 0, :hidden_out//feature_group_count] != 0).all())
    self.assertTrue((out[0, 0, hidden_out//feature_group_count:-1] == 0).all())

    # Make sure time step 5 and 7 it is the second half exactly.
    self.assertTrue((out[0, 5, :hidden_out//feature_group_count] == 0).all())
    self.assertTrue((out[0, 7, hidden_out//feature_group_count:-1] != 0).all())


class Conv1DTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_computation_padding_same(self, with_bias):
    expected_out = [2, 3, 3, 3, 2]
    def f():
      data = jnp.ones([1, 5, 1])
      net = conv.Conv1D(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="SAME",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)

    self.assertEqual(out.shape, (1, 5, 1))
    out = jnp.squeeze(out, axis=(0, 2))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_allclose(out, expected_out, rtol=1e-5)

  @parameterized.parameters(True, False)
  def test_computation_padding_valid(self, with_bias):
    expected_out = [3, 3, 3]

    def f():
      data = jnp.ones([1, 5, 1])
      net = conv.Conv1D(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)

    self.assertEqual(out.shape, (1, 3, 1))
    out = np.squeeze(out, axis=(0, 2))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_allclose(out, expected_out, rtol=1e-5)


class Conv2DTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_computation_padding_same(self, with_bias):
    expected_out = [[4, 6, 6, 6, 4], [6, 9, 9, 9, 6], [6, 9, 9, 9, 6],
                    [6, 9, 9, 9, 6], [4, 6, 6, 6, 4]]
    def f():
      data = jnp.ones([1, 5, 5, 1])
      net = conv.Conv2D(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="SAME",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)

    self.assertEqual(out.shape, (1, 5, 5, 1))
    out = np.squeeze(out, axis=(0, 3))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_allclose(out, expected_out, rtol=1e-5)

  @parameterized.parameters(True, False)
  def test_computation_padding_valid(self, with_bias):
    expected_out = [[9, 9, 9], [9, 9, 9], [9, 9, 9]]

    def f():
      data = jnp.ones([1, 5, 5, 1])
      net = conv.Conv2D(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)

    self.assertEqual(out.shape, (1, 3, 3, 1))
    out = np.squeeze(out, axis=(0, 3))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_allclose(out, expected_out, rtol=1e-5)


class Conv3DTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_computation_padding_same(self, with_bias):
    expected_out = np.asarray([
        9, 13, 13, 13, 9, 13, 19, 19, 19, 13, 13, 19, 19, 19, 13, 13, 19, 19,
        19, 13, 9, 13, 13, 13, 9, 13, 19, 19, 19, 13, 19, 28, 28, 28, 19, 19,
        28, 28, 28, 19, 19, 28, 28, 28, 19, 13, 19, 19, 19, 13, 13, 19, 19, 19,
        13, 19, 28, 28, 28, 19, 19, 28, 28, 28, 19, 19, 28, 28, 28, 19, 13, 19,
        19, 19, 13, 13, 19, 19, 19, 13, 19, 28, 28, 28, 19, 19, 28, 28, 28, 19,
        19, 28, 28, 28, 19, 13, 19, 19, 19, 13, 9, 13, 13, 13, 9, 13, 19, 19,
        19, 13, 13, 19, 19, 19, 13, 13, 19, 19, 19, 13, 9, 13, 13, 13, 9
    ],
                              dtype=float).reshape((5, 5, 5))
    if not with_bias:
      expected_out -= 1

    def f():
      data = jnp.ones([1, 5, 5, 5, 1])
      net = conv.Conv3D(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="SAME",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)

    self.assertEqual(out.shape, (1, 5, 5, 5, 1))
    out = np.squeeze(out, axis=(0, 4))
    np.testing.assert_allclose(out, expected_out, rtol=1e-5)

  @parameterized.parameters(True, False)
  def test_computation_padding_valid(self, with_bias):
    expected_out = np.asarray([
        28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
        28, 28, 28, 28, 28, 28, 28, 28, 28
    ],
                              dtype=float).reshape((3, 3, 3))
    if not with_bias:
      expected_out -= 1

    def f():
      data = jnp.ones([1, 5, 5, 5, 1])
      net = conv.Conv3D(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)

    self.assertEqual(out.shape, (1, 3, 3, 3, 1))
    out = np.squeeze(out, axis=(0, 4))
    np.testing.assert_allclose(out, expected_out, rtol=1e-5)

  @test_utils.transform_and_run(run_apply=False)
  def test_invalid_input_shape(self):
    with_bias = True

    with self.assertRaisesRegex(
        ValueError,
        r"Input to ConvND needs to have rank in \[4, 5\], but input has shape"):
      data = jnp.ones([1, 5, 5, 5, 1, 9, 9])
      net = conv.Conv3D(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      net(data)


def default_output_shape(input_shape, kernel, stride, padding):
  if padding == "SAME":
    return input_shape * stride
  elif padding == "VALID":
    return (input_shape - 1) * stride + kernel


class ConvTransposeTest(parameterized.TestCase):

  @parameterized.parameters(0, -2)
  def testIncorrectN(self, n):
    init_fn, _ = transform.transform(
        lambda: conv.ConvNDTranspose(n, output_channels=1, kernel_shape=3))
    with self.assertRaisesRegex(
        ValueError,
        "convolution operations for `num_spatial_dims` greater than 0"):
      init_fn(None)

  @parameterized.parameters(1, 2, 3)
  def test_connect_conv_transpose_same(self, n):
    def f():
      input_shape = [2] + [16]*n + [4]
      data = jnp.zeros(input_shape)
      net = conv.ConvNDTranspose(
          n, output_channels=3, kernel_shape=3, padding="SAME")
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2,) + (16,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(1, 2, 3)
  def test_connect_conv_transpose_valid(self, n):
    def f():
      input_shape = [2] + [16]*n + [4]
      data = jnp.zeros(input_shape)
      net = conv.ConvNDTranspose(
          n, output_channels=3, kernel_shape=3, padding="VALID")
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2,) + (18,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(1, 2, 3)
  def test_connect_conv_transpose_strided(self, n):
    def f():
      input_shape = [2] + [8]*n + [4]
      data = jnp.zeros(input_shape)
      net = conv.ConvNDTranspose(
          n, output_channels=3, kernel_shape=3, stride=3)
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2,) + (24,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(1, 2, 3)
  def test_unbatched(self, n):
    def f():
      input_shape = [8]*n + [4]
      data = jnp.zeros(input_shape)
      net = conv.ConvNDTranspose(
          n, output_channels=3, kernel_shape=3, stride=3)
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (24,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(1, 2, 3)
  def test_connect_conv_transpose_channels_first(self, n):
    def f():
      input_shape = [2, 4] + [16]*n
      data = jnp.zeros(input_shape)
      net = conv.ConvNDTranspose(
          n, output_channels=3, kernel_shape=3, data_format="channels_first")
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_output_shape = (2, 3) + (16,)*n
    self.assertEqual(out.shape, expected_output_shape)

  @test_utils.transform_and_run(run_apply=False)
  def test_invalid_input_shape(self):
    n = 1
    with self.assertRaisesRegex(ValueError,
                                "Input to ConvNDTranspose needs to have rank"):
      input_shape = [2, 4] + [16]*n
      data = jnp.zeros(input_shape*2)
      net = conv.ConvNDTranspose(
          n, output_channels=3, kernel_shape=3, data_format="channels_first")
      return net(data)

  @test_utils.transform_and_run(run_apply=False)
  def test_invalid_input_mask(self):
    n = 2
    with self.assertRaisesRegex(ValueError, "Mask needs to have the same "
                                            "shape as weights. Shapes are:"):
      input_shape = [2, 4] + [16]*n
      data = jnp.zeros(input_shape)
      net = conv.ConvNDTranspose(
          n, output_channels=3, kernel_shape=3,
          data_format="channels_first",
          mask=jnp.zeros([1, 2, 3]))
      net(data)

  @test_utils.transform_and_run
  def test_valid_input_mask(self):
    n = 2
    input_shape = [2, 4] + [16]*n
    data = jnp.zeros(input_shape)
    net = conv.ConvNDTranspose(
        n, output_channels=3, kernel_shape=3,
        data_format="channels_first",
        mask=jnp.zeros([3, 3, 3, 4]))
    out = net(data)
    expected_output_shape = (2, 3, 16, 16)
    self.assertEqual(out.shape, expected_output_shape)

  @parameterized.parameters(
      (1, (3,), 128, 5, "NWC"),
      (2, (4, 4), 64, 3, "NHWC"),
      (3, (4, 4, 4), 64, 3, "NDHWC"))
  @test_utils.transform_and_run
  def test_initializer_variance(self, num_spatial_dims, kernel_shape,
                                in_channels, output_channels, data_format):
    c = conv.ConvNDTranspose(
        num_spatial_dims=num_spatial_dims,
        kernel_shape=kernel_shape,
        output_channels=output_channels,
        data_format=data_format)

    inputs = jnp.ones([16] + ([32] * num_spatial_dims) + [in_channels])
    c(inputs)

    w = c.params_dict()["conv_nd_transpose/w"]
    actual_std = w.std()
    expected_std = 1 / (np.sqrt(np.prod(kernel_shape + (in_channels,))))

    # This ratio of the error compared to the expected std might be somewhere
    # around 0.15 normally. We check it is not > 0.5, as that would indicate
    # something seriously wrong (ie the previous buggy initialization).
    rel_diff = np.abs(actual_std - expected_std) / expected_std
    self.assertLess(rel_diff, 0.5)

  @parameterized.parameters(
      (10, 20, 5, 2, "SAME", (3, 2)),
      (11, 77, 4, 7, "SAME", (3, 6)),  # Tests max(0, padding_needed) branch.
      (10, 23, 5, 2, "VALID", (4, 4)),
  )
  @test_utils.transform_and_run
  def test_compute_adjusted_padding(self, input_size, output_size, kernel,
                                    stride, padding, expected_adjusted_padding):
    self.assertEqual(
        conv.compute_adjusted_padding(
            input_size=input_size,
            output_size=output_size,
            kernel_size=kernel,
            stride=stride,
            padding=padding), expected_adjusted_padding)

  @parameterized.parameters(
      ([7, 9], None, 5, 3, "SAME", "channels_first"),
      ([7, 9, 16], None, 5, 2, "VALID", "channels_first"),
      ([9, 13], None, 5, 4, "VALID", "channels_last"),
      ([7, 9, 13], None, 5, 3, "VALID", "channels_last"),
      # Default is: 21, 27, 48
      ([7, 9, 16], [19, 25, 48], 5, 3, "SAME", "channels_first"),
      # Default is:  23, 41, 50
      ([7, 13, 16], [25, 42, 50], 5, 3, "VALID", "channels_first"),
      # Default is:  45, 65, 80
      ([9, 13, 16], [43, 64, 80], 6, 5, "SAME", "channels_last"),
      # Default is: 36, 46, 66
      ([7, 9, 13], [38, 48, 67], 6, 5, "VALID", "channels_last"),
  )
  @test_utils.transform_and_run
  def test_output_sizes(self, input_shape, output_shape, kernel, stride,
                        padding, data_format):
    batch_dim = 2
    num_channels = 3
    if data_format == "channels_first":
      data = jnp.zeros([batch_dim, num_channels] + input_shape)
    if data_format == "channels_last":
      data = jnp.zeros([batch_dim] + input_shape + [num_channels])

    net = conv.ConvNDTranspose(
        num_spatial_dims=len(input_shape),
        output_channels=3,
        kernel_shape=kernel,
        output_shape=output_shape,
        stride=stride,
        padding=padding,
        data_format=data_format)
    out = net(data)

    if output_shape is None:
      output_shape = [
          default_output_shape(in_shape, kernel, stride, padding)
          for in_shape in input_shape
      ]
    if data_format == "channels_first":
      expected_shape = tuple([batch_dim, num_channels] + output_shape)
    if data_format == "channels_last":
      expected_shape = tuple([batch_dim] + output_shape + [num_channels])
    self.assertEqual(out.shape, expected_shape)


class Conv1DTransposeTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_computation_padding_same(self, with_bias):
    expected_out = [2, 3, 2]
    def f():
      data = jnp.ones([1, 3, 1])
      net = conv.Conv1DTranspose(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="SAME",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)

    self.assertEqual(out.shape, (1, 3, 1))
    out = np.squeeze(out, axis=(0, 2))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_allclose(out, expected_out, rtol=1e-5)

  @parameterized.parameters(True, False)
  def test_computation_padding_valid(self, with_bias):
    expected_out = [1, 2, 3, 2, 1]

    def f():
      data = jnp.ones([1, 3, 1])
      net = conv.Conv1DTranspose(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)

    self.assertEqual(out.shape, (1, 5, 1))
    out = np.squeeze(out, axis=(0, 2))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_allclose(out, expected_out, rtol=1e-5)


class Conv2TransposeTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_computation_padding_same(self, with_bias):
    def f():
      data = np.ones([1, 3, 3, 1])
      net = conv.Conv2DTranspose(
          output_channels=1,
          kernel_shape=3,
          padding="SAME",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_out = np.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]])
    if with_bias:
      expected_out += 1
    expected_out = np.expand_dims(np.atleast_3d(expected_out), axis=0)
    np.testing.assert_allclose(out, expected_out, rtol=1e-5)

  @parameterized.parameters(True, False)
  def test_computation_padding_valid(self, with_bias):
    """Example taken from Figure 5 of https://link.medium.com/suSvMCsDv1 ."""
    def f():
      data = np.ones([1, 4, 4, 1])
      net = conv.Conv2DTranspose(
          output_channels=1,
          kernel_shape=3,
          padding="VALID",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)
    expected_out = np.array([[1, 2, 3, 3, 2, 1],
                             [2, 4, 6, 6, 4, 2],
                             [3, 6, 9, 9, 6, 3],
                             [3, 6, 9, 9, 6, 3],
                             [2, 4, 6, 6, 4, 2],
                             [1, 2, 3, 3, 2, 1]])
    if with_bias:
      expected_out += 1
    expected_out = np.expand_dims(np.atleast_3d(expected_out), axis=0)
    np.testing.assert_allclose(out, expected_out, rtol=1e-5)


class Conv3DTransposeTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_computation_padding_same(self, with_bias):
    expected_out = np.asarray([
        8, 12, 8, 12, 18, 12, 8, 12, 8, 12, 18, 12, 18, 27, 18, 12, 18, 12, 8,
        12, 8, 12, 18, 12, 8, 12, 8
    ]).reshape((3, 3, 3))
    if with_bias:
      expected_out += 1

    def f():
      data = jnp.ones([1, 3, 3, 3, 1])
      net = conv.Conv3DTranspose(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="SAME",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)

    self.assertEqual(out.shape, (1, 3, 3, 3, 1))
    out = np.squeeze(out, axis=(0, 4))
    np.testing.assert_allclose(out, expected_out, rtol=1e-5)

  @parameterized.parameters(True, False)
  def test_computation_padding_valid(self, with_bias):
    expected_out = np.asarray([
        1, 2, 3, 2, 1, 2, 4, 6, 4, 2, 3, 6, 9, 6, 3, 2, 4, 6, 4, 2, 1, 2, 3, 2,
        1, 2, 4, 6, 4, 2, 4, 8, 12, 8, 4, 6, 12, 18, 12, 6, 4, 8, 12, 8, 4, 2,
        4, 6, 4, 2, 3, 6, 9, 6, 3, 6, 12, 18, 12, 6, 9, 18, 27, 18, 9, 6, 12,
        18, 12, 6, 3, 6, 9, 6, 3, 2, 4, 6, 4, 2, 4, 8, 12, 8, 4, 6, 12, 18, 12,
        6, 4, 8, 12, 8, 4, 2, 4, 6, 4, 2, 1, 2, 3, 2, 1, 2, 4, 6, 4, 2, 3, 6, 9,
        6, 3, 2, 4, 6, 4, 2, 1, 2, 3, 2, 1.
    ]).reshape((5, 5, 5))
    if with_bias:
      expected_out += 1

    def f():
      data = jnp.ones([1, 3, 3, 3, 1])
      net = conv.Conv3DTranspose(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)), None)

    self.assertEqual(out.shape, (1, 5, 5, 5, 1))
    out = np.squeeze(out, axis=(0, 4))
    np.testing.assert_allclose(out, expected_out, rtol=1e-5)

PRECISIONS = (None, jax.lax.Precision.DEFAULT, jax.lax.Precision.HIGH,
              jax.lax.Precision.HIGHEST)
NAMED_PRECISIONS = ((str(p), p) for p in PRECISIONS)


class PrecisionTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(
      NAMED_PRECISIONS,
      (("ConvND", conv.ConvND), ("ConvNDTranspose", conv.ConvNDTranspose)))
  def test_precision(self, precision, cls):

    def f(x):
      net = cls(2, output_channels=3, kernel_shape=3, padding="VALID")
      return net(x, precision=precision)

    f = transform.transform(f)
    rng = jax.random.PRNGKey(42)
    x = jnp.zeros([2, 16, 16, 4])
    params = f.init(rng, x)
    c = jax.xla_computation(lambda x: f.apply(params, None, x))(x)
    hlo = c.as_hlo_text()
    op_line = next(l for l in hlo.split("\n") if "convolution(" in l)
    if precision is not None and precision != jax.lax.Precision.DEFAULT:
      name = str(precision).lower()
      self.assertRegex(op_line, f"operand_precision={{{name},{name}}}")
    else:
      self.assertNotIn("operand_precision", op_line)

if __name__ == "__main__":
  absltest.main()
