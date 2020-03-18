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
"""Tests for haiku._src.conv."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import conv
from haiku._src import initializers
from haiku._src import test_utils
from haiku._src import transform
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

  @parameterized.parameters(0, 4)
  def testIncorrectN(self, n):
    init_fn, _ = transform.transform(
        lambda: conv.ConvND(n, output_channels=1, kernel_shape=3))
    with self.assertRaisesRegex(
        ValueError,
        "only support convolution operations for num_spatial_dims=1, 2 or 3"):
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
    expected_output_shape = (2, 3) + (16,)*n
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
    expected_output_shape = (2,) + (16,)*n + (3,)
    self.assertEqual(out.shape, expected_output_shape)
    self.assertEqual(reached[0], n*2)

  @test_utils.transform_and_run
  def test_invalid_input_shape(self):
    n = 1
    input_shape = [2, 4] + [16]*n

    with self.assertRaisesRegex(ValueError, "Input to ConvND needs to have "
                                            "rank 3, but input has shape"):
      data = jnp.zeros(input_shape * 2)
      net = conv.ConvND(n, output_channels=3, kernel_shape=3,
                        data_format="channels_first")
      net(data)

  @test_utils.transform_and_run
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
    out = apply_fn(init_fn(random.PRNGKey(428)))

    self.assertEqual(out.shape, (1, 5, 1))
    out = jnp.squeeze(out, axis=(0, 2))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_allclose(out, expected_out)

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
    out = apply_fn(init_fn(random.PRNGKey(428)))

    self.assertEqual(out.shape, (1, 3, 1))
    out = np.squeeze(out, axis=(0, 2))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_allclose(out, expected_out)


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
    out = apply_fn(init_fn(random.PRNGKey(428)))

    self.assertEqual(out.shape, (1, 5, 5, 1))
    out = np.squeeze(out, axis=(0, 3))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_equal(out, expected_out)

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
    out = apply_fn(init_fn(random.PRNGKey(428)))

    self.assertEqual(out.shape, (1, 3, 3, 1))
    out = np.squeeze(out, axis=(0, 3))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_equal(out, expected_out)


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
    out = apply_fn(init_fn(random.PRNGKey(428)))

    self.assertEqual(out.shape, (1, 5, 5, 5, 1))
    out = np.squeeze(out, axis=(0, 4))
    np.testing.assert_equal(out, expected_out)

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
    out = apply_fn(init_fn(random.PRNGKey(428)))

    self.assertEqual(out.shape, (1, 3, 3, 3, 1))
    out = np.squeeze(out, axis=(0, 4))
    np.testing.assert_equal(out, expected_out)

  @test_utils.transform_and_run
  def test_invalid_input_shape(self):
    with_bias = True

    with self.assertRaisesRegex(ValueError, "Input to ConvND needs to have "
                                            "rank 5, but input has shape"):
      data = jnp.ones([1, 5, 5, 5, 1, 9, 9])
      net = conv.Conv3D(
          output_channels=1,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          **create_constant_initializers(1.0, 1.0, with_bias))
      net(data)


class ConvTransposeTest(parameterized.TestCase):

  @parameterized.parameters(0, 4)
  def testIncorrectN(self, n):
    init_fn, _ = transform.transform(
        lambda: conv.ConvNDTranspose(n, output_channels=1, kernel_shape=3))
    with self.assertRaisesRegex(
        ValueError,
        "only support convolution operations for num_spatial_dims=1, 2 or 3"):
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
    expected_output_shape = (2,) + (24,)*n + (3,)
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
    out = apply_fn(init_fn(random.PRNGKey(428)))
    expected_output_shape = (2, 3) + (16,)*n
    self.assertEqual(out.shape, expected_output_shape)

  @test_utils.transform_and_run
  def test_invalid_input_shape(self):
    n = 1
    with self.assertRaisesRegex(ValueError,
                                "Input to ConvND needs to have rank"):
      input_shape = [2, 4] + [16]*n
      data = jnp.zeros(input_shape*2)
      net = conv.ConvNDTranspose(
          n, output_channels=3, kernel_shape=3, data_format="channels_first")
      return net(data)

  @test_utils.transform_and_run
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
        mask=jnp.zeros([3, 3, 4, 3]))
    out = net(data)
    expected_output_shape = (2, 3, 16, 16)
    self.assertEqual(out.shape, expected_output_shape)


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
    out = apply_fn(init_fn(random.PRNGKey(428)))

    self.assertEqual(out.shape, (1, 3, 1))
    out = np.squeeze(out, axis=(0, 2))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_equal(out, expected_out)

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
    out = apply_fn(init_fn(random.PRNGKey(428)))

    self.assertEqual(out.shape, (1, 5, 1))
    out = np.squeeze(out, axis=(0, 2))

    expected_out = np.asarray(expected_out, dtype=float)
    if with_bias:
      expected_out += 1

    np.testing.assert_equal(out, expected_out)


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
    out = apply_fn(init_fn(random.PRNGKey(428)))
    expected_out = np.array([[4, 6, 4], [6, 9, 6], [4, 6, 4]])
    if with_bias:
      expected_out += 1
    expected_out = np.expand_dims(np.atleast_3d(expected_out), axis=0)
    np.testing.assert_equal(out, expected_out)

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
    out = apply_fn(init_fn(random.PRNGKey(428)))
    expected_out = np.array([[1, 2, 3, 3, 2, 1],
                             [2, 4, 6, 6, 4, 2],
                             [3, 6, 9, 9, 6, 3],
                             [3, 6, 9, 9, 6, 3],
                             [2, 4, 6, 6, 4, 2],
                             [1, 2, 3, 3, 2, 1]])
    if with_bias:
      expected_out += 1
    expected_out = np.expand_dims(np.atleast_3d(expected_out), axis=0)
    np.testing.assert_equal(out, expected_out)


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
    out = apply_fn(init_fn(random.PRNGKey(428)))

    self.assertEqual(out.shape, (1, 3, 3, 3, 1))
    out = np.squeeze(out, axis=(0, 4))
    np.testing.assert_equal(out, expected_out)

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
    out = apply_fn(init_fn(random.PRNGKey(428)))

    self.assertEqual(out.shape, (1, 5, 5, 5, 1))
    out = np.squeeze(out, axis=(0, 4))
    np.testing.assert_equal(out, expected_out)


if __name__ == "__main__":
  absltest.main()
