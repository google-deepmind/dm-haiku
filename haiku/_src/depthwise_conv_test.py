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
"""Tests for haiku._src.depthwise_conv."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import depthwise_conv
from haiku._src import initializers
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


class DepthwiseConv2DTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_convolution(self, with_bias):
    def f():
      data = np.ones([1, 10, 10, 3])
      data[0, :, :, 1] += 1
      data[0, :, :, 2] += 2
      data = jnp.array(data)
      net = depthwise_conv.DepthwiseConv2D(
          channel_multiplier=1,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          data_format="channels_last",
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)))
    self.assertEqual(out.shape, (1, 8, 8, 3))
    self.assertLen(np.unique(out[0, :, :, 0]), 1)
    self.assertLen(np.unique(out[0, :, :, 1]), 1)
    self.assertLen(np.unique(out[0, :, :, 2]), 1)
    if with_bias:
      self.assertEqual(np.unique(out[0, :, :, 0])[0], 1*3.0*3.0 +1)
      self.assertEqual(np.unique(out[0, :, :, 1])[0], 2*3.0*3.0 +1)
      self.assertEqual(np.unique(out[0, :, :, 2])[0], 3*3.0*3.0 +1)
    else:
      self.assertEqual(np.unique(out[0, :, :, 0])[0], 1*3.0*3.0)
      self.assertEqual(np.unique(out[0, :, :, 1])[0], 2*3.0*3.0)
      self.assertEqual(np.unique(out[0, :, :, 2])[0], 3*3.0*3.0)

  @parameterized.parameters(True, False)
  def test_padding(self, with_bias):
    def f():
      data = np.ones([1, 10, 10, 3])
      data[0, :, :, 1] += 1
      data[0, :, :, 2] += 2
      data = jnp.array(data)
      net = depthwise_conv.DepthwiseConv2D(
          channel_multiplier=1,
          kernel_shape=3,
          stride=1,
          padding="SAME",
          with_bias=with_bias,
          data_format="channels_last",
          **create_constant_initializers(1.0, 0.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)))
    self.assertEqual(out.shape, (1, 10, 10, 3))

  @parameterized.parameters(True, False)
  def test_channel_multiplier(self, with_bias):
    def f():
      data = np.ones([1, 10, 10, 3])
      data[0, :, :, 1] += 1
      data[0, :, :, 2] += 2
      data = jnp.array(data)
      net = depthwise_conv.DepthwiseConv2D(
          channel_multiplier=3,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          data_format="channels_last",
          **create_constant_initializers(1.0, 0.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)))
    self.assertEqual(out.shape, (1, 8, 8, 9))

  @parameterized.parameters(True, False)
  def test_channels_first(self, with_bias):
    def f():
      data = np.ones([1, 3, 10, 10])
      data[0, 1, :, :] += 1
      data[0, 2, :, :] += 2
      data = jnp.array(data)
      net = depthwise_conv.DepthwiseConv2D(
          channel_multiplier=1,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          data_format="channels_first",
          **create_constant_initializers(1.0, 1.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)))
    self.assertEqual(out.shape, (1, 3, 8, 8))
    if with_bias:
      self.assertEqual(np.unique(out[0, 0, :, :])[0], 1*3.0*3.0+1)
      self.assertEqual(np.unique(out[0, 1, :, :])[0], 2*3.0*3.0+1)
      self.assertEqual(np.unique(out[0, 2, :, :])[0], 3*3.0*3.0+1)
    else:
      self.assertEqual(np.unique(out[0, 0, :, :])[0], 1*3.0*3.0)
      self.assertEqual(np.unique(out[0, 1, :, :])[0], 2*3.0*3.0)
      self.assertEqual(np.unique(out[0, 2, :, :])[0], 3*3.0*3.0)

  @parameterized.parameters(True, False)
  def test_channels_first_mult(self, with_bias):
    def f():
      data = np.ones([1, 3, 10, 10])
      data[0, 1, :, :] += 1
      data[0, 2, :, :] += 2
      data = jnp.array(data)
      net = depthwise_conv.DepthwiseConv2D(
          channel_multiplier=9,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          with_bias=with_bias,
          data_format="channels_first",
          **create_constant_initializers(1.0, 0.0, with_bias))
      return net(data)

    init_fn, apply_fn = transform.transform(f)
    out = apply_fn(init_fn(random.PRNGKey(428)))
    self.assertEqual(out.shape, (1, 27, 8, 8))

if __name__ == "__main__":
  absltest.main()
