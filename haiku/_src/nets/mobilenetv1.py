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
"""MobileNet V1, from https://arxiv.org/abs/1704.04861.

Achieves ~71% top-1 performance on ImageNet.
Depending on the input size, may want to adjust strides from their default
configuration.
With a 32x32 input, last block output should be (N, 1, 1, 1024), before
average pooling.
With 224x224 input, will be (N, 7, 7, 1024).
The average pooling is currently done via a mean, and returns (N, 1, 1, 1024).
If something different is desired, replace with AvgPool.
"""

import types
from typing import Optional, Sequence

from haiku._src import basic
from haiku._src import batch_norm
from haiku._src import conv
from haiku._src import depthwise_conv
from haiku._src import module
from haiku._src import reshape

import jax
import jax.numpy as jnp

# If forking replace this block with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.Module = module.Module
hk.BatchNorm = batch_norm.BatchNorm
hk.Conv2D = conv.Conv2D
hk.DepthwiseConv2D = depthwise_conv.DepthwiseConv2D
hk.Flatten = reshape.Flatten
hk.Linear = basic.Linear
del basic, batch_norm, conv, depthwise_conv, module, reshape


class MobileNetV1Block(hk.Module):
  """Block for MobileNetV1."""

  def __init__(
      self,
      channels: int,
      stride: int,
      use_bn: bool = True,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.channels = channels
    self.stride = stride
    self.use_bn = use_bn
    self.with_bias = not use_bn

  def __call__(self, inputs: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    depthwise = hk.DepthwiseConv2D(
        channel_multiplier=1,
        kernel_shape=3,
        stride=self.stride,
        padding=((1, 1), (1, 1)),
        with_bias=self.with_bias,
        name="depthwise_conv")

    pointwise = hk.Conv2D(
        output_channels=self.channels,
        kernel_shape=(1, 1),
        stride=1,
        padding="VALID",
        with_bias=self.with_bias,
        name="pointwise_conv")

    out = depthwise(inputs)
    if self.use_bn:
      bn1 = hk.BatchNorm(create_scale=True, create_offset=True,
                         decay_rate=0.999)
      out = bn1(out, is_training)
    out = jax.nn.relu(out)
    out = pointwise(out)
    if self.use_bn:
      bn2 = hk.BatchNorm(create_scale=True, create_offset=True,
                         decay_rate=0.999)
      out = bn2(out, is_training)
    out = jax.nn.relu(out)
    return out


class MobileNetV1(hk.Module):
  """MobileNetV1 model."""
  # TODO(jordanhoffmann) add width multiplier

  def __init__(
      self,
      strides: Sequence[int] = (1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1),
      channels: Sequence[int] = (64, 128, 128, 256, 256, 512, 512,
                                 512, 512, 512, 512, 1024, 1024),
      num_classes: int = 1000,
      use_bn: bool = True,
      name: Optional[str] = None,
  ):
    """Constructs a MobileNetV1 model.

    Args:
      strides: The stride to use the in depthwise convolution in each mobilenet
               block.
      channels: Number of output channels from the pointwise convolution to use
                in each block.
      num_classes: Number of classes.
      use_bn: Whether or not to use batch normalization. Defaults to True. When
              true, biases are not used. When false, biases are used.
      name: Name of the module.
    """
    super().__init__(name=name)
    if len(strides) != len(channels):
      raise ValueError("`strides` and `channels` must have the same length.")

    self.strides = strides
    self.channels = channels
    self.use_bn = use_bn
    self.with_bias = not use_bn
    self.num_classes = num_classes

  def __call__(self, inputs: jnp.ndarray, is_training: bool) -> jnp.ndarray:
    initial_conv = hk.Conv2D(
        output_channels=32,
        kernel_shape=(3, 3),
        stride=2,
        padding="VALID",
        with_bias=self.with_bias)

    out = initial_conv(inputs)
    if self.use_bn:
      bn = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.999)
      out = bn(out, is_training)
    out = jax.nn.relu(out)
    for i in range(len(self.strides)):
      block = MobileNetV1Block(self.channels[i],
                               self.strides[i],
                               self.use_bn)
      out = block(out, is_training)
    out = jnp.mean(out, axis=(1, 2))
    out = hk.Flatten()(out)
    out = hk.Linear(self.num_classes, name="logits")(out)
    return out
