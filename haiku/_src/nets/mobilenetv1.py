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

from typing import Optional, Sequence, Text

from haiku._src import basic
from haiku._src import batch_norm
from haiku._src import conv
from haiku._src import depthwise_conv
from haiku._src import module
from haiku._src import reshape

import jax
import jax.numpy as jnp


class MobileNetV1Block(module.Module):
  """Block for MobileNetV1."""

  def __init__(self,
               channels: int,
               stride: int,
               use_bn: bool = True,
               name: Optional[Text] = None):
    super(MobileNetV1Block, self).__init__(name=name)
    self._channels = channels
    self._stride = stride
    self._use_bn = use_bn
    self._with_bias = not use_bn

  def __call__(self, inputs, is_training):
    dwc_layer = depthwise_conv.DepthwiseConv2D(1, 3,
                                               stride=self._stride,
                                               padding=((1, 1), (1, 1)),
                                               with_bias=self._with_bias,
                                               name="depthwise_conv")
    pwc_layer = conv.Conv2D(self._channels,
                            (1, 1),
                            stride=1,
                            padding="VALID",
                            with_bias=self._with_bias,
                            name="pointwise_conv")

    net = inputs
    net = dwc_layer(net)
    if self._use_bn:
      bn = batch_norm.BatchNorm(
          create_scale=True, create_offset=True, decay_rate=0.999)
      net = bn(net, is_training)
    net = jax.nn.relu(net)
    net = pwc_layer(net)
    if self._use_bn:
      bn = batch_norm.BatchNorm(
          create_scale=True, create_offset=True, decay_rate=0.999)
      net = bn(net, is_training)
    net = jax.nn.relu(net)
    return net


class MobileNetV1(module.Module):
  """MobileNetV1 model."""
  # TODO(jordanhoffmann) add width multiplier

  def __init__(self,
               strides: Sequence[int] = (1, 2, 1, 2, 1, 2, 1,
                                         1, 1, 1, 1, 2, 1),
               channels: Sequence[int] = (64, 128, 128, 256, 256, 512, 512,
                                          512, 512, 512, 512, 1024, 1024),
               num_classes: int = 1000,
               use_bn: bool = True,
               name: Optional[Text] = None):
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
    super(MobileNetV1, self).__init__(name=name)
    if len(strides) != len(channels):
      raise ValueError(
          "`strides` and `channels` must have the same length."
          )
    self._strides = strides
    self._channels = channels
    self._use_bn = use_bn
    self._with_bias = not use_bn
    self._num_classes = num_classes

  def __call__(self, inputs, is_training):
    initial_conv = conv.Conv2D(32, (3, 3),
                               stride=2,
                               padding="VALID",
                               with_bias=self._with_bias)
    net = initial_conv(inputs)
    if self._use_bn:
      bn = batch_norm.BatchNorm(
          create_scale=True, create_offset=True, decay_rate=0.999)
      net = bn(net, is_training)
    net = jax.nn.relu(net)
    for i in range(len(self._strides)):
      net = MobileNetV1Block(self._channels[i],
                             self._strides[i],
                             self._use_bn)(net, is_training)
    net = jnp.mean(net, axis=(1, 2))
    net = reshape.Flatten()(net)
    net = basic.Linear(self._num_classes, name="logits")(net)
    return net
