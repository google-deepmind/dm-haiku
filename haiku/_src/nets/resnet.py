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
"""Resnet."""

from typing import Mapping, Optional, Sequence, Text, Union

from haiku._src import basic
from haiku._src import batch_norm
from haiku._src import conv
from haiku._src import module
from haiku._src import pool
import jax
import jax.numpy as jnp


class BottleNeckBlockV1(module.Module):
  """Bottleneck Block for a ResNet implementation."""

  def __init__(self,
               channels: int,
               stride: Union[int, Sequence[int]],
               use_projection: bool,
               bn_config: Mapping[Text, float],
               name: Optional[Text] = None):
    super().__init__(name=name)
    self._use_projection = use_projection

    bn_config = dict(bn_config)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)
    bn_config.setdefault("decay_rate", 0.999)

    if self._use_projection:
      self._proj_conv = conv.Conv2D(output_channels=channels,
                                    kernel_shape=1,
                                    stride=stride,
                                    with_bias=False,
                                    padding="SAME",
                                    name="shortcut_conv")

      self._proj_batchnorm = batch_norm.BatchNorm(name="shortcut_batchnorm",
                                                  **bn_config)

    conv_0 = conv.Conv2D(output_channels=channels // 4,
                         kernel_shape=1,
                         stride=1,
                         with_bias=False,
                         padding="SAME",
                         name="conv_0")
    bn_0 = batch_norm.BatchNorm(name="batchnorm_0", **bn_config)

    conv_1 = conv.Conv2D(output_channels=channels // 4,
                         kernel_shape=3,
                         stride=stride,
                         with_bias=False,
                         padding="SAME",
                         name="conv_1")

    bn_1 = batch_norm.BatchNorm(name="batchnorm_1", **bn_config)

    conv_2 = conv.Conv2D(output_channels=channels,
                         kernel_shape=1,
                         stride=1,
                         with_bias=False,
                         padding="SAME",
                         name="conv_2")

    bn_2 = batch_norm.BatchNorm(name="batchnorm_2", scale_init=jnp.zeros,
                                **bn_config)

    self._layers = ((conv_0, bn_0), (conv_1, bn_1), (conv_2, bn_2))

  def __call__(self, inputs, is_training):
    x = shortcut = inputs

    if self._use_projection:
      shortcut = self._proj_conv(shortcut)
      shortcut = self._proj_batchnorm(shortcut, is_training=is_training)

    for i, (conv_i, bn_i) in enumerate(self._layers):
      x = conv_i(x)
      x = bn_i(x, is_training=is_training)
      if i < 2:  # Don't apply relu on last layer
        x = jax.nn.relu(x)

    return jax.nn.relu(x + shortcut)


class BottleNeckBlockV2(module.Module):
  """Bottleneck Block for a Resnet implementation."""

  def __init__(self,
               channels: int,
               stride: Union[int, Sequence[int]],
               use_projection: bool,
               bn_config: Mapping[Text, float],
               name: Optional[Text] = None):
    super().__init__(name=name)
    self._use_projection = use_projection

    bn_config = dict(bn_config)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)

    if self._use_projection:
      self._proj_conv = conv.Conv2D(output_channels=channels,
                                    kernel_shape=1,
                                    stride=stride,
                                    with_bias=False,
                                    padding="SAME",
                                    name="shortcut_conv")

    conv_0 = conv.Conv2D(output_channels=channels // 4,
                         kernel_shape=1,
                         stride=1,
                         with_bias=False,
                         padding="SAME",
                         name="conv_0")

    bn_0 = batch_norm.BatchNorm(name="batchnorm_0", **bn_config)

    conv_1 = conv.Conv2D(output_channels=channels // 4,
                         kernel_shape=3,
                         stride=stride,
                         with_bias=False,
                         padding="SAME",
                         name="conv_1")

    bn_1 = batch_norm.BatchNorm(name="batchnorm_1", **bn_config)

    conv_2 = conv.Conv2D(output_channels=channels,
                         kernel_shape=1,
                         stride=1,
                         with_bias=False,
                         padding="SAME",
                         name="conv_2")

    # NOTE: Some implementations of ResNet50 v2 suggest initializing gamma/scale
    # here to zeros.
    bn_2 = batch_norm.BatchNorm(name="batchnorm_2", **bn_config)

    self._layers = ((conv_0, bn_0), (conv_1, bn_1), (conv_2, bn_2))

  def __call__(self, inputs, is_training):
    x = shortcut = inputs

    for i, (conv_i, bn_i) in enumerate(self._layers):
      x = bn_i(x, is_training=is_training)
      x = jax.nn.relu(x)
      if i == 0 and self._use_projection:
        shortcut = self._proj_conv(x)
      x = conv_i(x)

    return x + shortcut


class BlockGroup(module.Module):
  """Higher level block for ResNet implementation."""

  def __init__(self,
               channels: int,
               num_blocks: int,
               stride: Union[int, Sequence[int]],
               bn_config: Mapping[Text, float],
               resnet_v2: bool = False,
               name: Optional[Text] = None):
    super().__init__(name=name)

    block_cls = BottleNeckBlockV2 if resnet_v2 else BottleNeckBlockV1

    self._blocks = []
    for i in range(num_blocks):
      self._blocks.append(
          block_cls(channels=channels,
                    stride=(1 if i else stride),
                    use_projection=(i == 0),
                    bn_config=bn_config,
                    name="block_%d" % (i)))

  def __call__(self, inputs, is_training):
    x = inputs
    for block in self._blocks:
      x = block(x, is_training=is_training)
    return x


def check_length(length, value, name):
  if len(value) != length:
    raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(module.Module):
  """ResNet model."""

  def __init__(self,
               blocks_per_group: Sequence[int],
               num_classes: int,
               bn_config: Optional[Mapping[Text, float]] = None,
               resnet_v2: bool = False,
               channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
               name: Optional[Text] = None):
    """Constructs a ResNet model.

    Args:
      blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers. By default the `decay_rate` is
        `0.9` and `eps` is `1e-5`.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        False.
      channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._resnet_v2 = resnet_v2

    bn_config = dict(bn_config or {})
    bn_config.setdefault("decay_rate", 0.9)
    bn_config.setdefault("eps", 1e-5)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)

    # Number of blocks in each group for ResNet.
    check_length(4, blocks_per_group, "blocks_per_group")
    check_length(4, channels_per_group, "channels_per_group")

    self._initial_conv = conv.Conv2D(output_channels=64,
                                     kernel_shape=7,
                                     stride=2,
                                     with_bias=False,
                                     padding="SAME",
                                     name="initial_conv")

    if not self._resnet_v2:
      self._initial_batchnorm = batch_norm.BatchNorm(name="initial_batchnorm",
                                                     **bn_config)

    self._block_groups = []
    strides = (1, 2, 2, 2)
    for i in range(4):
      self._block_groups.append(
          BlockGroup(channels=channels_per_group[i],
                     num_blocks=blocks_per_group[i],
                     stride=strides[i],
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     name="block_group_%d" % (i)))

    if self._resnet_v2:
      self._final_batchnorm = batch_norm.BatchNorm(name="final_batchnorm",
                                                   **bn_config)

    self._logits = basic.Linear(num_classes, w_init=jnp.zeros, name="logits")

  def __call__(self, inputs, is_training):
    x = inputs
    x = self._initial_conv(x)
    if not self._resnet_v2:
      x = self._initial_batchnorm(x, is_training=is_training)
      x = jax.nn.relu(x)

    x = pool.max_pool(x,
                      window_shape=(1, 3, 3, 1),
                      strides=(1, 2, 2, 1),
                      padding="SAME")

    for block_group in self._block_groups:
      x = block_group(x, is_training)

    if self._resnet_v2:
      x = self._final_batchnorm(x, is_training=is_training)
      x = jax.nn.relu(x)
    x = jnp.mean(x, axis=[1, 2])
    return self._logits(x)


class ResNet50(ResNet):
  """ResNet50."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[Text, float]] = None,
               resnet_v2: bool = False,
               name: Optional[Text] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 4, 6, 3),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     name=name)


class ResNet101(ResNet):
  """ResNet101."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[Text, float]] = None,
               resnet_v2: bool = False,
               name: Optional[Text] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 4, 23, 3),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     name=name)


class ResNet152(ResNet):
  """ResNet152."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[Text, float]] = None,
               resnet_v2: bool = False,
               name: Optional[Text] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 8, 36, 3),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     name=name)


class ResNet200(ResNet):
  """ResNet200."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[Text, float]] = None,
               resnet_v2: bool = False,
               name: Optional[Text] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
        to False.
      name: Name of the module.
    """
    super().__init__(blocks_per_group=(3, 24, 36, 3),
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     name=name)
