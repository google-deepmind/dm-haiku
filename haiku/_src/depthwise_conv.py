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
"""Depthwise Convolutional Haiku module."""

import types
from typing import Optional, Sequence, Union, Tuple

from haiku._src import base
from haiku._src import initializers
from haiku._src import module
from haiku._src import utils

from jax import lax
import jax.numpy as jnp
import numpy as np

# If you are forking replace this block with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.initializers = initializers
hk.get_parameter = base.get_parameter
hk.Module = module.Module
del base, module, initializers

DIMENSION_NUMBERS = {
    1: ("NWC", "WIO", "NWC"),
    2: ("NHWC", "HWIO", "NHWC"),
    3: ("NDHWC", "DHWIO", "NDHWC")
}

DIMENSION_NUMBERS_NCSPATIAL = {
    1: ("NCH", "HIO", "NCH"),
    2: ("NCHW", "HWIO", "NCHW"),
    3: ("NCDHW", "DHWIO", "NCDHW")
}


class DepthwiseConv2D(hk.Module):
  """2-D Depthwise Convolution Module."""
  # TODO(tycai): Generalize to ConvND.

  def __init__(
      self,
      channel_multiplier: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "NHWC",
      name: Optional[str] = None,
  ):
    """Construct a 2D Depthwise Convolution.

    Args:
      channel_multiplier: Multiplicity of output channels. To keep the number of
        output channels the same as the number of input channels, set 1.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length ``num_spatial_dims``.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length ``num_spatial_dims``. Defaults to 1.
      padding: Optional padding algorithm. Either ``VALID``, ``SAME`` or a
        sequence of ``before, after`` pairs. Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input.  Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default, ``channels_last``.
      name: The name of the module.
    """
    super().__init__(name=name)
    self.kernel_shape = utils.replicate(kernel_shape, 2, "kernel_shape")
    self.lhs_dilation = (1,) * len(self.kernel_shape)
    self.rhs_dilation = (1,) * len(self.kernel_shape)
    self.channel_multiplier = channel_multiplier
    self.padding = padding
    self.stride = utils.replicate(stride, 2, "strides")
    self.data_format = data_format
    self.channel_index = utils.get_channel_index(data_format)
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros
    self.num_spatial_dims = 2
    if self.channel_index == -1:
      self.dn = DIMENSION_NUMBERS[self.num_spatial_dims]
    else:
      self.dn = DIMENSION_NUMBERS_NCSPATIAL[self.num_spatial_dims]

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    channel_index = utils.get_channel_index(self.data_format)
    w_shape = self.kernel_shape + (1, self.channel_multiplier *
                                   inputs.shape[channel_index])

    w_init = self.w_init
    if w_init is None:
      fan_in_shape = np.prod(w_shape[:-1])
      stddev = 1. / np.sqrt(fan_in_shape)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

    out = lax.conv_general_dilated(
        inputs,
        w,
        window_strides=self.stride,
        padding=self.padding,
        lhs_dilation=self.lhs_dilation,
        rhs_dilation=self.rhs_dilation,
        dimension_numbers=self.dn,
        feature_group_count=inputs.shape[channel_index])

    if self.with_bias:
      if channel_index == -1:
        b_shape = (self.channel_multiplier * inputs.shape[channel_index],)
      else:
        b_shape = (self.channel_multiplier * inputs.shape[channel_index], 1, 1)
      b = hk.get_parameter("b", b_shape, inputs.dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    return out


class SeparableDepthwiseConv2D(hk.Module):
  """Separable 2-D Depthwise Convolution Module."""

  def __init__(
      self,
      channel_multiplier: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "NHWC",
      name: Optional[str] = None,
  ):
    """Construct a Separable 2D Depthwise Convolution module.

    Args:
      channel_multiplier: Multiplicity of output channels. To keep the number of
        output channels the same as the number of input channels, set 1.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length ``num_spatial_dims``.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length ``num_spatial_dims``. Defaults to 1.
      padding: Optional padding algorithm. Either ``VALID``, ``SAME`` or a
        sequence of ``before, after`` pairs. Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input.  Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default, ``channels_last``.
      name: The name of the module.
    """
    super().__init__(name=name)
    kernel_shape = utils.replicate(kernel_shape, 2, "kernel_shape")
    self._conv1 = DepthwiseConv2D(
        channel_multiplier=channel_multiplier,
        kernel_shape=[kernel_shape[0], 1],
        stride=stride,
        padding=padding,
        with_bias=False,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format)

    self._conv2 = DepthwiseConv2D(
        channel_multiplier=1,
        kernel_shape=[1, kernel_shape[1]],
        stride=1,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format)

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    return self._conv2(self._conv1(inputs))

