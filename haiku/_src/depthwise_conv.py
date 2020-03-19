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
"""Depthwise Convolutional Haiku module."""

from haiku._src import base
from haiku._src import initializers
from haiku._src import module
from haiku._src import utils

from jax import lax
import jax.numpy as jnp
import numpy as np

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


class DepthwiseConv2D(module.Module):
  """2-D Depthwise Convolution Module."""
  # TODO(tycai): Generalize to ConvND.

  def __init__(self,
               channel_multiplier,
               kernel_shape,
               stride=1,
               padding="SAME",
               with_bias=True,
               w_init=None,
               b_init=None,
               data_format="NHWC",
               name=None):
    """Construct a 2D Depthwise Convolution.

    Args:
      channel_multiplier: Multiplicity of output channels. To keep the number of
        output channels the same as the number of input channels, set 1.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length `num_spatial_dims`.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length `num_spatial_dims`. Defaults to 1.
      padding: Optional padding algorithm. Either "VALID" or "SAME" or
        a callable or sequence of callables of size `num_spatial_dims`. Any
        callables must take a single integer argument equal to the effective
        kernel size and return a list of two integers representing the padding
        before and after. See haiku.pad.* for more details and example
        functions. Defaults to "SAME". See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input.  Can be either
        `channels_first`, `channels_last`, `N...C` or `NC...`. By default,
        `channels_last`.
      name: The name of the module.
    """
    super(DepthwiseConv2D, self).__init__(name=name)
    self._kernel_shape = utils.replicate(kernel_shape, 2, "kernel_shape")
    self._lhs_dilation = (1,) * len(self._kernel_shape)
    self._rhs_dilation = (1,) * len(self._kernel_shape)
    self._channel_multiplier = channel_multiplier
    self._padding = padding
    self._stride = utils.replicate(stride, 2, "strides")
    self._data_format = data_format
    self._channel_index = utils.get_channel_index(data_format)
    self._with_bias = with_bias
    self._w_init = w_init
    self._b_init = b_init or jnp.zeros
    self._num_spatial_dims = 2

  def __call__(self, inputs):

    channel_index = utils.get_channel_index(self._data_format)
    weight_shape = self._kernel_shape + (1, self._channel_multiplier *
                                         inputs.shape[channel_index])
    fan_in_shape = np.prod(weight_shape[:-1])
    stddev = 1. / np.sqrt(fan_in_shape)
    w_init = self._w_init or initializers.TruncatedNormal(stddev=stddev)
    w = base.get_parameter("w", weight_shape, inputs.dtype, init=w_init)
    if self._channel_index == -1:
      dn = DIMENSION_NUMBERS[self._num_spatial_dims]
    else:
      dn = DIMENSION_NUMBERS_NCSPATIAL[self._num_spatial_dims]
    result = lax.conv_general_dilated(
        inputs,
        w,
        self._stride,
        self._padding,
        self._lhs_dilation,
        self._rhs_dilation,
        dn,
        feature_group_count=inputs.shape[channel_index])
    if self._with_bias:
      if channel_index == -1:
        bias_shape = (self._channel_multiplier * inputs.shape[channel_index],)
      else:
        bias_shape = (self._channel_multiplier *
                      inputs.shape[channel_index], 1, 1)
      b = base.get_parameter("b", bias_shape, init=self._b_init)
      result = result + jnp.broadcast_to(b, result.shape)
    return result
