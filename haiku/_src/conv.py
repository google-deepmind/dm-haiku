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
"""Convolutional Haiku modules."""

from haiku._src import base
from haiku._src import initializers
from haiku._src import module
from haiku._src import pad
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


class ConvND(module.Module):
  """A general N-dimensional convolutional module."""

  def __init__(self,
               num_spatial_dims,
               output_channels,
               kernel_shape,
               stride=1,
               rate=1,
               padding="SAME",
               with_bias=True,
               w_init=None,
               b_init=None,
               data_format="channels_last",
               mask=None,
               name=None):
    """Constructs a `ConvND` module.

    Args:
      num_spatial_dims: The number of spatial dimensions of the input.
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length `num_spatial_dims`.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length `num_spatial_dims`. Defaults to 1.
      rate: Optional kernel dilation rate. Either an integer or a sequence of
        length `num_spatial_dims`. 1 corresponds to standard ND convolution,
        `rate > 1` corresponds to dilated convolution. Defaults to 1.
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
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super(ConvND, self).__init__(name=name)

    if not 1 <= num_spatial_dims <= 3:
      raise ValueError(
          "We only support convolution operations for num_spatial_dims=1, 2 or "
          "3, received num_spatial_dims={}.".format(num_spatial_dims))
    self._num_spatial_dims = num_spatial_dims
    self._output_channels = output_channels
    self._kernel_shape = utils.replicate(kernel_shape, num_spatial_dims,
                                         "kernel_shape")
    self._with_bias = with_bias
    self._stride = utils.replicate(stride, num_spatial_dims, "strides")
    self._w_init = w_init
    self._b_init = b_init or jnp.zeros
    self._mask = mask
    self._lhs_dilation = utils.replicate(1, num_spatial_dims, "lhs_dilation")
    self._kernel_dilation = utils.replicate(rate, num_spatial_dims,
                                            "kernel_dilation")
    self._data_format = data_format
    self._channel_index = utils.get_channel_index(data_format)
    if self._channel_index == -1:
      self._dn = DIMENSION_NUMBERS[self._num_spatial_dims]
    else:
      self._dn = DIMENSION_NUMBERS_NCSPATIAL[self._num_spatial_dims]

    if isinstance(padding, str):
      self._padding = padding.upper()
    else:
      self._padding = pad.create(
          padding=padding,
          kernel=self._kernel_shape,
          rate=self._kernel_dilation,
          n=self._num_spatial_dims)

  def __call__(self, inputs):
    """Connects `ConvND` layer.

    Args:
      inputs: A rank-N+2 array with shape [N, spatial_dims, C].

    Returns:
      A rank-N+2 array with shape [N, spatial_dims, output_channels].
    """
    if len(inputs.shape) != self._num_spatial_dims + 2:
      raise ValueError("Input to ConvND needs to have rank {}, but input "
                       "has shape {}.".format(
                           self._num_spatial_dims + 2, inputs.shape))
    weight_shape = self._kernel_shape + (inputs.shape[self._channel_index],
                                         self._output_channels)

    fan_in_shape = np.prod(weight_shape[:-1])
    stddev = 1. / np.sqrt(fan_in_shape)
    w_init = self._w_init or initializers.TruncatedNormal(stddev=stddev)
    w = base.get_parameter("w", weight_shape, inputs.dtype, init=w_init)

    if self._mask is not None:
      if self._mask.shape != w.shape:
        raise ValueError("Mask needs to have the same shape as weights. "
                         "Shapes are: {}, {}".format(self._mask.shape, w.shape))
      w *= self._mask
    result = lax.conv_general_dilated(
        inputs,
        w,
        self._stride,
        self._padding,
        lhs_dilation=self._lhs_dilation,
        rhs_dilation=self._kernel_dilation,
        dimension_numbers=self._dn)
    if self._with_bias:
      if self._channel_index == -1:
        bias_shape = (self._output_channels,)
      else:
        bias_shape = (self._output_channels,) + (1,)*self._num_spatial_dims
      b = base.get_parameter("b", bias_shape, inputs.dtype, init=self._b_init)
      result = result + jnp.broadcast_to(b, result.shape)
    return result


class Conv1D(ConvND):
  """Conv1D module."""

  def __init__(self,
               output_channels,
               kernel_shape,
               stride=1,
               rate=1,
               padding="SAME",
               with_bias=True,
               w_init=None,
               b_init=None,
               data_format="NWC",
               mask=None,
               name=None):
    """Initializes a Conv1D module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 1.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 1. Defaults to 1.
      rate: Optional kernel dilation rate. Either an integer or a sequence of
        length 1. 1 corresponds to standard ND convolution,
        `rate > 1` corresponds to dilated convolution. Defaults to 1.
      padding: Optional padding algorithm. Either "VALID" or "SAME" or
        a callable or sequence of callables of length 1. Any callables must
        take a single integer argument equal to the effective kernel size and
        return a list of two integers representing the padding before and after.
        See haiku.pad.* for more details and example functions.
        Defaults to "SAME". See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either `NWC` or `NCW`. By
        default, `NWC`.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super(Conv1D, self).__init__(
        num_spatial_dims=1,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        name=name)


class Conv2D(ConvND):
  """Conv2D module."""

  def __init__(self,
               output_channels,
               kernel_shape,
               stride=1,
               rate=1,
               padding="SAME",
               with_bias=True,
               w_init=None,
               b_init=None,
               data_format="NHWC",
               mask=None,
               name=None):
    """Initializes a Conv2D module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 2.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 2. Defaults to 1.
      rate: Optional kernel dilation rate. Either an integer or a sequence of
        length 2. 1 corresponds to standard ND convolution,
        `rate > 1` corresponds to dilated convolution. Defaults to 1.
      padding: Optional padding algorithm. Either "VALID" or "SAME" or
        a callable or sequence of callables of length 2. Any callables must
        take a single integer argument equal to the effective kernel size and
        return a list of two integers representing the padding before and after.
        See haiku.pad.* for more details and example functions.
        Defaults to "SAME". See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either `NHWC` or `NCHW`. By
        default, `NHWC`.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super(Conv2D, self).__init__(
        num_spatial_dims=2,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        name=name)


class Conv3D(ConvND):
  """Conv3D module."""

  def __init__(self,
               output_channels,
               kernel_shape,
               stride=1,
               rate=1,
               padding="SAME",
               with_bias=True,
               w_init=None,
               b_init=None,
               data_format="NDHWC",
               mask=None,
               name=None):
    """Initializes a Conv3D module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 3.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 3. Defaults to 1.
      rate: Optional kernel dilation rate. Either an integer or a sequence of
        length 3. 1 corresponds to standard ND convolution,
        `rate > 1` corresponds to dilated convolution. Defaults to 1.
      padding: Optional padding algorithm. Either "VALID" or "SAME" or
        a callable or sequence of callables of length 3. Any callables must
        take a single integer argument equal to the effective kernel size and
        return a list of two integers representing the padding before and after.
        See haiku.pad.* for more details and example functions.
        Defaults to "SAME". See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either `NDHWC` or `NCDHW`. By
        default, `NDHWC`.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super(Conv3D, self).__init__(
        num_spatial_dims=3,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        name=name)


class ConvNDTranspose(module.Module):
  """ConvNDTranspose module."""

  def __init__(self,
               num_spatial_dims,
               output_channels,
               kernel_shape,
               stride=1,
               padding="SAME",
               with_bias=True,
               w_init=None,
               b_init=None,
               data_format="channels_last",
               mask=None,
               name=None):
    """Initializes a Conv2DTranspose module.

    Args:
      num_spatial_dims: The number of spatial dimensions of the input.
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length `num_spatial_dims`.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length `num_spatial_dims`. Defaults to 1.
      padding: Optional padding algorithm. Either "VALID" or "SAME".
        Defaults to "SAME". See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Can be either
        `channels_first`, `channels_last`, `N...C` or `NC...`. By default,
        `channels_last`.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super(ConvNDTranspose, self).__init__(name=name)
    if not 1 <= num_spatial_dims <= 3:
      raise ValueError(
          "We only support convolution operations for num_spatial_dims=1, 2 or "
          "3, received num_spatial_dims={}.".format(num_spatial_dims))
    self._num_spatial_dims = num_spatial_dims
    self._output_channels = output_channels
    self._kernel_shape = utils.replicate(kernel_shape, num_spatial_dims,
                                         "kernel_shape")
    self._with_bias = with_bias
    self._stride = utils.replicate(stride, num_spatial_dims, "strides")
    self._w_init = w_init
    self._b_init = b_init or jnp.zeros
    self._mask = mask
    self._padding = padding

    self._data_format = data_format
    self._channel_index = utils.get_channel_index(data_format)
    if self._channel_index == -1:
      self._dn = DIMENSION_NUMBERS[self._num_spatial_dims]
    else:
      self._dn = DIMENSION_NUMBERS_NCSPATIAL[self._num_spatial_dims]

  def __call__(self, inputs):
    """Connects Conv2DTranspose layer.

    Args:
      inputs: A rank-N+2 array with shape [N, spatial_dims, C].

    Returns:
      A rank-N+2 array with shape [N, spatial_dims, output_channels].
    """
    if len(inputs.shape) != self._num_spatial_dims + 2:
      raise ValueError("Input to ConvND needs to have rank {}, but input "
                       "has shape {}.".format(
                           self._num_spatial_dims + 2, inputs.shape))
    weight_shape = self._kernel_shape + (inputs.shape[self._channel_index],
                                         self._output_channels)

    fan_in_shape = np.sqrt(np.prod(weight_shape[:-1]))
    stddev = 1. / fan_in_shape
    w_init = self._w_init or initializers.TruncatedNormal(stddev=stddev)
    w = base.get_parameter("w", weight_shape, inputs.dtype, init=w_init)

    if self._mask is not None:
      if self._mask.shape != w.shape:
        raise ValueError("Mask needs to have the same shape as weights. "
                         "Shapes are: {}, {}".format(self._mask.shape, w.shape))
      w *= self._mask

    result = lax.conv_transpose(
        inputs,
        w,
        self._stride,
        self._padding,
        dimension_numbers=self._dn)
    if self._with_bias:
      if self._channel_index == -1:
        bias_shape = (self._output_channels,)
      else:
        bias_shape = (self._output_channels,) + (1,)*self._num_spatial_dims
      b = base.get_parameter("b", bias_shape, init=self._b_init)
      result = result + jnp.broadcast_to(b, result.shape)
    return result


class Conv1DTranspose(ConvNDTranspose):
  """Conv1DTranspose module."""

  def __init__(self,
               output_channels,
               kernel_shape,
               stride=1,
               padding="SAME",
               with_bias=True,
               w_init=None,
               b_init=None,
               data_format="NWC",
               mask=None,
               name=None):
    """Initializes a Conv1DTranspose module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 1.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 1. Defaults to 1.
      padding: Optional padding algorithm. Either "VALID" or "SAME".
        Defaults to "SAME". See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either `NWC` or `NCW`. By
        default, `NWC`.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super(Conv1DTranspose, self).__init__(
        num_spatial_dims=1,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        name=name)


class Conv2DTranspose(ConvNDTranspose):
  """Conv2DTranspose module."""

  def __init__(self,
               output_channels,
               kernel_shape,
               stride=1,
               padding="SAME",
               with_bias=True,
               w_init=None,
               b_init=None,
               data_format="NHWC",
               mask=None,
               name=None):
    """Initializes a Conv2DTranspose module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 2.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 2. Defaults to 1.
      padding: Optional padding algorithm. Either "VALID" or "SAME".
        Defaults to "SAME". See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either `NHWC` or `NCHW`. By
        default, `NHWC`.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super(Conv2DTranspose, self).__init__(
        num_spatial_dims=2,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        name=name)


class Conv3DTranspose(ConvNDTranspose):
  """Conv3DTranspose module."""

  def __init__(self,
               output_channels,
               kernel_shape,
               stride=1,
               padding="SAME",
               with_bias=True,
               w_init=None,
               b_init=None,
               data_format="NDHWC",
               mask=None,
               name=None):
    """Initializes a Conv3DTranspose module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 3.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 3. Defaults to 1.
      padding: Optional padding algorithm. Either "VALID" or "SAME".
        Defaults to "SAME". See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either `NDHWC` or `NCDHW`. By
        default, `NDHWC`.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super(Conv3DTranspose, self).__init__(
        num_spatial_dims=3,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        name=name)
