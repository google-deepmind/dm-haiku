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

import types
from typing import Optional, Sequence, Union, Tuple

from haiku._src import base
from haiku._src import initializers
from haiku._src import module
from haiku._src import pad
from haiku._src import utils
from jax import lax
import jax.numpy as jnp
import numpy as np

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.initializers = initializers
hk.pad = pad
hk.get_parameter = base.get_parameter
hk.Module = module.Module
del base, module, initializers, pad


def to_dimension_numbers(
    num_spatial_dims: int,
    channels_last: bool,
    transpose: bool,
) -> lax.ConvDimensionNumbers:
  """Create a `lax.ConvDimensionNumbers` for the given inputs."""
  num_dims = num_spatial_dims + 2

  if channels_last:
    spatial_dims = tuple(range(1, num_dims - 1))
    image_dn = (0, num_dims - 1) + spatial_dims
  else:
    spatial_dims = tuple(range(2, num_dims))
    image_dn = (0, 1) + spatial_dims

  if transpose:
    kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
  else:
    kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))

  return lax.ConvDimensionNumbers(lhs_spec=image_dn, rhs_spec=kernel_dn,
                                  out_spec=image_dn)


class ConvND(hk.Module):
  """General N-dimensional convolutional."""

  def __init__(
      self,
      num_spatial_dims: int,
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      rate: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]], hk.pad.PadFn,
                     Sequence[hk.pad.PadFn]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "channels_last",
      mask: Optional[jnp.ndarray] = None,
      feature_group_count: int = 1,
      name: Optional[str] = None,
  ):
    """Initializes the module.

    Args:
      num_spatial_dims: The number of spatial dimensions of the input.
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length ``num_spatial_dims``.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length ``num_spatial_dims``. Defaults to 1.
      rate: Optional kernel dilation rate. Either an integer or a sequence of
        length ``num_spatial_dims``. 1 corresponds to standard ND convolution,
        ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or a
        sequence of n ``(low, high)`` integer pairs that give the padding to
        apply before and after each spatial dimension. or a callable or sequence
        of callables of size ``num_spatial_dims``. Any callables must take a
        single integer argument equal to the effective kernel size and return a
        sequence of two integers representing the padding before and after. See
        ``haiku.pad.*`` for more details and example functions. Defaults to
        ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input.  Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default, ``channels_last``.
      mask: Optional mask of the weights.
      feature_group_count: Optional number of groups in group convolution.
        Default value of 1 corresponds to normal dense convolution. If a higher
        value is used, convolutions are applied separately to that many groups,
        then stacked together. This reduces the number of parameters
        and possibly the compute for a given ``output_channels``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      name: The name of the module.
    """
    super().__init__(name=name)
    if num_spatial_dims <= 0:
      raise ValueError(
          "We only support convolution operations for `num_spatial_dims` "
          f"greater than 0, received num_spatial_dims={num_spatial_dims}.")

    self.num_spatial_dims = num_spatial_dims
    self.output_channels = output_channels
    self.kernel_shape = (
        utils.replicate(kernel_shape, num_spatial_dims, "kernel_shape"))
    self.with_bias = with_bias
    self.stride = utils.replicate(stride, num_spatial_dims, "strides")
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros
    self.mask = mask
    self.feature_group_count = feature_group_count
    self.lhs_dilation = utils.replicate(1, num_spatial_dims, "lhs_dilation")
    self.kernel_dilation = (
        utils.replicate(rate, num_spatial_dims, "kernel_dilation"))
    self.data_format = data_format
    self.channel_index = utils.get_channel_index(data_format)
    self.dimension_numbers = to_dimension_numbers(
        num_spatial_dims, channels_last=(self.channel_index == -1),
        transpose=False)

    if isinstance(padding, str):
      self.padding = padding.upper()
    elif hk.pad.is_padfn(padding):
      self.padding = hk.pad.create_from_padfn(padding=padding,
                                              kernel=self.kernel_shape,
                                              rate=self.kernel_dilation,
                                              n=self.num_spatial_dims)
    else:
      self.padding = hk.pad.create_from_tuple(padding, self.num_spatial_dims)

  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      precision: Optional[lax.Precision] = None,
  ) -> jnp.ndarray:
    """Connects ``ConvND`` layer.

    Args:
      inputs: An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
        or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.
      precision: Optional :class:`jax.lax.Precision` to pass to
        :func:`jax.lax.conv_general_dilated`.

    Returns:
      An array of shape ``[spatial_dims, output_channels]`` and rank-N+1 if
        unbatched, or an array of shape ``[N, spatial_dims, output_channels]``
        and rank-N+2 if batched.
    """
    unbatched_rank = self.num_spatial_dims + 1
    allowed_ranks = [unbatched_rank, unbatched_rank + 1]
    if inputs.ndim not in allowed_ranks:
      raise ValueError(f"Input to ConvND needs to have rank in {allowed_ranks},"
                       f" but input has shape {inputs.shape}.")

    unbatched = inputs.ndim == unbatched_rank
    if unbatched:
      inputs = jnp.expand_dims(inputs, axis=0)

    if inputs.shape[self.channel_index] % self.feature_group_count != 0:
      raise ValueError(f"Inputs channels {inputs.shape[self.channel_index]} "
                       f"should be a multiple of feature_group_count "
                       f"{self.feature_group_count}")
    w_shape = self.kernel_shape + (
        inputs.shape[self.channel_index] // self.feature_group_count,
        self.output_channels)

    if self.mask is not None and self.mask.shape != w_shape:
      raise ValueError("Mask needs to have the same shape as weights. "
                       f"Shapes are: {self.mask.shape}, {w_shape}")

    w_init = self.w_init
    if w_init is None:
      fan_in_shape = np.prod(w_shape[:-1])
      stddev = 1. / np.sqrt(fan_in_shape)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

    if self.mask is not None:
      w *= self.mask

    out = lax.conv_general_dilated(inputs,
                                   w,
                                   window_strides=self.stride,
                                   padding=self.padding,
                                   lhs_dilation=self.lhs_dilation,
                                   rhs_dilation=self.kernel_dilation,
                                   dimension_numbers=self.dimension_numbers,
                                   feature_group_count=self.feature_group_count,
                                   precision=precision)

    if self.with_bias:
      if self.channel_index == -1:
        bias_shape = (self.output_channels,)
      else:
        bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
      b = hk.get_parameter("b", bias_shape, inputs.dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    if unbatched:
      out = jnp.squeeze(out, axis=0)
    return out


class Conv1D(ConvND):
  """One dimensional convolution."""

  def __init__(
      self,
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      rate: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]], hk.pad.PadFn,
                     Sequence[hk.pad.PadFn]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "NWC",
      mask: Optional[jnp.ndarray] = None,
      feature_group_count: int = 1,
      name: Optional[str] = None,
  ):
    """Initializes the module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 1.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 1. Defaults to 1.
      rate: Optional kernel dilation rate. Either an integer or a sequence of
        length 1. 1 corresponds to standard ND convolution,
        ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or
        a callable or sequence of callables of length 1. Any callables must
        take a single integer argument equal to the effective kernel size and
        return a list of two integers representing the padding before and after.
        See haiku.pad.* for more details and example functions.
        Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either ``NWC`` or ``NCW``. By
        default, ``NWC``.
      mask: Optional mask of the weights.
      feature_group_count: Optional number of groups in group convolution.
        Default value of 1 corresponds to normal dense convolution. If a higher
        value is used, convolutions are applied separately to that many groups,
        then stacked together. This reduces the number of parameters
        and possibly the compute for a given ``output_channels``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      name: The name of the module.
    """
    super().__init__(
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
        feature_group_count=feature_group_count,
        name=name)


class Conv2D(ConvND):
  """Two dimensional convolution."""

  def __init__(
      self,
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      rate: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]], hk.pad.PadFn,
                     Sequence[hk.pad.PadFn]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "NHWC",
      mask: Optional[jnp.ndarray] = None,
      feature_group_count: int = 1,
      name: Optional[str] = None,
  ):
    """Initializes the module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 2.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 2. Defaults to 1.
      rate: Optional kernel dilation rate. Either an integer or a sequence of
        length 2. 1 corresponds to standard ND convolution,
        ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or
        a callable or sequence of callables of length 2. Any callables must
        take a single integer argument equal to the effective kernel size and
        return a list of two integers representing the padding before and after.
        See haiku.pad.* for more details and example functions.
        Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either ``NHWC`` or ``NCHW``. By
        default, ``NHWC``.
      mask: Optional mask of the weights.
      feature_group_count: Optional number of groups in group convolution.
        Default value of 1 corresponds to normal dense convolution. If a higher
        value is used, convolutions are applied separately to that many groups,
        then stacked together. This reduces the number of parameters
        and possibly the compute for a given ``output_channels``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      name: The name of the module.
    """
    super().__init__(
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
        feature_group_count=feature_group_count,
        name=name)


class Conv3D(ConvND):
  """Three dimensional convolution."""

  def __init__(
      self,
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      rate: Union[int, Sequence[int]] = 1,
      padding: Union[str, Sequence[Tuple[int, int]], hk.pad.PadFn,
                     Sequence[hk.pad.PadFn]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "NDHWC",
      mask: Optional[jnp.ndarray] = None,
      feature_group_count: int = 1,
      name: Optional[str] = None,
  ):
    """Initializes the module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 3.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 3. Defaults to 1.
      rate: Optional kernel dilation rate. Either an integer or a sequence of
        length 3. 1 corresponds to standard ND convolution,
        `rate > 1` corresponds to dilated convolution. Defaults to 1.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or
        a callable or sequence of callables of length 3. Any callables must
        take a single integer argument equal to the effective kernel size and
        return a list of two integers representing the padding before and after.
        See haiku.pad.* for more details and example functions.
        Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either ``NDHWC`` or ``NCDHW``.
        By default, ``NDHWC``.
      mask: Optional mask of the weights.
      feature_group_count: Optional number of groups in group convolution.
        Default value of 1 corresponds to normal dense convolution. If a higher
        value is used, convolutions are applied separately to that many groups,
        then stacked together. This reduces the number of parameters
        and possibly the compute for a given ``output_channels``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      name: The name of the module.
    """
    super().__init__(
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
        feature_group_count=feature_group_count,
        name=name)


def compute_adjusted_padding(
    input_size: int,
    output_size: int,
    kernel_size: int,
    stride: int,
    padding: str,
    dilation: int = 1,
) -> Tuple[int, int]:
  """Computes adjusted padding for desired ConvTranspose `output_size`."""
  kernel_size = (kernel_size - 1) * dilation + 1
  if padding == "VALID":
    expected_input_size = (output_size - kernel_size + stride) // stride
    if input_size != expected_input_size:
      raise ValueError(f"The expected input size with the current set of input "
                       f"parameters is {expected_input_size} which doesn't "
                       f"match the actual input size {input_size}.")
    padding_before = 0
  elif padding == "SAME":
    expected_input_size = (output_size + stride - 1) // stride
    if input_size != expected_input_size:
      raise ValueError(f"The expected input size with the current set of input "
                       f"parameters is {expected_input_size} which doesn't "
                       f"match the actual input size {input_size}.")
    padding_needed = max(0,
                         (input_size - 1) * stride + kernel_size - output_size)
    padding_before = padding_needed // 2
  else:
    raise ValueError(f"`padding` must be 'VALID' or 'SAME'. Passed: {padding}.")

  expanded_input_size = (input_size - 1) * stride + 1
  padded_out_size = output_size + kernel_size - 1
  pad_before = kernel_size - 1 - padding_before
  pad_after = padded_out_size - expanded_input_size - pad_before
  return (pad_before, pad_after)


class ConvNDTranspose(hk.Module):
  """General n-dimensional transposed convolution (aka. deconvolution)."""

  def __init__(
      self,
      num_spatial_dims: int,
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      output_shape: Optional[Union[int, Sequence[int]]] = None,
      padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "channels_last",
      mask: Optional[jnp.ndarray] = None,
      name: Optional[str] = None,
  ):
    """Initializes the module.

    Args:
      num_spatial_dims: The number of spatial dimensions of the input.
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length ``num_spatial_dims``.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length ``num_spatial_dims``. Defaults to 1.
      output_shape: Output shape of the spatial dimensions of a transpose
        convolution. Can be either an integer or an iterable of integers. If a
        `None` value is given, a default shape is automatically calculated.
      padding: Optional padding algorithm. Either "VALID" or "SAME".
        Defaults to "SAME". See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default, ``channels_last``.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super().__init__(name=name)

    if num_spatial_dims <= 0:
      raise ValueError(
          "We only support convolution operations for `num_spatial_dims` "
          f"greater than 0, received num_spatial_dims={num_spatial_dims}.")

    self.num_spatial_dims = num_spatial_dims
    self.output_channels = output_channels
    self.kernel_shape = (
        utils.replicate(kernel_shape, num_spatial_dims, "kernel_shape"))
    self.output_shape = output_shape
    if self.output_shape is not None:
      self.output_shape = (
          utils.replicate(output_shape, num_spatial_dims, "output_shape"))
      if not isinstance(padding, str):
        raise ValueError("When specifying `output_shape`, ensure that paddding "
                         "is 'VALID' or 'SAME'.")
    self.with_bias = with_bias
    self.stride = utils.replicate(stride, num_spatial_dims, "strides")
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros
    self.mask = mask
    # TODO(tomhennigan) Make use of hk.pad.create_from_tuple here?
    self.padding = padding
    self.data_format = data_format
    self.channel_index = utils.get_channel_index(data_format)
    self.dimension_numbers = to_dimension_numbers(
        num_spatial_dims, channels_last=(self.channel_index == -1),
        transpose=True)

  def __call__(
      self,
      inputs: jnp.ndarray,
      *,
      precision: Optional[lax.Precision] = None,
  ) -> jnp.ndarray:
    """Computes the transposed convolution of the input.

    Args:
      inputs: An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
        or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.
      precision: Optional :class:`jax.lax.Precision` to pass to
        :func:`jax.lax.conv_transpose`.

    Returns:
      An array of shape ``[spatial_dims, output_channels]`` and rank-N+1 if
        unbatched, or an array of shape ``[N, spatial_dims, output_channels]``
        and rank-N+2 if batched.
    """
    unbatched_rank = self.num_spatial_dims + 1
    allowed_ranks = [unbatched_rank, unbatched_rank + 1]
    if inputs.ndim not in allowed_ranks:
      raise ValueError(f"Input to ConvNDTranspose needs to have rank in "
                       f"{allowed_ranks}, but input has shape {inputs.shape}.")

    unbatched = inputs.ndim == unbatched_rank
    if unbatched:
      inputs = jnp.expand_dims(inputs, axis=0)

    input_channels = inputs.shape[self.channel_index]
    w_shape = self.kernel_shape + (self.output_channels, input_channels)

    if self.mask is not None and self.mask.shape != w_shape:
      raise ValueError("Mask needs to have the same shape as weights. "
                       f"Shapes are: {self.mask.shape}, {w_shape}")

    w_init = self.w_init
    if w_init is None:
      fan_in_shape = self.kernel_shape + (input_channels,)
      stddev = 1. / np.sqrt(np.prod(fan_in_shape))
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

    if self.mask is not None:
      w = w * self.mask

    padding = self.padding
    if self.output_shape is not None:
      input_shape = (
          inputs.shape[2:] if self.channel_index == 1 else inputs.shape[1:-1])
      padding = tuple(map(
          lambda i, o, k, s: compute_adjusted_padding(i, o, k, s, self.padding),
          input_shape, self.output_shape, self.kernel_shape, self.stride))

    out = lax.conv_transpose(inputs,
                             w,
                             strides=self.stride,
                             padding=padding,
                             dimension_numbers=self.dimension_numbers,
                             precision=precision)

    if self.with_bias:
      if self.channel_index == -1:
        bias_shape = (self.output_channels,)
      else:
        bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
      b = hk.get_parameter("b", bias_shape, inputs.dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    if unbatched:
      out = jnp.squeeze(out, axis=0)
    return out


class Conv1DTranspose(ConvNDTranspose):
  """One dimensional transposed convolution (aka. deconvolution)."""

  def __init__(
      self,
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      output_shape: Optional[Union[int, Sequence[int]]] = None,
      padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "NWC",
      mask: Optional[jnp.ndarray] = None,
      name: Optional[str] = None,
  ):
    """Initializes the module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 1.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 1. Defaults to 1.
      output_shape: Output shape of the spatial dimensions of a transpose
        convolution. Can be either an integer or an iterable of integers. If a
        `None` value is given, a default shape is automatically calculated.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME``.
        Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either ``NWC`` or ``NCW``. By
        default, ``NWC``.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super().__init__(
        num_spatial_dims=1,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        output_shape=output_shape,
        stride=stride,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        name=name)


class Conv2DTranspose(ConvNDTranspose):
  """Two dimensional transposed convolution (aka. deconvolution)."""

  def __init__(
      self,
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      output_shape: Optional[Union[int, Sequence[int]]] = None,
      padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "NHWC",
      mask: Optional[jnp.ndarray] = None,
      name: Optional[str] = None,
  ):
    """Initializes the module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 2.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 2. Defaults to 1.
      output_shape: Output shape of the spatial dimensions of a transpose
        convolution. Can be either an integer or an iterable of integers. If a
        `None` value is given, a default shape is automatically calculated.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME``.
        Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either ``NHWC`` or ``NCHW``. By
        default, ``NHWC``.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super().__init__(
        num_spatial_dims=2,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        output_shape=output_shape,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        name=name)


class Conv3DTranspose(ConvNDTranspose):
  """Three dimensional transposed convolution (aka. deconvolution)."""

  def __init__(
      self,
      output_channels: int,
      kernel_shape: Union[int, Sequence[int]],
      stride: Union[int, Sequence[int]] = 1,
      output_shape: Optional[Union[int, Sequence[int]]] = None,
      padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
      with_bias: bool = True,
      w_init: Optional[hk.initializers.Initializer] = None,
      b_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "NDHWC",
      mask: Optional[jnp.ndarray] = None,
      name: Optional[str] = None,
  ):
    """Initializes the module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: The shape of the kernel. Either an integer or a sequence of
        length 3.
      stride: Optional stride for the kernel. Either an integer or a sequence of
        length 3. Defaults to 1.
      output_shape: Output shape of the spatial dimensions of a transpose
        convolution. Can be either an integer or an iterable of integers. If a
        `None` value is given, a default shape is automatically calculated.
      padding: Optional padding algorithm. Either ``VALID`` or ``SAME``.
        Defaults to ``SAME``. See:
        https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
      with_bias: Whether to add a bias. By default, true.
      w_init: Optional weight initialization. By default, truncated normal.
      b_init: Optional bias initialization. By default, zeros.
      data_format: The data format of the input. Either ``NDHWC`` or ``NCDHW``.
        By default, ``NDHWC``.
      mask: Optional mask of the weights.
      name: The name of the module.
    """
    super().__init__(
        num_spatial_dims=3,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        output_shape=output_shape,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        mask=mask,
        name=name)
