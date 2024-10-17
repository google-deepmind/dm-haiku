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
"""Layer Norm."""

import collections.abc
from collections.abc import Sequence

from haiku._src import base
from haiku._src import initializers
from haiku._src import module
from haiku._src import utils
import jax
import jax.numpy as jnp
import numpy as np


# If you are forking replace this with `import haiku as hk`.
# pylint: disable=invalid-name
class hk:
  get_parameter = base.get_parameter
  initializers = initializers
  Module = module.Module
  get_channel_index = utils.get_channel_index
# pylint: enable=invalid-name
del base, module, initializers, utils

AxisOrAxes = int | Sequence[int] | slice
AxesOrSlice = tuple[int, ...] | slice

# TODO(tomhennigan): Update users to `param_axis=-1` and flip + remove this.
ERROR_IF_PARAM_AXIS_NOT_EXPLICIT = False


def to_axes_or_slice(axis: AxisOrAxes) -> AxesOrSlice:
  if isinstance(axis, slice):
    return axis
  elif isinstance(axis, int):
    return (axis,)
  elif (isinstance(axis, collections.abc.Iterable) and
        all(isinstance(ax, int) for ax in axis)):
    return tuple(axis)
  else:
    raise ValueError(
        f"`axis` should be an int, slice or iterable of ints. Got: {axis}")


def to_abs_axes(axis: AxesOrSlice, ndim: int) -> tuple[int, ...]:
  if isinstance(axis, slice):
    return tuple(range(ndim)[axis])
  else:
    return tuple(sorted({a % ndim for a in axis}))


class LayerNorm(hk.Module):
  """LayerNorm module.

  See: https://arxiv.org/abs/1607.06450.

  Example usage:

  >>> ln = hk.LayerNorm(axis=-1, param_axis=-1,
  ...                   create_scale=True, create_offset=True)
  >>> x = ln(jnp.ones([8, 224, 224, 3]))
  """

  def __init__(
      self,
      axis: AxisOrAxes,
      create_scale: bool,
      create_offset: bool,
      eps: float = 1e-5,
      scale_init: hk.initializers.Initializer | None = None,
      offset_init: hk.initializers.Initializer | None = None,
      use_fast_variance: bool = False,
      name: str | None = None,
      *,
      param_axis: AxisOrAxes | None = None,
  ):
    """Constructs a LayerNorm module.

    Args:
      axis: Integer, list of integers, or slice indicating which axes to
        normalize over. Note that the shape of the scale/offset parameters are
        controlled by the ``param_axis`` argument.
      create_scale: Bool, defines whether to create a trainable scale
        per channel applied after the normalization.
      create_offset: Bool, defines whether to create a trainable offset
        per channel applied after normalization and scaling.
      eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
        as in the paper and Sonnet.
      scale_init: Optional initializer for gain (aka scale). By default, one.
      offset_init: Optional initializer for bias (aka offset). By default, zero.
      use_fast_variance: If true, use a faster but less numerically stable
        formulation for computing variance.
      name: The module name.
      param_axis: Axis used to determine the parameter shape of the learnable
        scale/offset. Sonnet sets this to the channel/feature axis (e.g. to
        ``-1`` for ``NHWC``). Other libraries set this to the same as the
        reduction axis (e.g. ``axis=param_axis``).
    """
    super().__init__(name=name)
    if not create_scale and scale_init is not None:
      raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
    if not create_offset and offset_init is not None:
      raise ValueError("Cannot set `offset_init` if `create_offset=False`.")

    self.axis = to_axes_or_slice(axis)
    self.eps = eps
    self.create_scale = create_scale
    self.create_offset = create_offset
    self.scale_init = scale_init or jnp.ones
    self.offset_init = offset_init or jnp.zeros
    self.use_fast_variance = use_fast_variance
    self._param_axis_passed_explicitly = param_axis is not None
    self.param_axis = (
        (-1,) if param_axis is None else to_axes_or_slice(param_axis))

  def __call__(
      self,
      inputs: jax.Array,
      scale: jax.Array | None = None,
      offset: jax.Array | None = None,
  ) -> jax.Array:
    """Connects the layer norm.

    Args:
      inputs: An array, where the data format is ``[N, ..., C]``.
      scale: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of ``inputs``. This is the scale applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        ``create_scale=True``.
      offset: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of ``inputs``. This is the offset applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        ``create_offset=True``.

    Returns:
      The array, normalized.
    """
    if self.create_scale and  scale is not None:
      raise ValueError(
          "Cannot pass `scale` at call time if `create_scale=True`.")
    if self.create_offset and offset is not None:
      raise ValueError(
          "Cannot pass `offset` at call time if `create_offset=True`.")

    axis = to_abs_axes(self.axis, inputs.ndim)
    mean = jnp.mean(inputs, axis=axis, keepdims=True)
    if self.use_fast_variance:
      mean_of_squares = jnp.mean(jnp.square(inputs), axis=axis, keepdims=True)
      variance = mean_of_squares - jnp.square(mean)
    else:
      variance = jnp.var(inputs, axis=axis, keepdims=True)

    if ((self.create_scale or self.create_offset) and
        not self._param_axis_passed_explicitly):
      if ERROR_IF_PARAM_AXIS_NOT_EXPLICIT and axis != (inputs.ndim - 1,):
        raise ValueError("When axis is not the final dimension we require "
                         "you to also pass `param_axis` in the ctor."
                         f" axis={axis} ndim={inputs.ndim}")

    # Shape for the learnable scale and offset is the number of channels.
    # See: https://arxiv.org/pdf/1803.08494.pdf around equation 6.
    param_axis = to_abs_axes(self.param_axis, inputs.ndim)
    if param_axis == (inputs.ndim - 1,):
      # For param_axis=-1 we store non-broadcast param shape for compatibility
      # with older checkpoints.
      param_shape = (inputs.shape[-1],)
    else:
      param_shape = tuple((inputs.shape[i] if i in param_axis else 1)
                          for i in range(inputs.ndim))

    if self.create_scale:
      scale = hk.get_parameter("scale", param_shape, inputs.dtype,
                               init=self.scale_init)
    elif scale is None:
      scale = np.array(1., dtype=inputs.dtype)

    if self.create_offset:
      offset = hk.get_parameter("offset", param_shape, inputs.dtype,
                                init=self.offset_init)
    elif offset is None:
      offset = np.array(0., dtype=inputs.dtype)

    if jax.config.jax_numpy_rank_promotion != "allow":
      # TODO(b/234327547): Explicit bcast triggers excessive mem usage on TPU.
      # We should remove the conditional (and always broadcast) when the
      # referenced bug is fixed.
      scale = jnp.broadcast_to(scale, inputs.shape)
      offset = jnp.broadcast_to(offset, inputs.shape)
      mean = jnp.broadcast_to(mean, inputs.shape)

    eps = jax.lax.convert_element_type(self.eps, variance.dtype)
    inv = scale * jax.lax.rsqrt(variance + eps)
    return inv * (inputs - mean) + offset


class InstanceNorm(LayerNorm):
  """Normalizes inputs along the spatial dimensions.

  See :class:`LayerNorm` for more details.
  """

  def __init__(
      self,
      create_scale: bool,
      create_offset: bool,
      eps: float = 1e-5,
      scale_init: hk.initializers.Initializer | None = None,
      offset_init: hk.initializers.Initializer | None = None,
      data_format: str = "channels_last",
      name: str | None = None,
  ):
    """Constructs an :class:`InstanceNorm` module.

    This method creates a module which normalizes over the spatial dimensions.

    Args:
      create_scale: ``bool`` representing whether to create a trainable scale
        per channel applied after the normalization.
      create_offset: ``bool`` representing whether to create a trainable offset
        per channel applied after normalization and scaling.
      eps: Small epsilon to avoid division by zero variance. Defaults to
        ``1e-5``.
      scale_init: Optional initializer for the scale variable. Can only be set
        if ``create_scale=True``. By default scale is initialized to ``1``.
      offset_init: Optional initializer for the offset variable. Can only be set
        if ``create_offset=True``. By default offset is initialized to ``0``.
      data_format: The data format of the input. Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default it is ``channels_last``. See :func:`get_channel_index`.
      name: Name of the module.
    """
    param_axis = hk.get_channel_index(data_format)
    if param_axis == 1:
      axis = slice(2, None)
    else:  # channel_index = -1
      assert param_axis == -1
      axis = slice(1, -1)
    super().__init__(
        axis=axis,
        create_scale=create_scale,
        create_offset=create_offset,
        eps=eps,
        scale_init=scale_init,
        offset_init=offset_init,
        param_axis=param_axis,
        name=name)
