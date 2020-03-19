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
"""Layer Norm."""

import collections

from haiku._src import base
from haiku._src import module
from haiku._src import utils
import jax
import jax.numpy as jnp


class LayerNorm(module.Module):
  """LayerNorm module.

  See: https://arxiv.org/abs/1607.06450.
  """

  def __init__(self,
               axis,
               create_scale,
               create_offset,
               eps=1e-5,
               scale_init=None,
               offset_init=None,
               name=None):
    """Constructs a LayerNorm module.

    Args:
      axis: Integer, list of integers, or slice indicating which axes to
        normalize over.
      create_scale: Bool, defines whether to create a trainable scale
        per channel applied after the normalization.
      create_offset: Bool, defines whether to create a trainable offset
        per channel applied after normalization and scaling.
      eps: Small epsilon to avoid division by zero variance. Defaults 1e-5, as
        in the paper and Sonnet.
      scale_init: Optional initializer for gain (aka scale). By default, one.
      offset_init: Optional initializer for bias (aka offset). By default, zero.
      name: The module name.
    """
    super(LayerNorm, self).__init__(name=name)
    if isinstance(axis, slice):
      self._axis = axis
    elif isinstance(axis, int):
      self._axis = (axis,)
    elif (isinstance(axis, collections.Iterable) and
          all(isinstance(ax, int) for ax in axis)):
      self._axis = axis
    else:
      raise ValueError("`axis` should be an int, slice or iterable of ints.")

    self._eps = eps

    self._create_scale = create_scale
    self._create_offset = create_offset

    if self._create_scale:
      self._scale_init = scale_init or jnp.ones
    elif scale_init is not None:
      raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
    if self._create_offset:
      self._offset_init = offset_init or jnp.zeros
    elif offset_init is not None:
      raise ValueError("Cannot set `offset_init` if `create_offset=False`.")

  def __call__(self, inputs, scale=None, offset=None):
    """Connects the layer norm.

    Args:
      inputs: An array, where the data format is [N, ..., C].
      scale: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of `inputs`. This is the scale applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        `create_scale=True`.
      offset: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of `inputs`. This is the offset applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        `create_offset=True`.

    Returns:
      The array, normalized.
    """
    if isinstance(self._axis, slice):
      axes = tuple(range(len(inputs.shape)))
      axis = axes[self._axis]
    else:
      axis = self._axis

    m = jnp.mean(inputs, axis=axis, keepdims=True)
    variance = jnp.var(inputs, axis=axis, keepdims=True)
    param_shape = inputs.shape[-1:]
    if self._create_scale:
      if scale is not None:
        raise ValueError(
            "Cannot pass `scale` at call time if `create_scale=True`.")
      scale = base.get_parameter("scale", param_shape, init=self._scale_init)
    elif scale is None:
      scale = 1.

    if self._create_offset:
      if offset is not None:
        raise ValueError(
            "Cannot pass `offset` at call time if `create_offset=True`.")
      offset = base.get_parameter("offset", param_shape, init=self._offset_init)
    elif offset is None:
      offset = 0.

    scale = jnp.broadcast_to(scale, inputs.shape)
    offset = jnp.broadcast_to(offset, inputs.shape)
    m = jnp.broadcast_to(m, inputs.shape)

    inv = scale * jax.lax.rsqrt(variance + self._eps)
    return inv * (inputs - m) + offset


class InstanceNorm(LayerNorm):
  """Normalizes inputs along the spatial dimensions.

  See :class:`LayerNorm` for more details.
  """

  def __init__(self,
               create_scale,
               create_offset,
               eps=1e-5,
               scale_init=None,
               offset_init=None,
               data_format="channels_last",
               name=None):
    """Constructs an `InstanceNorm` module.

    This method creates a module which normalizes over the spatial dimensions.

    Args:
      create_scale: `bool` representing whether to create a trainable scale
        per channel applied after the normalization.
      create_offset: `bool` representing whether to create a trainable offset
        per channel applied after normalization and scaling.
      eps: Small epsilon to avoid division by zero variance. Defaults to
        `1e-5`.
      scale_init: Optional initializer for the scale variable. Can only be set
        if `create_scale=True`. By default scale is initialized to `1`.
      offset_init: Optional initializer for the offset variable. Can only be set
        if `create_offset=True`. By default offset is initialized to `0`.
      data_format: The data format of the input. Can be either
        `channels_first`, `channels_last`, `N...C` or `NC...`. By
        default it is `channels_last`.
      name: Name of the module.
    """
    if utils.get_channel_index(data_format) == 1:
      axis = slice(2, None)
    else:  # channel_index = -1
      axis = slice(1, -1)
    super(InstanceNorm, self).__init__(
        axis=axis,
        create_scale=create_scale,
        create_offset=create_offset,
        eps=eps,
        scale_init=scale_init,
        offset_init=offset_init,
        name=name)
