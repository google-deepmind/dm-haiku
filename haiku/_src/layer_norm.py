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
import types
from typing import Optional, Sequence, Union

from haiku._src import base
from haiku._src import initializers
from haiku._src import module
from haiku._src import utils
import jax
import jax.numpy as jnp
import numpy as np

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.get_parameter = base.get_parameter
hk.initializers = initializers
hk.Module = module.Module
del base, module, initializers


class LayerNorm(hk.Module):
  """LayerNorm module.

  See: https://arxiv.org/abs/1607.06450.
  """

  def __init__(
      self,
      axis: Union[int, Sequence[int], slice],
      create_scale: bool,
      create_offset: bool,
      eps: float = 1e-5,
      scale_init: Optional[hk.initializers.Initializer] = None,
      offset_init: Optional[hk.initializers.Initializer] = None,
      use_fast_variance: bool = False,
      name: Optional[str] = None,
  ):
    """Constructs a LayerNorm module.

    Args:
      axis: Integer, list of integers, or slice indicating which axes to
        normalize over.
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
    """
    super().__init__(name=name)
    if not create_scale and scale_init is not None:
      raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
    if not create_offset and offset_init is not None:
      raise ValueError("Cannot set `offset_init` if `create_offset=False`.")

    if isinstance(axis, slice):
      self.axis = axis
    elif isinstance(axis, int):
      self.axis = (axis,)
    elif (isinstance(axis, collections.abc.Iterable) and
          all(isinstance(ax, int) for ax in axis)):
      self.axis = tuple(axis)
    else:
      raise ValueError("`axis` should be an int, slice or iterable of ints.")

    self.eps = eps
    self.create_scale = create_scale
    self.create_offset = create_offset
    self.scale_init = scale_init or jnp.ones
    self.offset_init = offset_init or jnp.zeros
    self.use_fast_variance = use_fast_variance

  def __call__(
      self,
      inputs: jnp.ndarray,
      scale: Optional[jnp.ndarray] = None,
      offset: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
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

    axis = self.axis
    if isinstance(axis, slice):
      axis = tuple(range(inputs.ndim)[axis])

    mean = jnp.mean(inputs, axis=axis, keepdims=True)
    if self.use_fast_variance:
      mean_of_squares = jnp.mean(jnp.square(inputs), axis=axis, keepdims=True)
      variance = mean_of_squares - jnp.square(mean)
    else:
      variance = jnp.var(inputs, axis=axis, keepdims=True)

    param_shape = inputs.shape[-1:]
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
      scale_init: Optional[hk.initializers.Initializer] = None,
      offset_init: Optional[hk.initializers.Initializer] = None,
      data_format: str = "channels_last",
      name: Optional[str] = None,
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
        default it is ``channels_last``.
      name: Name of the module.
    """
    if utils.get_channel_index(data_format) == 1:
      axis = slice(2, None)
    else:  # channel_index = -1
      axis = slice(1, -1)
    super().__init__(
        axis=axis,
        create_scale=create_scale,
        create_offset=create_offset,
        eps=eps,
        scale_init=scale_init,
        offset_init=offset_init,
        name=name)
