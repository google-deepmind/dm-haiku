# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Root Mean Square Layer Normalization.

Reference: https://arxiv.org/abs/1910.07467
"""

from collections import abc
from collections.abc import Sequence
from typing import Union

from haiku._src import base
from haiku._src import initializers
from haiku._src import layer_norm
from haiku._src import module
import jax
import jax.numpy as jnp


# If you are forking replace this with `import haiku as hk`.
# pylint: disable=invalid-name
class hk:
  get_parameter = base.get_parameter
  initializers = initializers
  Module = module.Module
# pylint: enable=invalid-name
del base, module, initializers

AxisOrAxes = Union[int, Sequence[int], slice]


class RMSNorm(hk.Module):
  """RMSNorm module.

  RMSNorm provides an alternative that can be both faster and more stable than
  LayerNorm. The inputs are normalized by the root-mean-squared (RMS) and scaled
  by a learned parameter, but they are not recentered around their mean.

  See https://arxiv.org/pdf/1910.07467.pdf
  """

  def __init__(
      self,
      axis: AxisOrAxes,
      eps: float = 1e-5,
      scale_init: hk.initializers.Initializer | None = None,
      name: str | None = None,
      create_scale: bool = True,
      *,
      param_axis: AxisOrAxes | None = None,
      ):
    """Constructs a RMSNorm module.

    Args:
      axis: Integer, list of integers, or slice indicating which axes to
        normalize over.
      eps: Small epsilon to avoid division by zero variance. Defaults to 1e-5.
      scale_init: Optional initializer for gain (aka scale). By default, one.
      name: The module name.
      create_scale: Bool, defines whether to create a trainable scale
        per channel applied after the normalization.
      param_axis: Axis used to determine the parameter shape of the learnable
        scale/offset. Sonnet sets this to the channel/feature axis (e.g. to
        ``-1`` for ``NHWC``). Other libraries set this to the same as the
        reduction axis (e.g. ``axis=param_axis``). `None` defaults to (-1,).
    """
    super().__init__(name=name)
    if not create_scale and scale_init is not None:
      raise ValueError("Cannot set `scale_init` if `create_scale=False`.")
    if isinstance(axis, slice):
      self.axis = axis
    elif isinstance(axis, int):
      self.axis = (axis,)
    elif (isinstance(axis, abc.Iterable) and
          all(isinstance(ax, int) for ax in axis)):
      self.axis = tuple(axis)
    else:
      raise ValueError("`axis` should be an int, slice or iterable of ints.")

    self.eps = eps
    self.create_scale = create_scale
    self.scale_init = scale_init or jnp.ones
    if param_axis is None:
      self.param_axis = (-1,)
    else:
      self.param_axis = layer_norm.to_axes_or_slice(param_axis)

  def __call__(self, inputs: jax.Array):
    """Connects the layer norm.

    Args:
      inputs: An array, where the data format is ``[N, ..., C]``.

    Returns:
      The normalized array, of the same shape as the inputs.
    """
    axis = self.axis
    if isinstance(axis, slice):
      axis = tuple(range(inputs.ndim)[axis])

    param_axis = layer_norm.to_abs_axes(self.param_axis, inputs.ndim)
    if param_axis == (inputs.ndim - 1,):
      # For param_axis=-1 we store non-broadcast param shape for compatibility
      # with older checkpoints.
      param_shape = inputs.shape[-1:]
    else:
      param_shape = tuple(
          (inputs.shape[i] if i in param_axis else 1)
          for i in range(inputs.ndim))
    if self.create_scale:
      scale = hk.get_parameter(
          "scale", param_shape, inputs.dtype, init=self.scale_init)
      scale = jnp.broadcast_to(scale, inputs.shape)
    else:
      scale = 1.

    mean_squared = jnp.mean(jnp.square(inputs), axis=axis, keepdims=True)
    mean_squared = jnp.broadcast_to(mean_squared, inputs.shape)

    return inputs * scale * jax.lax.rsqrt(mean_squared + self.eps)
