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

import collections
import types
from typing import Optional, Sequence, Union

from haiku._src import base
from haiku._src import initializers
from haiku._src import module
import jax
import jax.numpy as jnp

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.get_parameter = base.get_parameter
hk.initializers = initializers
hk.Module = module.Module
del base, module, initializers


class RMSNorm(hk.Module):
  """RMSNorm module.

  RMSNorm provides an alternative that can be both faster and more stable than
  LayerNorm. The inputs are normalized by the root-mean-squared (RMS) and scaled
  by a learned parameter, but they are not recentered around their mean.

  See https://arxiv.org/pdf/1910.07467.pdf
  """

  def __init__(
      self,
      axis: Union[int, Sequence[int], slice],
      eps: float = 1e-5,
      scale_init: Optional[hk.initializers.Initializer] = None,
      name: Optional[str] = None):
    """Constructs a RMSNorm modulke.

    Args:
      axis: Integer, list of integers, or slice indicating which axes to
        normalize over.
      eps: Small epsilon to avoid division by zero variance. Defaults to 1e-5.
      scale_init: Optional initializer for gain (aka scale). By default, one.
      name: The module name.
    """
    super().__init__(name=name)
    if isinstance(axis, slice):
      self.axis = axis
    elif isinstance(axis, int):
      self.axis = (axis,)
    elif (isinstance(axis, collections.Iterable) and
          all(isinstance(ax, int) for ax in axis)):
      self.axis = tuple(axis)
    else:
      raise ValueError("`axis` should be an int, slice or iterable of ints.")

    self.eps = eps
    self.scale_init = scale_init or jnp.ones

  def __call__(self, inputs: jnp.ndarray):
    """Connects the layer norm.

    Args:
      inputs: An array, where the data format is ``[N, ..., C]``.

    Returns:
      The normalized array, of the same shape as the inputs..
    """
    axis = self.axis
    if isinstance(axis, slice):
      axis = tuple(range(inputs.ndim)[axis])

    scale = hk.get_parameter("scale", inputs.shape[-1:], inputs.dtype,
                             init=self.scale_init)
    scale = jnp.broadcast_to(scale, inputs.shape)

    mean_squared = jnp.mean(jnp.square(inputs), axis=axis, keepdims=True)
    mean_squared = jnp.broadcast_to(mean_squared, inputs.shape)

    return inputs * scale * jax.lax.rsqrt(mean_squared + self.eps)
