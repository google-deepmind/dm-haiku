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
"""Pooling Haiku modules."""

import types
from typing import Optional

from haiku._src import module
from haiku._src.typing import Shape, ShapeLike  # pylint: disable=g-multiple-import
from jax import lax
import jax.numpy as jnp

# If you are forking replace this block with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.Module = module.Module
del module


def _infer_shape(
    x: jnp.ndarray,
    size: ShapeLike,
    channel_axis: Optional[int] = -1,
) -> Shape:
  """Infer shape for pooling window or strides."""
  if isinstance(size, int):
    if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
      raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
    if channel_axis and channel_axis < 0:
      channel_axis = x.ndim + channel_axis
    return [1,] + [size if d != channel_axis else 1 for d in range(1, x.ndim)]
  else:
    assert x.ndim == len(size)
    return size


def max_pool(
    value: jnp.ndarray,
    window_shape: ShapeLike,
    strides: ShapeLike,
    padding: str,
    channel_axis: Optional[int] = -1,
) -> jnp.ndarray:
  """Max pool.

  Args:
    value: Value to pool.
    window_shape: Shape of the pooling window, an int or same rank as value.
    strides: Strides of the pooling window, an int or same rank as value.
    padding: Padding algorithm. Either "VALID" or "SAME".
    channel_axis: Axis of the spatial channels for which pooling is skipped,
      used to infer `window_shape` or `strides` if they are an integer.

  Returns:
    Pooled result. Same rank as value.
  """
  if padding not in ["SAME", "VALID"]:
    raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

  window_shape = _infer_shape(value, window_shape, channel_axis)
  strides = _infer_shape(value, strides, channel_axis)

  return lax.reduce_window(value, -jnp.inf, lax.max, window_shape, strides,
                           padding)


def avg_pool(
    value: jnp.ndarray,
    window_shape: ShapeLike,
    strides: ShapeLike,
    padding: str,
    channel_axis: Optional[int] = -1,
) -> jnp.ndarray:
  """Average pool.

  Args:
    value: Value to pool.
    window_shape: Shape of the pooling window, an int or same rank as value.
    strides: Strides of the pooling window, an int or same rank as value.
    padding: Padding algorithm. Either "VALID" or "SAME".
    channel_axis: Axis of the spatial channels for which pooling is skipped,
      used to infer `window_shape` or `strides` if they are an integer.

  Returns:
    Pooled result. Same rank as value.

  Raises:
    ValueError: If the padding is not VALID.
  """
  if padding not in ("SAME", "VALID"):
    raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

  window_shape = _infer_shape(value, window_shape, channel_axis)
  strides = _infer_shape(value, strides, channel_axis)

  reduce_window_args = (0., lax.add, window_shape, strides, padding)
  pooled = lax.reduce_window(value, *reduce_window_args)
  if padding == "VALID":
    # Avoid the extra reduce_window.
    return pooled / jnp.prod(window_shape)
  else:
    # Count the number of valid entries at each input point, then use that for
    # computing average. Assumes that any two arrays of same shape will be
    # padded the same.
    window_counts = lax.reduce_window(jnp.ones_like(value), *reduce_window_args)
    assert pooled.shape == window_counts.shape
    return pooled / window_counts


class MaxPool(hk.Module):
  """Max pool.

  Equivalent to partial application of `hk.max_pool`.
  """

  def __init__(
      self,
      window_shape: ShapeLike,
      strides: ShapeLike,
      padding: str,
      channel_axis: Optional[int] = -1,
      name: Optional[str] = None,
  ):
    """Max pool.

    Args:
      window_shape: Shape of window to pool over. Same rank as value or int.
      strides: Strides for the window. Same rank as value or int.
      padding: Padding algorithm. Either "VALID" or "SAME".
      channel_axis: Axis of the spatial channels for which pooling is skipped.
      name: String name for the module.
    """
    super().__init__(name=name)
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding
    self.channel_axis = channel_axis

  def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
    return max_pool(value, self.window_shape, self.strides,
                    self.padding, self.channel_axis)


class AvgPool(hk.Module):
  """Average pool.

  Equivalent to partial application of `hk.avg_pool`.
  """

  def __init__(
      self,
      window_shape: ShapeLike,
      strides: ShapeLike,
      padding: str,
      channel_axis: Optional[int] = -1,
      name: Optional[str] = None,
  ):
    """Average pool.

    Args:
      window_shape: Shape of window to pool over. Same rank as value or int.
      strides: Strides for the window. Same rank as value or int.
      padding: Padding algorithm. Either "VALID" or "SAME".
      channel_axis: Axis of the spatial channels for which pooling is skipped.
      name: String name for the module.
    """
    super().__init__(name=name)
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding
    self.channel_axis = channel_axis

  def __call__(self, value: jnp.ndarray) -> jnp.ndarray:
    return avg_pool(value, self.window_shape, self.strides,
                    self.padding, self.channel_axis)
