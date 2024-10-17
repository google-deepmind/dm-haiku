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

from collections.abc import Sequence
import warnings

from haiku._src import module
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


# If you are forking replace this block with `import haiku as hk`.
# pylint: disable=invalid-name
class hk:
  Module = module.Module
# pylint: enable=invalid-name
del module


def _infer_shape(
    x: jax.Array,
    size: int | Sequence[int],
    channel_axis: int | None = -1,
) -> tuple[int, ...]:
  """Infer shape for pooling window or strides."""
  if isinstance(size, int):
    if channel_axis and not 0 <= abs(channel_axis) < x.ndim:
      raise ValueError(f"Invalid channel axis {channel_axis} for {x.shape}")
    if channel_axis and channel_axis < 0:
      channel_axis = x.ndim + channel_axis
    return (1,) + tuple(size if d != channel_axis else 1
                        for d in range(1, x.ndim))
  elif len(size) < x.ndim:
    # Assume additional dimensions are batch dimensions.
    return (1,) * (x.ndim - len(size)) + tuple(size)
  else:
    assert x.ndim == len(size)
    return tuple(size)


_VMAP_SHAPE_INFERENCE_WARNING = (
    "When running under vmap, passing an `int` (except for `1`) for "
    "`window_shape` or `strides` will result in the wrong shape being inferred "
    "because the batch dimension is not visible to Haiku. Please update your "
    "code to specify a full unbatched size.\n"
    "For example if you had `pool(x, window_shape=3, strides=1)` before, you "
    "should now pass `pool(x, window_shape=(3, 3, 1), strides=1)`. \n"
    "Haiku will assume that any additional dimensions in your input are "
    "batch dimensions, and will pad `window_shape` and `strides` accordingly "
    "making your module support both batched and per-example inputs."
)


def _warn_if_unsafe(window_shape, strides):
  unsafe = lambda size: isinstance(size, int) and size != 1
  if unsafe(window_shape) or unsafe(strides):
    warnings.warn(_VMAP_SHAPE_INFERENCE_WARNING, DeprecationWarning)


def max_pool(
    value: jax.Array,
    window_shape: int | Sequence[int],
    strides: int | Sequence[int],
    padding: str,
    channel_axis: int | None = -1,
) -> jax.Array:
  """Max pool.

  Args:
    value: Value to pool.
    window_shape: Shape of the pooling window, same rank as value.
    strides: Strides of the pooling window, same rank as value.
    padding: Padding algorithm. Either ``VALID`` or ``SAME``.
    channel_axis: Axis of the spatial channels for which pooling is skipped.

  Returns:
    Pooled result. Same rank as value.
  """
  if padding not in ("SAME", "VALID"):
    raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

  _warn_if_unsafe(window_shape, strides)
  window_shape = _infer_shape(value, window_shape, channel_axis)
  strides = _infer_shape(value, strides, channel_axis)

  return lax.reduce_window(value, -jnp.inf, lax.max, window_shape, strides,
                           padding)


def avg_pool(
    value: jax.Array,
    window_shape: int | Sequence[int],
    strides: int | Sequence[int],
    padding: str,
    channel_axis: int | None = -1,
) -> jax.Array:
  """Average pool.

  Args:
    value: Value to pool.
    window_shape: Shape of the pooling window, same rank as value.
    strides: Strides of the pooling window, same rank as value.
    padding: Padding algorithm. Either ``VALID`` or ``SAME``.
    channel_axis: Axis of the spatial channels for which pooling is skipped.

  Returns:
    Pooled result. Same rank as value.

  Raises:
    ValueError: If the padding is not valid.
  """
  if padding not in ("SAME", "VALID"):
    raise ValueError(f"Invalid padding '{padding}', must be 'SAME' or 'VALID'.")

  _warn_if_unsafe(window_shape, strides)
  window_shape = _infer_shape(value, window_shape, channel_axis)
  strides = _infer_shape(value, strides, channel_axis)

  reduce_window_args = (0., lax.add, window_shape, strides, padding)
  pooled = lax.reduce_window(value, *reduce_window_args)
  if padding == "VALID":
    # Avoid the extra reduce_window.
    return pooled / np.prod(window_shape)
  else:
    # Count the number of valid entries at each input point, then use that for
    # computing average. Assumes that any two arrays of same shape will be
    # padded the same. Avoid broadcasting on axis where pooling is skipped.
    shape = [(v if w != 1 else 1) for (v, w) in zip(value.shape, window_shape)]
    window_counts = lax.reduce_window(
        jnp.ones(shape, value.dtype), *reduce_window_args)
    return pooled / window_counts


class MaxPool(hk.Module):
  """Max pool.

  Equivalent to partial application of :func:`max_pool`.
  """

  def __init__(
      self,
      window_shape: int | Sequence[int],
      strides: int | Sequence[int],
      padding: str,
      channel_axis: int | None = -1,
      name: str | None = None,
  ):
    """Max pool.

    Args:
      window_shape: Shape of the pooling window, same rank as value.
      strides: Strides of the pooling window, same rank as value.
      padding: Padding algorithm. Either ``VALID`` or ``SAME``.
      channel_axis: Axis of the spatial channels for which pooling is skipped.
      name: String name for the module.
    """
    super().__init__(name=name)
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding
    self.channel_axis = channel_axis

  def __call__(self, value: jax.Array) -> jax.Array:
    return max_pool(value, self.window_shape, self.strides,
                    self.padding, self.channel_axis)


class AvgPool(hk.Module):
  """Average pool.

  Equivalent to partial application of :func:`avg_pool`.
  """

  def __init__(
      self,
      window_shape: int | Sequence[int],
      strides: int | Sequence[int],
      padding: str,
      channel_axis: int | None = -1,
      name: str | None = None,
  ):
    """Average pool.

    Args:
      window_shape: Shape of the pooling window, same rank as value.
      strides: Strides of the pooling window, same rank as value.
      padding: Padding algorithm. Either ``VALID`` or ``SAME``.
      channel_axis: Axis of the spatial channels for which pooling is skipped.
      name: String name for the module.
    """
    super().__init__(name=name)
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding
    self.channel_axis = channel_axis

  def __call__(self, value: jax.Array) -> jax.Array:
    return avg_pool(value, self.window_shape, self.strides,
                    self.padding, self.channel_axis)
