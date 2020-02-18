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

from haiku._src import module
from jax import lax
import jax.numpy as jnp


def max_pool(value, window_shape, strides, padding):
  """Max pool.

  Args:
    value: Value to pool.
    window_shape: Shape of window to pool over. Same rank as value.
    strides: Strides for the window. Same rank as value.
    padding: Padding algorithm. Either "VALID" or "SAME".

  Returns:
    Pooled result. Same rank as value.
  """
  return lax.reduce_window(value, -jnp.inf, lax.max, window_shape, strides,
                           padding)


def avg_pool(value, window_shape, strides, padding):
  """Average pool.

  Args:
    value: Value to pool.
    window_shape: Shape of window to pool over. Same rank as value.
    strides: Strides for the window. Same rank as value.
    padding: Padding algorithm. Either "VALID" or "SAME".

  Returns:
    Pooled result. Same rank as value.

  Raises:
    ValueError: If the padding is not VALID.
  """
  reduce_window_args = (0., lax.add, window_shape, strides, padding)
  pooled = lax.reduce_window(value, *reduce_window_args)
  if padding == "VALID":
    # Avoid the extra reduce_window.
    return pooled / jnp.prod(window_shape)
  else:
    # Count the number of valid entries at each input point, then use that for
    # computing average. Assumes that any two arrays of same shape will be
    # padded the same.
    # TODO(tycai): This mask is computed at runtime. Give option to bake it
    # in as a constant instead.
    window_counts = lax.reduce_window(jnp.ones_like(value), *reduce_window_args)
    assert pooled.shape == window_counts.shape
    return pooled / window_counts


class MaxPool(module.Module):
  """Max pool.

  Equivalent to partial application of `hk.max_pool`.
  """

  def __init__(self, window_shape, strides, padding, name=None):
    """Max pool.

    Args:
      window_shape: Shape of window to pool over. Same rank as value.
      strides: Strides for the window. Same rank as value.
      padding: Padding algorithm. Either "VALID" or "SAME".
      name: String name for the module.
    """
    super(MaxPool, self).__init__(name=name)
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding

  def __call__(self, value):
    return max_pool(value, window_shape=self.window_shape, strides=self.strides,
                    padding=self.padding)


class AvgPool(module.Module):
  """Average pool.

  Equivalent to partial application of `hk.avg_pool`.
  """

  def __init__(self, window_shape, strides, padding, name=None):
    """Average pool.

    Args:
      window_shape: Shape of window to pool over. Same rank as value.
      strides: Strides for the window. Same rank as value.
      padding: Padding algorithm. Either "VALID" or "SAME".
      name: String name for the module.
    """
    super(AvgPool, self).__init__(name=name)
    self.window_shape = window_shape
    self.strides = strides
    self.padding = padding

  def __call__(self, value):
    return avg_pool(value, window_shape=self.window_shape, strides=self.strides,
                    padding=self.padding)
