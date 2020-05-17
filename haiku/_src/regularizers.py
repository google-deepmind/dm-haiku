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
"""Haiku regularizers."""

from haiku._src import base
from haiku._src.typing import Shape, DType, Regularizer
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp


class L1(Regularizer):
  """L1 regularizer."""

  def __init__(self, scale):
    """Create an L1 regularizer.
    Args:
      scale: A non-negative regularization factor.
    Raises:
      ValueError: if scale is <0.
    """
    if scale < 0:
        raise ValueError("scale must be a non-negative value")
    self.scale = scale


  def __call__(self, parameters) -> jnp.array:
    values, _ = ravel_pytree(parameters)
    return self.scale * jnp.sum(jnp.abs(values))

class L2(Regularizer):
  """L2 regularizer."""

  def __init__(self, scale):
    """Create an L2 regularizer.
    Args:
      scale: A non-negative regularization factor.
    Raises:
      ValueError: if scale is <0.
    """
    if scale < 0:
        raise ValueError("scale must be a non-negative value")
    self.scale = scale


  def __call__(self, parameters) -> jnp.array:
    values, _ = ravel_pytree(parameters)
    return self.scale * jnp.sum(jnp.square(values))
