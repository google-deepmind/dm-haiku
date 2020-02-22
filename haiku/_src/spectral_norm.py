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
"""Spectral Normalization tools.

This implementation follows the use in:
  https://arxiv.org/abs/1802.05957
  https://arxiv.org/abs/1805.08318
  https://arxiv.org/abs/1809.11096
"""

import re

from haiku._src import base
from haiku._src import data_structures
from haiku._src import initializers
from haiku._src import module
import jax
import jax.lax
import jax.numpy as jnp


def _l2_normalize(tensor, axis=None, eps=1e-12):
  """Normalizes along dimension `axis` using an L2 norm.

  This specialized function exists for numerical stability reasons.

  Args:
    tensor: An input ndarray.
    axis: Dimension along which to normalize, e.g. `1` to separately normalize
      vectors in a batch. Passing `None` views `t` as a flattened vector when
      calculating the norm (equivalent to Frobenius norm).
    eps: Epsilon to avoid dividing by zero.

  Returns:
    An array of the same shape as 't' L2-normalized along 'axis'.
  """
  return tensor * jax.lax.rsqrt((tensor * tensor).sum(
      axis=axis, keepdims=True) + eps)


class SpectralNorm(module.Module):
  """Normalizes an input by its first singular value.

  This module uses power iteration to calculate this value based on the
  input and an internal hidden state.
  """

  def __init__(self, eps=1e-4, n_steps=1, name=None):
    """Initializes an SpectralNorm module.

    Args:
      eps: The constant used for numerical stability.
      n_steps: How many steps of power iteration to perform to approximate the
        singular value of the input.
      name: The name of the module.
    """
    super(SpectralNorm, self).__init__(name=name)
    self._eps = eps
    self._n_steps = n_steps

  def __call__(self, value, update_stats=True, error_on_non_matrix=False):
    """Performs Spectral Normalization and returns the new value.

    Args:
      value: The array-like object for which you would like to perform an
        spectral normalization on.
      update_stats: A boolean defaulting to True. Regardless of this arg, this
        function will return the normalized input. When
        `update_stats` is True, the internal state of this object will also be
        updated to reflect the input value. When `update_stats` is False the
        internal stats will remain unchanged.
      error_on_non_matrix: Spectral normalization is only defined on matrices.
        By default, this module will return scalars unchanged and flatten
        higher-order tensors in their leading dimensions. Setting this flag to
        True will instead throw errors in those cases.
    Returns:
      The input value normalized by it's first singular value.
    Raises:
      ValueError: If `error_on_non_matrix` is True and `value` has ndims > 2.
    """
    value = jnp.asarray(value)
    value_shape = value.shape

    # Handle scalars.
    if value.ndim <= 1:
      raise ValueError("Spectral normalization is not well defined for "
                       "scalar or vector inputs.")
    # Handle higher-order tensors.
    elif value.ndim > 2:
      if error_on_non_matrix:
        raise ValueError(
            "Input is {}D but error_on_non_matrix is True".format(value.ndim))
      else:
        value = jnp.reshape(value, [-1, value.shape[-1]])

    u0 = base.get_state(
        "u0",
        shape=[1, value.shape[-1]],
        dtype=value.dtype,
        init=initializers.RandomNormal())

    # Power iteration for the weight's singular value.
    for _ in range(self._n_steps):
      v0 = _l2_normalize(jnp.matmul(u0, value.transpose([1, 0])), eps=self._eps)
      u0 = _l2_normalize(jnp.matmul(v0, value), eps=self._eps)

    u0 = jax.lax.stop_gradient(u0)
    v0 = jax.lax.stop_gradient(v0)

    sigma = jnp.matmul(jnp.matmul(v0, value), jnp.transpose(u0))[0, 0]

    value /= sigma
    value_bar = value.reshape(value_shape)

    if update_stats:
      base.set_state("u0", u0)
      base.set_state("sigma", sigma)
    return value_bar

  @property
  def u0(self):
    return base.get_state("u0")

  @property
  def sigma(self):
    return base.get_state("sigma", shape=(), init=jnp.ones)


class SNParamsTree(module.Module):
  """Applies Spectral Normalization to all parameters in a tree.

  This is isomorphic to EMAParamsTree in moving_averages.py.
  """

  def __init__(self, eps=1e-4, n_steps=1, ignore_regex="", name=None):
    """Initializes an SNParamsTree module.

    Args:
      eps: The constant used for numerical stability.
      n_steps: How many steps of power iteration to perform to approximate the
        singular value of the input.
      ignore_regex: A string. Any parameter in the tree whose name matches this
        regex will not have spectral normalization applied to it. The empty
        string means this module apply to all parameters.
      name: The name of the module.
    """
    super(SNParamsTree, self).__init__(name=name)
    self._eps = eps
    self._n_steps = n_steps
    self._ignore_regex = ignore_regex

  def __call__(self, tree, update_stats=True):
    def maybe_sn(k, v):
      if self._ignore_regex and re.match(self._ignore_regex, k):
        return v
      else:
        sn_name = k.replace("/", "__").replace("~", "_tilde")
        return SpectralNorm(self._eps, self._n_steps, name=sn_name)(
            v, update_stats=update_stats)

    # We want to potentially replace params with Spectral Normalized versions.
    new_values = {}
    for module_name, param_dict in tree.items():
      new_values[module_name] = {
          k: maybe_sn("/".join([module_name, k]), v)
          for k, v in param_dict.items()
      }
    return data_structures.to_immutable_dict(new_values)
