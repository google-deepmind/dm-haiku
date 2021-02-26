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
"""Haiku initializers."""

import types
from typing import Any, Sequence, Union

from haiku._src import base
from haiku._src.typing import Initializer
import jax
import jax.numpy as jnp
import numpy as np

# If forking replace this block with `import haiku as hk`.
hk = types.ModuleType('haiku')
hk.next_rng_key = base.next_rng_key
hk.initializers = types.ModuleType('haiku.initializers')
hk.initializers.Initializer = Initializer
del base


def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape."""
  if len(shape) < 1:
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in, fan_out = shape
  else:
    # Assuming convolution kernels (2D, 3D, or more.)
    # kernel_shape: (..., input_depth, depth)
    receptive_field_size = np.prod(shape[:-2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out


class Constant(hk.initializers.Initializer):
  """Initializes with a constant."""

  def __init__(self, constant: Union[float, int, jnp.ndarray]):
    """Constructs a Constant initializer.

    Args:
      constant: Constant to initialize with.
    """
    self.constant = constant

  def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
    return jnp.broadcast_to(self.constant, shape).astype(dtype)


class RandomNormal(hk.initializers.Initializer):
  """Initializes by sampling from a normal distribution."""

  def __init__(self, stddev=1., mean=0.):
    """Constructs a :class:`RandomNormal` initializer.

    Args:
      stddev: The standard deviation of the normal distribution to sample from.
      mean: The mean of the normal distribution to sample from.
    """
    self.stddev = stddev
    self.mean = mean

  def __call__(self, shape: Sequence[int], dtype) -> jnp.ndarray:
    m = jax.lax.convert_element_type(self.mean, dtype)
    s = jax.lax.convert_element_type(self.stddev, dtype)
    return m + s * jax.random.normal(hk.next_rng_key(), shape, dtype)


class TruncatedNormal(hk.initializers.Initializer):
  """Initializes by sampling from a truncated normal distribution."""

  def __init__(self,
               stddev: Union[float, jnp.ndarray] = 1.,
               mean: Union[float, jnp.ndarray] = 0.):
    """Constructs a :class:`TruncatedNormal` initializer.

    Args:
      stddev: The standard deviation parameter of the truncated
        normal distribution.
      mean: The mean of the truncated normal distribution.
    """
    self.stddev = stddev
    self.mean = mean

  def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
    m = jax.lax.convert_element_type(self.mean, dtype)
    s = jax.lax.convert_element_type(self.stddev, dtype)
    unscaled = jax.random.truncated_normal(hk.next_rng_key(), -2., 2., shape,
                                           dtype)
    return s * unscaled + m


class RandomUniform(hk.initializers.Initializer):
  """Initializes by sampling from a uniform distribution."""

  def __init__(self, minval=0., maxval=1.):
    """Constructs a :class:`RandomUniform` initializer.

    Args:
      minval: The lower limit of the uniform distribution.
      maxval: The upper limit of the uniform distribution.
    """
    self.minval = minval
    self.maxval = maxval

  def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
    return jax.random.uniform(hk.next_rng_key(), shape, dtype, self.minval,
                              self.maxval)


class VarianceScaling(hk.initializers.Initializer):
  """Initializer which adapts its scale to the shape of the initialized array.

  The initializer first computes the scaling factor ``s = scale / n``, where n
  is:

    - Number of input units in the weight tensor, if ``mode = fan_in``.
    - Number of output units, if ``mode = fan_out``.
    - Average of the numbers of input and output units, if ``mode = fan_avg``.

  Then, with ``distribution="truncated_normal"`` or ``"normal"``,
  samples are drawn from a distribution with a mean of zero and a standard
  deviation (after truncation, if used) ``stddev = sqrt(s)``.

  With ``distribution=uniform``, samples are drawn from a uniform distribution
  within ``[-limit, limit]``, with ``limit = sqrt(3 * s)``.

  The variance scaling initializer can be configured to generate other standard
  initializers using the scale, mode and distribution arguments. Here are some
  example configurations:

  ==============  ==============================================================
  Name            Parameters
  ==============  ==============================================================
  glorot_uniform  VarianceScaling(1.0, "fan_avg", "uniform")
  glorot_normal   VarianceScaling(1.0, "fan_avg", "truncated_normal")
  lecun_uniform   VarianceScaling(1.0, "fan_in",  "uniform")
  lecun_normal    VarianceScaling(1.0, "fan_in",  "truncated_normal")
  he_uniform      VarianceScaling(2.0, "fan_in",  "uniform")
  he_normal       VarianceScaling(2.0, "fan_in",  "truncated_normal")
  ==============  ==============================================================
  """

  def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal'):
    """Constructs the :class:`VarianceScaling` initializer.

    Args:
      scale: Scale to multiply the variance by.
      mode: One of ``fan_in``, ``fan_out``, ``fan_avg``
      distribution: Random distribution to use. One of ``truncated_normal``,
        ``normal`` or ``uniform``.
    """
    if scale < 0.0:
      raise ValueError('`scale` must be a positive float.')
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
      raise ValueError('Invalid `mode` argument:', mode)
    distribution = distribution.lower()
    if distribution not in {'normal', 'truncated_normal', 'uniform'}:
      raise ValueError('Invalid `distribution` argument:', distribution)
    self.scale = scale
    self.mode = mode
    self.distribution = distribution

  def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape)
    if self.mode == 'fan_in':
      scale /= max(1.0, fan_in)
    elif self.mode == 'fan_out':
      scale /= max(1.0, fan_out)
    else:
      scale /= max(1.0, (fan_in + fan_out) / 2.0)

    if self.distribution == 'truncated_normal':
      stddev = np.sqrt(scale)
      # Adjust stddev for truncation.
      # Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      distribution_stddev = np.asarray(.87962566103423978, dtype=dtype)
      stddev = stddev / distribution_stddev
      return TruncatedNormal(stddev=stddev)(shape, dtype)
    elif self.distribution == 'normal':
      stddev = np.sqrt(scale)
      return RandomNormal(stddev=stddev)(shape, dtype)
    else:
      limit = np.sqrt(3.0 * scale)
      return RandomUniform(minval=-limit, maxval=limit)(shape, dtype)


class UniformScaling(hk.initializers.Initializer):
  """Uniform scaling initializer.

  Initializes by sampling from a uniform distribution, but with the variance
  scaled by the inverse square root of the number of input units, multiplied by
  the scale.
  """

  def __init__(self, scale=1.0):
    """Constructs the :class:`UniformScaling` initializer.

    Args:
      scale: Scale to multiply the upper limit of the uniform distribution by.
    """
    self.scale = scale

  def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
    input_size = np.product(shape[:-1])
    max_val = np.sqrt(3 / input_size) * self.scale
    return RandomUniform(-max_val, max_val)(shape, dtype)


class Orthogonal(hk.initializers.Initializer):
  """Uniform scaling initializer."""

  def __init__(self, scale=1.0, axis=-1):
    """Construct an initializer for uniformly distributed orthogonal matrices.

    These matrices will be row-orthonormal along the access specified by
    ``axis``. If the rank of the weight is greater than 2, the shape will be
    flattened in all other dimensions and then will be row-orthonormal along the
    final dimension. Note that this only works if the ``axis`` dimension is
    larger, otherwise the matrix will be transposed (equivalently, it will be
    column orthonormal instead of row orthonormal).

    If the shape is not square, the matrices will have orthonormal rows or
    columns depending on which side is smaller.

    Args:
      scale: Scale factor.
      axis: Which axis corresponds to the "output dimension" of the tensor.

    Returns:
      An orthogonally initialized parameter.
    """
    self.scale = scale
    self.axis = axis

  def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
    if len(shape) < 2:
      raise ValueError('Orthogonal initializer requires at least a 2D shape.')
    n_rows = shape[self.axis]
    n_cols = np.prod(shape) // n_rows
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    norm_dst = jax.random.normal(hk.next_rng_key(), matrix_shape, dtype)
    q_mat, r_mat = jnp.linalg.qr(norm_dst)
    # Enforce Q is uniformly distributed
    q_mat *= jnp.sign(jnp.diag(r_mat))
    if n_rows < n_cols:
      q_mat = q_mat.T
    q_mat = jnp.reshape(q_mat, (n_rows,) + tuple(np.delete(shape, self.axis)))
    q_mat = jnp.moveaxis(q_mat, 0, self.axis)
    return jax.lax.convert_element_type(self.scale, dtype) * q_mat


class Identity(Initializer):
  """Initializer that generates the identity matrix.

  Constructs a 2D identity matrix or batches of these.
  """

  def __init__(self, gain: Union[float, jnp.ndarray] = 1.0):
    """Constructs an :class:`Identity` initializer.

    Args:
      gain: Multiplicative factor to apply to the identity matrix.
    """
    self.gain = gain

  def __call__(self, shape: Sequence[int], dtype: Any) -> jnp.ndarray:
    shape = tuple(shape)
    if len(shape) < 2:
      raise ValueError('Identity initializer requires at least a 2D shape.')

    eye = jnp.eye(shape[-2], shape[-1], dtype=dtype)
    if eye.shape != shape:
      eye = jnp.broadcast_to(eye, shape)
    gain = jax.lax.convert_element_type(self.gain, dtype)
    return gain * eye
