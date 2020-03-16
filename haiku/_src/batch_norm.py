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
"""Batch Norm."""

from haiku._src import base
from haiku._src import module
from haiku._src import moving_averages
from haiku._src import utils
import jax
import jax.numpy as jnp


class BatchNorm(module.Module):
  """BatchNorm module.

  See: https://arxiv.org/abs/1502.03167.

  There are many different variations for how users want to manage scale and
  offset if they require them at all. These are:

    - No scale/offset in which case `create_*` should be set to `False` and
      `scale`/`offset` aren't passed when the module is called.
    - Trainable scale/offset in which case `create_*` should be set to
      `True` and again `scale`/`offset` aren't passed when the module is
      called. In this case this module creates and owns the `scale`/`offset`
      variables.
    - Externally generated `scale`/`offset`, such as for conditional
      normalization, in which case `create_*` should be set to `False` and
      then the values fed in at call time.

  NOTE: `jax.vmap(hk.transform(BatchNorm))` will update summary statistics and
  normalize values on a per-batch basis; we currently do *not* support
  normalizing across a batch axis introduced by vmap.
  """

  def __init__(self,
               create_scale,
               create_offset,
               decay_rate,
               eps=1e-5,
               scale_init=None,
               offset_init=None,
               axis=None,
               cross_replica_axis=None,
               data_format="channels_last",
               name=None):
    """Constructs a BatchNorm module.

    Args:
      create_scale: Whether to include a trainable scaling factor.
      create_offset: Whether to include a trainable offset.
      decay_rate: Decay rate for EMA.
      eps: Small epsilon to avoid division by zero variance. Defaults 1e-5, as
        in the paper and Sonnet.
      scale_init: Optional initializer for gain (aka scale). Can only be set
        if `create_scale=True`. By default, one.
      offset_init: Optional initializer for bias (aka offset). Can only be set
        if `create_offset=True`. By default, zero.
      axis: Which axes to reduce over. The default (None)
        signifies that all but the channel axis should be normalized. Otherwise
        this is a list of axis indices which will have normalization
        statistics calculated.
      cross_replica_axis: If not None, it should be a string representing
        the axis name over which this module is being run within a jax.pmap.
        Supplying this argument means that batch statistics are calculated
        across all replicas on that axis.
      data_format: The data format of the input. Can be either
        `channels_first`, `channels_last`, `N...C` or `NC...`. By
        default it is `channels_last`.
      name: The module name.
    """
    super(BatchNorm, self).__init__(name=name)
    self._create_scale = create_scale
    self._create_offset = create_offset
    if not self._create_scale and scale_init is not None:
      raise ValueError("Cannot set `scale_init` if `create_scale=False`")
    self._scale_init = scale_init or jnp.ones
    if not self._create_offset and offset_init is not None:
      raise ValueError("Cannot set `offset_init` if `create_offset=False`")
    self._offset_init = offset_init or jnp.zeros
    self._eps = eps

    self._cross_replica_axis = cross_replica_axis
    self._data_format = data_format
    self._channel_index = utils.get_channel_index(data_format)
    self._axis = axis

    self._mean_ema = moving_averages.ExponentialMovingAverage(
        decay_rate, name="mean_ema")
    self._var_ema = moving_averages.ExponentialMovingAverage(
        decay_rate, name="var_ema")

  def __call__(self, inputs, is_training, test_local_stats=False,
               scale=None, offset=None):
    """Connects the batch norm.

    Args:
      inputs: An array, where the data format is [..., C].
      is_training: Whether this is during training.
      test_local_stats: Whether local stats are used when is_training=False.
      scale: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of `inputs`. This is the scale applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        `create_scale=True`.
      offset: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of `inputs`. This is the offset applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        `create_offset=True`.

    Returns:
      The array, normalized across all but the last dimension.
    """
    rank = inputs.ndim
    channel_index = self._channel_index
    if self._channel_index < 0:
      channel_index += rank

    if self._axis:
      axis = self._axis
    else:
      axis = [i for i in range(rank) if i != channel_index]
    if is_training or test_local_stats:
      if self._cross_replica_axis:
        # Calculate global statistics - n is the number of replicas which could
        # differ from jax.device_count() in cases of nested pmaps.
        n = jax.lax.psum(1, self._cross_replica_axis)

        mean = jnp.mean(inputs, axis, keepdims=True)
        mean = jax.lax.psum(mean, axis_name=self._cross_replica_axis) / n
        mean_of_squares = jnp.mean(inputs**2, axis, keepdims=True)
        mean_of_squares = jax.lax.psum(
            mean_of_squares, axis_name=self._cross_replica_axis) / n
        var = mean_of_squares - mean ** 2
      else:
        mean = jnp.mean(inputs, axis, keepdims=True)
        # This uses E[(X - E[X])^2].
        # TODO(tycai): Consider the faster, but possibly less stable
        # E[X^2] - E[X]^2 method.
        var = jnp.var(inputs, axis, keepdims=True)
    else:
      mean = self._mean_ema.average
      var = self._var_ema.average

    # Update moving averages.
    if is_training:
      self._mean_ema(mean)
      self._var_ema(var)

    params_shape = tuple(
        1 if i in axis else inputs.shape[i] for i in range(rank))
    if self._create_scale:
      if scale is not None:
        raise ValueError(
            "Cannot pass `scale` at call time if `create_scale=True`.")
      scale = base.get_parameter("scale", params_shape, inputs.dtype,
                                 self._scale_init)
    elif scale is None:
      scale = 1.
    if self._create_offset:
      if offset is not None:
        raise ValueError(
            "Cannot pass `offset` at call time if `create_offset=True`.")
      offset = base.get_parameter("offset", params_shape, inputs.dtype,
                                  self._offset_init)
    elif offset is None:
      offset = 0.

    inv = scale * jax.lax.rsqrt(var + self._eps)
    return (inputs - mean) * inv + offset
