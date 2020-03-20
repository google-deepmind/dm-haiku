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
"""Moving averages."""

import re

from haiku._src import base
from haiku._src import data_structures
from haiku._src import initializers
from haiku._src import module
import jax
import jax.numpy as jnp


class ExponentialMovingAverage(module.Module):
  """Maintains an exponential moving average.

  This uses the Adam debiasing procedure.
  See https://arxiv.org/pdf/1412.6980.pdf for details.
  """

  def __init__(self, decay, zero_debias=True, warmup_length=0, name=None):
    """Initializes an ExponentialMovingAverage module.

    Args:
      decay: The chosen decay. Must in [0, 1). Values close to 1 result in slow
        decay; values close to 0 result in fast decay.
      zero_debias: Whether to run with zero-debiasing.
      warmup_length: A positive integer, EMA has no effect until
        the internal counter has reached `warmup_length` at which point the
        initial value for the decaying average is initialized to the input value
        after `warmup_length` iterations.
      name: The name of the module.
    """
    super(ExponentialMovingAverage, self).__init__(name=name)
    self._decay = decay
    if warmup_length < 0:
      raise ValueError(
          f"`warmup_length` is {warmup_length}, but should be non-negative.")
    self._warmup_length = warmup_length
    self._zero_debias = zero_debias
    if warmup_length and zero_debias:
      raise ValueError(
          "Zero debiasing does not make sense when warming up the value of the "
          "average to an initial value. Set zero_debias=False if setting "
          "warmup_length to a non-zero value.")

  def _cond(self, cond, t, f, dtype):
    """Internal, implements jax.lax.cond without control flow."""
    c = cond.astype(dtype)
    return c * t + (1. - c) * f

  def __call__(self, value, update_stats=True):
    """Updates the EMA and returns the new value.

    Args:
      value: The array-like object for which you would like to perform an
        exponential decay on.
      update_stats: A Boolean, whether to update the internal state
        of this object to reflect the input value. When `update_stats` is False
        the internal stats will remain unchanged.

    Returns:
      The exponentially weighted average of the input value.
    """
    if not isinstance(value, jnp.ndarray):
      value = jnp.asarray(value)

    counter = base.get_state("counter", (), jnp.int32,
                             init=initializers.Constant(-self._warmup_length))
    counter += 1

    decay = jax.lax.convert_element_type(self._decay, value.dtype)
    if self._warmup_length > 0:
      decay = self._cond(counter <= 0, 0.0, decay, value.dtype)

    one = jnp.ones([], value.dtype)
    hidden = base.get_state("hidden", value.shape, value.dtype, init=jnp.zeros)
    hidden = hidden * decay + value * (one - decay)

    average = hidden
    if self._zero_debias:
      average /= (one - jnp.power(decay, counter))

    if update_stats:
      base.set_state("counter", counter)
      base.set_state("hidden", hidden)
      base.set_state("average", average)

    return average

  @property
  def average(self):
    return base.get_state("average")


class EMAParamsTree(module.Module):
  """Maintains an exponential moving average for all parameters in a tree.

  While ExponentialMovingAverage is meant to be applied to single parameters
  within a function, this class is meant to be applied to the entire tree of
  parameters for a function.

  Given a set of parameters for some network:

  >>> network_fn = lambda x: hk.Linear(10)(x)
  >>> x = jnp.ones([1, 1])
  >>> params = hk.transform(network_fn).init(jax.random.PRNGKey(428), x)

  You might use the EMAParamsTree like follows:

  >>> ema_fn = hk.transform_with_state(lambda x: hk.EMAParamsTree(0.2)(x))
  >>> _, ema_state = ema_fn.init(None, params)
  >>> ema_params, ema_state = ema_fn.apply(None, ema_state, None, params)

  Here, we are transforming a Haiku function and constructing its parameters via
  an init_fn as normal, but are creating a second transformed function which
  expects a tree of parameters as input. This function is then called with
  the current parameters as input, which then returns an identical tree with
  every parameter replaced with its exponentially decayed average. This
  ema_params object can then be passed into the `network_fn` as usual, and will
  cause it to run with EMA weights.
  """

  def __init__(self, decay, zero_debias=True, warmup_length=0, ignore_regex="",
               name=None):
    """Initializes an EMAParamsTree module.

    Args:
      decay: The chosen decay. Must in [0, 1). Values close to 1 result in slow
        decay; values close to 0 result in fast decay.
      zero_debias: Whether to run with zero-debiasing.
      warmup_length: A positive integer, EMA has no effect until
        the internal counter has reached `warmup_length` at which point the
        initial value for the decaying average is initialized to the input value
        after `warmup_length` iterations.
      ignore_regex: A string. Any parameter in the tree whose name matches this
        regex will not have any moving average applied to it. The empty string
        means this module will EMA all parameters.
      name: The name of the module.
    """
    super(EMAParamsTree, self).__init__(name=name)
    self._decay = decay
    self._zero_debias = zero_debias
    self._warmup_length = warmup_length
    self._ignore_regex = ignore_regex

  def __call__(self, tree, update_stats=True):
    def maybe_ema(k, v):
      if self._ignore_regex and re.match(self._ignore_regex, k):
        return v
      else:
        ema_name = k.replace("/", "__").replace("~", "_tilde_")
        return ExponentialMovingAverage(
            self._decay, self._zero_debias, self._warmup_length, name=ema_name)(
                v, update_stats=update_stats)

    # We want to potentially replace params with EMA'd versions.
    new_values = {}
    for module_name, param_dict in tree.items():
      new_values[module_name] = {
          k: maybe_ema("/".join([module_name, k]), v)
          for k, v in param_dict.items()
      }
    return data_structures.to_immutable_dict(new_values)
