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
import types
from typing import Optional
import warnings

from haiku._src import base
from haiku._src import data_structures
from haiku._src import initializers
from haiku._src import module
import jax
import jax.numpy as jnp

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.get_state = base.get_state
hk.set_state = base.set_state
hk.Module = module.Module
hk.data_structures = data_structures
hk.initializers = initializers
del base, data_structures, module, initializers


class ExponentialMovingAverage(hk.Module):
  """Maintains an exponential moving average.

  This uses the Adam debiasing procedure.
  See https://arxiv.org/pdf/1412.6980.pdf for details.
  """

  def __init__(
      self,
      decay,
      zero_debias: bool = True,
      warmup_length: int = 0,
      name: Optional[str] = None,
  ):
    """Initializes an ExponentialMovingAverage module.

    Args:
      decay: The chosen decay. Must in ``[0, 1)``. Values close to 1 result in
        slow decay; values close to ``0`` result in fast decay.
      zero_debias: Whether to run with zero-debiasing.
      warmup_length: A positive integer, EMA has no effect until
        the internal counter has reached `warmup_length` at which point the
        initial value for the decaying average is initialized to the input value
        after `warmup_length` iterations.
      name: The name of the module.
    """
    super().__init__(name=name)
    self.decay = decay
    self.warmup_length = warmup_length
    self.zero_debias = zero_debias

    if warmup_length < 0:
      raise ValueError(
          f"`warmup_length` is {warmup_length}, but should be non-negative.")

    if warmup_length and zero_debias:
      raise ValueError(
          "Zero debiasing does not make sense when warming up the value of the "
          "average to an initial value. Set zero_debias=False if setting "
          "warmup_length to a non-zero value.")

  def initialize(self, shape, dtype=jnp.float32):
    """If uninitialized sets the average to ``zeros`` of the given shape/dtype."""
    if hasattr(shape, "shape"):
      warnings.warn("Passing a value into initialize instead of a shape/dtype "
                    "is deprecated. Update your code to use: "
                    "`ema.initialize(v.shape, v.dtype)`.",
                    category=DeprecationWarning)
      shape, dtype = shape.shape, shape.dtype

    hk.get_state("hidden", shape, dtype, init=jnp.zeros)
    hk.get_state("average", shape, dtype, init=jnp.zeros)

  def __call__(
      self,
      value: jnp.ndarray,
      update_stats: bool = True,
  ) -> jnp.ndarray:
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

    counter = hk.get_state("counter", (), jnp.int32,
                           init=hk.initializers.Constant(-self.warmup_length))
    counter = counter + 1

    decay = jax.lax.convert_element_type(self.decay, value.dtype)
    if self.warmup_length > 0:
      decay = jax.lax.select(counter <= 0, 0.0, decay)

    one = jnp.ones([], value.dtype)
    hidden = hk.get_state("hidden", value.shape, value.dtype, init=jnp.zeros)
    hidden = hidden * decay + value * (one - decay)

    average = hidden
    if self.zero_debias:
      average /= (one - jnp.power(decay, counter))

    if update_stats:
      hk.set_state("counter", counter)
      hk.set_state("hidden", hidden)
      hk.set_state("average", average)

    return average

  @property
  def average(self):
    return hk.get_state("average")


class EMAParamsTree(hk.Module):
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

  def __init__(
      self,
      decay,
      zero_debias: bool = True,
      warmup_length: int = 0,
      ignore_regex: str = "",
      name: Optional[str] = None,
  ):
    """Initializes an EMAParamsTree module.

    Args:
      decay: The chosen decay. Must in ``[0, 1)``. Values close to ``1`` result
        in slow decay; values close to ``0`` result in fast decay.
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
    super().__init__(name=name)
    self.decay = decay
    self.zero_debias = zero_debias
    self.warmup_length = warmup_length
    self.ignore_regex = ignore_regex

  def __call__(self, tree, update_stats=True):
    def maybe_ema(k, v):
      if self.ignore_regex and re.match(self.ignore_regex, k):
        return v
      else:
        ema_name = k.replace("/", "__").replace("~", "_tilde_")
        return ExponentialMovingAverage(
            self.decay, self.zero_debias, self.warmup_length, name=ema_name)(
                v, update_stats=update_stats)

    # We want to potentially replace params with EMA'd versions.
    new_values = {}
    for module_name, param_dict in tree.items():
      new_values[module_name] = {
          k: maybe_ema("/".join([module_name, k]), v)
          for k, v in param_dict.items()
      }
    return hk.data_structures.to_haiku_dict(new_values)
