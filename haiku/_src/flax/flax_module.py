# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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

"""Utilities for converting Haiku modules to Flax modules."""

from typing import TypeVar

import flax.core
import flax.linen as nn
from haiku._src import filtering
from haiku._src import transform as transform_lib
from haiku._src import typing
from haiku._src.flax import utils


# pylint: disable=invalid-name
# If you are forking replace this with `import haiku as hk`.
class hk:
  Params = typing.Params
  State = typing.State
  SupportsCall = typing.SupportsCall
  transform = transform_lib.transform
  transform_with_state = transform_lib.transform_with_state
  Transformed = transform_lib.Transformed
  TransformedWithState = transform_lib.TransformedWithState
  with_empty_state = transform_lib.with_empty_state


# If you are forking replace this with `import haiku.data_structures as hkds`.
class hkds:
  traverse = filtering.traverse
# pylint: enable=invalid-name
del filtering, transform_lib, typing

T = TypeVar('T')

FlaxCollection = utils.FlaxCollection
FlaxVariables = utils.FlaxVariables
HaikuParamsOrState = utils.HaikuParamsOrState
MutableHaikuParamsOrState = utils.MutableHaikuParamsOrState


def store_haiku_collections(
    scope: flax.core.Scope,
    **collections: HaikuParamsOrState,
):
  """Stores the given Haiku params/state in nested collections under scope."""
  for collection_name, collection in collections.items():
    for mod_name, name, value in hkds.traverse(collection):
      subscope = scope
      for part in mod_name.split('/'):
        subscope = subscope.push(part, reuse=True)
      subscope.put_variable(collection_name, name, value)


def only_changed_state(old_state: hk.State, new_state: hk.State) -> hk.State:
  """Returns the subset of new_state that has changed from old_state."""
  updated_state = {}
  for mod_name, name, value in hkds.traverse(new_state):
    if mod_name in old_state and name in old_state[mod_name]:
      if old_state[mod_name][name] is not value:
        if mod_name not in updated_state:
          updated_state[mod_name] = {}
        updated_state[mod_name][name] = value
  return updated_state


class Module(nn.Module):
  """A Flax ``nn.Module`` that runs a Haiku transformed function.

  This type is designed to make it easy to take a Haiku transformed function
  and/or a Haiku module and use it inside a program that otherwise uses Flax.

  Given a Haiku transformed function

  >>> def f(x):
  ...   return hk.Linear(1)(x)
  >>> f = hk.transform(f)

  You can convert it into a Flax module using:

  >>> mod = hk.experimental.flax.Module(f)

  Calling this module is the same as calling any regular Flax module:

  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.ones([1, 1])
  >>> variables = mod.init(rng, x)
  >>> out = mod.apply(variables, x)

  If you just want to convert a Haiku module class such that it can be used
  with Flax you can use the ``create`` class method:

  >>> mod = hk.experimental.flax.Module.create(hk.Linear, 1)
  >>> variables = mod.init(rng, x)
  >>> out = mod.apply(variables, x)
  """

  transformed: hk.Transformed | hk.TransformedWithState

  def __post_init__(self):
    super().__post_init__()
    if isinstance(self.transformed, hk.Transformed):
      self.transformed = hk.with_empty_state(self.transformed)

  @classmethod
  def create(
      cls, hk_cls: type[hk.SupportsCall], *init_args, **init_kwargs
  ) -> 'Module':
    """Converts a given Haiku module into a Flax ``nn.Module``.

    TODO(tomhennigan): Support multiple forward methods.

    Example usage:

    >>> mod = hk.experimental.flax.Module.create(hk.Linear, 1)  # hk.Linear(1)
    >>> rng = jax.random.PRNGKey(42)
    >>> x = jnp.ones([1, 1])
    >>> variables = mod.init(rng, x)
    >>> out = mod.apply(variables, x)

    For a stateful module like resnet, you need to also handle output state:

    >>> mod = hk.experimental.flax.Module.create(hk.nets.ResNet50, 10)
    >>> rng = jax.random.PRNGKey(42)
    >>> x = jnp.ones([1, 224, 224, 3])
    >>> variables = mod.init(rng, x, is_training=True)
    >>> out, state_out = mod.apply(variables, x, is_training=True,
    ...                            mutable=['state'])

    Args:
      hk_cls: A Haiku module type (e.g. ``hk.Linear``).
      *init_args: Positional arguments for the constructor.
      **init_kwargs: Keyword arguments for the constructor.

    Returns:
      A Flax ``nn.Module`` wrapping the given class.
    """

    def fn(*args, **kwargs):
      mod = hk_cls(*init_args, **init_kwargs)
      return mod(*args, **kwargs)

    fn = hk.transform_with_state(fn)
    return Module(fn)

  @nn.compact
  def __call__(self, *args, **kwargs):
    if self.is_initializing():
      rng = self.make_rng('params')
      params, state = self.transformed.init(rng, *args, **kwargs)
      store_haiku_collections(self.scope, params=params, state=state)
    else:
      params = utils.flatten_flax_to_haiku(self.variables.get('params', {}))
      state = utils.flatten_flax_to_haiku(self.variables.get('state', {}))

    rng = self.make_rng('apply') if self.has_rng('apply') else None
    out, state_out = self.transformed.apply(params, state, rng, *args, **kwargs)
    if not self.is_initializing():
      updated_state = only_changed_state(state, state_out)
      store_haiku_collections(self.scope, state=updated_state)

    return out
