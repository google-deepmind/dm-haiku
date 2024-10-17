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

"""Utilities for converting Flax modules to use with Haiku."""

from collections.abc import Callable, Mapping
from typing import Any

import flax.errors
import flax.linen as nn
from haiku._src import base
from haiku._src import filtering
from haiku._src import lift as lift_lib
from haiku._src import transform
from haiku._src import typing
from haiku._src.flax import utils


# pylint: disable=invalid-name
# If you are forking replace this with `import haiku as hk`.
class hk:
  lift_with_state = lift_lib.lift_with_state
  maybe_next_rng_key = base.maybe_next_rng_key
  Params = typing.Params
  running_init = transform.running_init
  State = typing.State
  TransformedWithState = transform.TransformedWithState


# If you are forking replace this with `import haiku.data_structures as hkds`.
class hkds:
  traverse = filtering.traverse
# pylint: enable=invalid-name
del typing, filtering, lift_lib


FlaxCollection = utils.FlaxCollection
FlaxVariables = utils.FlaxVariables
MutableFlaxVariables = utils.MutableFlaxVariables


def _from_haiku_params(params: hk.Params) -> FlaxCollection:
  """Converts Haiku parameters to a nested Flax collection."""
  collection = {}
  for mod_name, name, value in hkds.traverse(params):
    if mod_name == '~':
      collection[name] = value
    else:
      nested_collection = collection
      for part in mod_name.split('/'):
        if part not in nested_collection:
          nested_collection[part] = {}
        nested_collection = nested_collection[part]
      nested_collection[name] = value
  return collection


def _to_haiku_state(variables: FlaxVariables) -> hk.State:
  """Converts a nested Flax collection to a Haiku state dict."""
  state = {}
  for collection_name, collection in variables.items():
    flat_collection = utils.flatten_flax_to_haiku(collection)
    for mod_name, name, value in hkds.traverse(flat_collection):
      mod_name = f'{collection_name}/{mod_name}'
      if mod_name not in state:
        state[mod_name] = {}
      state[mod_name][name] = value
  return state


def _from_haiku_state(state: hk.State) -> MutableFlaxVariables:
  """Converts a Haiku state dict to a nested Flax collection."""
  variables = {}
  for name, module_state in state.items():
    collection_name, *mod_name, name = name.split('/')
    if collection_name not in variables:
      variables[collection_name] = {}
    if not mod_name and name == '~':
      variables[collection_name].update(module_state)
    else:
      nested_collection = variables[collection_name]
      for part in mod_name:
        if part not in nested_collection:
          nested_collection[part] = {}
        nested_collection = nested_collection[part]
      nested_collection[name] = dict(module_state)
  return variables


def _flax_transform_with_state(mod: nn.Module) -> hk.TransformedWithState:
  """Transforms a Flax ``nn.Module`` into a Haiku transformed function.

  Example usage:

  >>> mod = nn.Dense(10)
  >>> f = _flax_transform_with_state(mod)
  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.ones([1, 1])
  >>> params, state = f.init(rng, x)
  >>> out, state = f.apply(params, state, rng, x)

  Args:
    mod: Any Flax ``nn.Module`` instance.

  Returns:
    A :class:`~haiku.TransformedWithState` instance (equivalent to the result of
    calling :func:`transform_with_state`).

  See also:
    lift: Use a Flax module as part of an outer :func:`~haiku.transform` or
      :func:`~haiku.transform_with_state`.
  """

  def init_fn(rng, *args, **kwargs):
    assert 'rngs' not in kwargs  # Handled in `lift`.
    variables = dict(mod.init(rng, *args, **kwargs))
    params = utils.flatten_flax_to_haiku(variables.pop('params', {}))
    state = _to_haiku_state(variables)
    return params, state

  def apply_fn(params, state, rng, *args, **kwargs):
    if rng is not None:
      raise ValueError(
          'RNGs passed in apply must be passed in the rngs keyword argument'
      )

    variables = _from_haiku_state(state)
    mutable = set(variables)
    variables['params'] = _from_haiku_params(params)
    out, variables = mod.apply(variables, *args, **kwargs, mutable=mutable)
    state = _to_haiku_state(variables)
    return out, state

  return hk.TransformedWithState(init_fn, apply_fn)


def lift(
    mod: nn.Module,
    *,
    name: str,
) -> Callable[..., Any]:
  """Lifts a flax nn.Module into a Haiku transformed function.

  For a Flax Module (e.g. ``mod = nn.Dense(10)``), ``mod = lift(mod)`` allows
  you to run the call method of the module as if the module was a regular Haiku
  module.

  Parameters and state from the Flax module are registered with Haiku and become
  part of the params/state dictionaries (as returned from ``init``/``apply``).

  >>> def f(x):
  ...   # Create and "lift" a Flax module.
  ...   mod = hk.experimental.flax.lift(nn.Dense(300), name='dense')
  ...   x = mod(x)                  # Any params/state will be registered
  ...                               # with Haiku when applying the module.
  ...   x = jax.nn.relu(x)
  ...   x = hk.nets.MLP([100, 10])  # You can of course mix Haiku modules in.
  ...   return x
  >>> f = hk.transform(f)
  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.ones([1, 1])
  >>> params = f.init(rng, x)
  >>> out = f.apply(params, None, x)

  Args:
    mod: Any Flax ``nn.Module`` instance.
    name: Name scope to prefix entries in the outer params/state dict.

  Returns:
    A function that when applied calls the call method of the given Flax module
    and returns its output. As a side effect of calling the module any module
    parameters and state variables are registered with Haiku.
  """
  mod = _flax_transform_with_state(mod)
  init_fn, updater = hk.lift_with_state(mod.init, name=name, allow_reuse=True)
  updater._used = True  # pylint: disable=protected-access

  def wrapped(*args, **kwargs):
    init_kwargs = dict(kwargs)
    if hk.running_init():
      if 'rngs' not in init_kwargs:
        init_rng = hk.maybe_next_rng_key()
      elif isinstance(init_kwargs['rngs'], Mapping):
        init_rng = dict(init_kwargs.pop('rngs'))
        if 'params' not in init_rng:
          rng = hk.maybe_next_rng_key()
          if rng is not None:
            init_rng['params'] = rng
      else:
        raise flax.errors.InvalidRngError(
            'rngs should be a dictionary mapping strings to `jax.PRNGKey`.'
        )
    else:
      init_rng = None
    params, state = init_fn(init_rng, *args, **init_kwargs)
    out, state = mod.apply(params, state, None, *args, **kwargs)
    updater._used = False  # pylint: disable=protected-access
    updater.update(state)
    return out

  return wrapped
