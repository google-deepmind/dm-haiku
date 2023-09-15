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

"""Utility functions for working with Haiku and Flax code."""

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, Union

from haiku._src import typing


# If you are forking replace this with `import haiku as hk`.
# pylint: disable=invalid-name
class hk:
  Params = typing.Params
  State = typing.State
# pylint: enable=invalid-name
del typing


FlaxCollection = Mapping[str, Any]
MutableFlaxCollection = MutableMapping[str, Any]
FlaxVariables = Mapping[str, FlaxCollection]
MutableFlaxVariables = MutableMapping[str, MutableFlaxCollection]
HaikuParamsOrState = Union[hk.Params, hk.State]
MutableHaikuParamsOrState = MutableMapping[str, MutableMapping[str, Any]]


def flatten_flax_to_haiku(collection: FlaxCollection) -> HaikuParamsOrState:
  """Flattens a Flax variable collection (e.g. params) to a Haiku dict."""
  out = {}
  for name, value in collection.items():
    if not isinstance(value, Mapping):
      if '~' not in out:
        out['~'] = {}
      out['~'][name] = value
    else:
      _flatten_flax_to_haiku_inner(value, out, (name,))
  return out


def _flatten_flax_to_haiku_inner(
    collection: FlaxCollection,
    out: MutableHaikuParamsOrState,
    prefix: Sequence[str],
) -> HaikuParamsOrState:
  """Recursive inner loop of `flatten_flax_to_haiku`."""
  for name, value in collection.items():
    if isinstance(value, Mapping):
      _flatten_flax_to_haiku_inner(value, out=out, prefix=(*prefix, name))
    else:
      assert prefix
      mod_name = '/'.join(prefix)
      if mod_name not in out:
        out[mod_name] = {}
      out[mod_name][name] = value
  return out
