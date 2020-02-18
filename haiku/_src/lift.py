# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Lifting parameters in Haiku."""

from haiku._src import base
from haiku._src import data_structures
from haiku._src import module
import jax.numpy as jnp

_SENTINEL_NAME = "sentinel"


def pack_into_dict(src, dst, prefix):
  """Puts items from src into dst, with an added prefix."""
  for key, value in src.items():
    new_key = f"{prefix}/{key}"
    assert new_key not in dst
    dst[new_key] = value


def unpack_from_dict(src, prefix):
  """Returns pairs from src where key begins with prefix, cutting off prefix."""
  result = dict()
  for key, value in src.items():
    if key.startswith(prefix):
      result[key[len(prefix):]] = value
  return data_structures.to_immutable_dict(result)


# TODO(tycai): Allow state=False as well.
# TODO(tycai): Make sure transformed functions have better names.
class LiftingModule(module.Module):
  """Lifts the given init function to a function in the current Haiku namespace.

  During init, the returned callable will run the given `init_fn`, and include
  the resulting params/state in the outer transform's dictionaries.
  During apply, the returned callable will instead pull the relevant parameters
  and state from the outer transform's dictionaries.

  Must be called inside hk.transform, and be passed the `init` member of a
  `hk.Transformed`.

  Currently, the given `init_fn` must not use state.
  """

  def __init__(self, init_fn, name=None):
    """Initializes the LiftingModule.

    Args:
      init_fn: The init_fn from a hk.Transformed. Requires state=True.
      name: Module name.
    """
    if name is None:
      name = f"lifted_{init_fn.__name__}"
    super(LiftingModule, self).__init__(name=name)
    self._init_fn = init_fn

  def __call__(self, *args, **kwargs):
    frame = base.current_frame()
    bundle_name = self.module_name
    if _SENTINEL_NAME in frame.params[bundle_name]:
      prefix = bundle_name + "/"
      lifted_params = unpack_from_dict(frame.params, prefix)
      lifted_state = unpack_from_dict(frame.state, prefix)
      return lifted_params, lifted_state
    else:
      # Ensure sentinel is set for apply.
      base.get_parameter(_SENTINEL_NAME, (), init=jnp.zeros)
      # Lift parameters into this transform's params_dict.
      params, state = self._init_fn(*args, **kwargs)
      pack_into_dict(params, frame.params, bundle_name)
      pack_into_dict(state, frame.state, bundle_name)
      return params, state

lift = LiftingModule  # pylint: disable=invalid-name
