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


# TODO(tycai): Accept state=True.
# TODO(tycai): Make sure transformed functions have better names.
class LiftingModule(module.Module):
  """Lifts the given init function to a function in the current Haiku namespace.

  During init, the returned callable will run the given `init_fn`, and include
  the resulting params in the outer transform's dictionaries.
  During apply, the returned callable will instead pull the relevant parameters
  from the outer transform's dictionaries.

  Must be called inside hk.transform, and be passed the `init` member of a
  `hk.Transformed`.

  The user must ensure that the given `init` does not accidentally catch modules
  from an outer `hk.transform` via functional closure.

  This is highly experimental and may be changed or removed at any time.
  """

  def __init__(self, init_fn, name="lifted"):
    """Initializes the LiftingModule.

    Args:
      init_fn: The init_fn from a hk.Transformed. Must not be stateful.
      name: Module name.
    """
    super().__init__(name=name)
    self._init_fn = init_fn

  def __call__(self, *args, **kwargs):
    frame = base.current_frame()
    bundle_name = self.module_name
    if base.params_frozen():
      prefix = bundle_name + "/"
      lifted_params = unpack_from_dict(frame.params, prefix)
      return lifted_params
    else:  # Inside init.
      # Lift parameters into this transform's params_dict.
      params = self._init_fn(*args, **kwargs)
      pack_into_dict(params, frame.params, bundle_name)
      return params

lift = LiftingModule  # pylint: disable=invalid-name
