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
"""Utilities for checkpoints."""

from typing import Any, Mapping

import haiku as hk
from haiku._src import utils
from haiku._src.integration import descriptors
import jax
import jax.numpy as jnp


def format_tensor(tensor: jnp.ndarray) -> str:
  shape = list(tensor.shape)
  dtype = utils.simple_dtype(tensor.dtype)
  return f"{dtype}{shape}"


def module_name(d: descriptors.ModuleDescriptor):
  name = hk.testing.transform_and_run(
      lambda: str(descriptors.unwrap(d.create())))()
  return name.split("\n")


def summarize(d: descriptors.ModuleDescriptor) -> Mapping[str, Any]:
  """Generates a summary of the given descriptor."""
  f = hk.transform_with_state(lambda x: d.create()(x))  # pylint: disable=unnecessary-lambda
  x = jnp.ones(d.shape, d.dtype)
  rng = jax.random.PRNGKey(42)
  params, state = map(hk.data_structures.to_mutable_dict,
                      jax.eval_shape(f.init, rng, x))
  out = {"module": module_name(d), "input": format_tensor(x)}
  if params:
    out["param_size"] = int(hk.data_structures.tree_size(params))
    out["param_bytes"] = int(hk.data_structures.tree_bytes(params))
    out["params"] = jax.tree_map(format_tensor, params)
  if state:
    out["state_size"] = int(hk.data_structures.tree_size(state))
    out["state_bytes"] = int(hk.data_structures.tree_bytes(state))
    out["state"] = jax.tree_map(format_tensor, state)
  return out
