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
"""Functions for filtering parameters and state in Haiku."""

import collections
from typing import Callable, Text, Tuple, TypeVar

from haiku._src import data_structures
from haiku._src.typing import Params, State  # pylint: disable=g-multiple-import
import jax.numpy as jnp

Predicate = Callable[[Text, Text, jnp.ndarray], bool]
T = TypeVar("T", Params, State)


def partition(predicate: Predicate, structure: T) -> Tuple[T, T]:
  """Partitions the input structure in two according to a given predicate.

  For a given set of parameters, you can use :func:`partition` to split them:

  >>> params = {'linear': {'w': None, 'b': None}}
  >>> predicate = lambda module_name, name, value: name == 'w'
  >>> weights, biases = hk.data_structures.partition(predicate, params)
  >>> weights
  frozendict({'linear': frozendict({'w': None})})
  >>> biases
  frozendict({'linear': frozendict({'b': None})})

  Note: returns new structures not a view.

  Args:
    predicate: criterion to be used to partition the input data.
      The `predicate` argument is expected to be a boolean function taking as
      inputs the name of the module, the name of a given entry in the module
      data bundle (e.g. parameter name) and the corresponding data.
    structure: Haiku params or state data structure to be partitioned.

  Returns:
    A tuple containing all the params or state as partitioned by the input
      predicate. Entries matching the predicate will be in the first structure,
      and the rest will be in the second.
  """
  true = collections.defaultdict(dict)
  false = collections.defaultdict(dict)

  for module_name, bundle in structure.items():
    for name, value in bundle.items():
      out = true if predicate(module_name, name, value) else false
      out[module_name][name] = value

  true = data_structures.to_immutable_dict(true)
  false = data_structures.to_immutable_dict(false)

  return true, false


def filter(predicate: Predicate, structure: T) -> T:  # pylint: disable=redefined-builtin
  """Filters a input structure according to a user specified predicate.

  >>> params = {'linear': {'w': None, 'b': None}}
  >>> predicate = lambda module_name, name, value: name == 'w'
  >>> hk.data_structures.filter(predicate, params)
  frozendict({'linear': frozendict({'w': None})})

  Note: returns a new structure not a view.

  Args:
    predicate: criterion to be used to partition the input data.
      The `predicate` argument is expected to be a boolean function taking as
      inputs the name of the module, the name of a given entry in the module
      data bundle (e.g. parameter name) and the corresponding data.
    structure: Haiku params or state data structure to be filtered.

  Returns:
    All the input parameters or state as selected by the input predicate.
  """
  out = collections.defaultdict(dict)

  for module_name, bundle in structure.items():
    for name, value in bundle.items():
      if predicate(module_name, name, value):
        out[module_name][name] = value

  return data_structures.to_immutable_dict(out)


def merge(*structures: T) -> T:
  """Merges multiple input structures.

  >>> weights = {'linear': {'w': None}}
  >>> biases = {'linear': {'b': None}}
  >>> hk.data_structures.merge(weights, biases)
  frozendict({'linear': frozendict({'b': None, 'w': None})})

  When structures are not disjoint the output will contain the value from the
  last structure for each path:

  >>> weights1 = {'linear': {'w': 1}}
  >>> weights2 = {'linear': {'w': 2}}
  >>> hk.data_structures.merge(weights1, weights2)
  frozendict({'linear': frozendict({'w': 2})})

  Note: returns a new structure not a view.

  Args:
    *structures: One or more structures to merge.

  Returns:
    A single structure with an entry for each path in the input structures.
  """
  out = collections.defaultdict(dict)
  for structure in structures:
    for module_name, bundle in structure.items():
      for name, value in bundle.items():
        out[module_name][name] = value
  return data_structures.to_immutable_dict(out)
