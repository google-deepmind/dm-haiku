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
from typing import Any, Callable, Generator, Mapping, Tuple, TypeVar

from haiku._src import data_structures
from haiku._src.typing import Params, State  # pylint: disable=g-multiple-import
import jax.numpy as jnp

T = TypeVar("T", Params, State)
InT = TypeVar("InT")
OutT = TypeVar("OutT")


def traverse(structure: T) -> Generator[Tuple[str, str, Any], None, None]:
  """Iterates over a structure yielding module names, names and values.

  NOTE: Items are iterated in key sorted order.

  Args:
    structure: The structure to traverse.

  Yields:
    Tuples of the module name, name and value from the given structure.
  """
  for module_name in sorted(structure):
    bundle = structure[module_name]
    for name in sorted(bundle):
      value = bundle[name]
      yield module_name, name, value


def partition(
    predicate: Callable[[str, str, jnp.ndarray], bool],
    structure: T,
) -> Tuple[T, T]:
  """Partitions the input structure in two according to a given predicate.

  For a given set of parameters, you can use :func:`partition` to split them:

  >>> params = {'linear': {'w': None, 'b': None}}
  >>> predicate = lambda module_name, name, value: name == 'w'
  >>> weights, biases = hk.data_structures.partition(predicate, params)
  >>> weights
  FlatMapping({'linear': FlatMapping({'w': None})})
  >>> biases
  FlatMapping({'linear': FlatMapping({'b': None})})

  Note: returns new structures not a view.

  Args:
    predicate: criterion to be used to partition the input data.
      The ``predicate`` argument is expected to be a boolean function taking as
      inputs the name of the module, the name of a given entry in the module
      data bundle (e.g. parameter name) and the corresponding data.
    structure: Haiku params or state data structure to be partitioned.

  Returns:
    A tuple containing all the params or state as partitioned by the input
      predicate. Entries matching the predicate will be in the first structure,
      and the rest will be in the second.
  """
  f = lambda m, n, v: int(not predicate(m, n, v))
  return partition_n(f, structure, 2)


def partition_n(
    fn: Callable[[str, str, jnp.ndarray], int],
    structure: T,
    n: int,
) -> Tuple[T, ...]:
  """Partitions a structure into `n` structures.

  For a given set of parameters, you can use :func:`partition_n` to split them
  into ``n`` groups. For example, to split your parameters/gradients by module
  name:

  >>> def partition_by_module(structure):
  ...   cnt = itertools.count()
  ...   d = collections.defaultdict(lambda: next(cnt))
  ...   fn = lambda m, n, v: d[m]
  ...   return hk.data_structures.partition_n(fn, structure, len(structure))

  >>> structure = {f'layer_{i}': {'w': None, 'b': None} for i in range(3)}
  >>> for substructure in partition_by_module(structure):
  ...   print(substructure)
  FlatMapping({'layer_0': FlatMapping({'b': None, 'w': None})})
  FlatMapping({'layer_1': FlatMapping({'b': None, 'w': None})})
  FlatMapping({'layer_2': FlatMapping({'b': None, 'w': None})})

  Args:
    fn: Callable returning which bucket in ``[0, n)`` the given element should
      be output.
    structure: Haiku params or state data structure to be partitioned.
    n: The total number of buckets.

  Returns:
    A tuple of size ``n``, where each element will contain the values for which
    the function returned the current index.
  """
  out = [collections.defaultdict(dict) for _ in range(n)]
  for module_name, name, value in traverse(structure):
    i = fn(module_name, name, value)
    assert 0 <= i < n, f"{i} must be in range [0, {n})"
    out[i][module_name][name] = value
  return tuple(data_structures.to_haiku_dict(o) for o in  out)


def filter(  # pylint: disable=redefined-builtin
    predicate: Callable[[str, str, jnp.ndarray], bool],
    structure: T,
) -> T:
  """Filters an input structure according to a user specified predicate.

  >>> params = hk.data_structures.to_immutable_dict(
  ...   {'linear': {'w': None, 'b': None}})
  >>> predicate = lambda module_name, name, value: name == 'w'
  >>> hk.data_structures.filter(predicate, params)
  FlatMapping({'linear': FlatMapping({'w': None})})

  Note: returns a new structure not a view.

  Args:
    predicate: criterion to be used to partition the input data.
      The ``predicate`` argument is expected to be a boolean function taking as
      inputs the name of the module, the name of a given entry in the module
      data bundle (e.g. parameter name) and the corresponding data.
    structure: Haiku params or state data structure to be filtered.

  Returns:
    All the input parameters or state as selected by the input predicate.
  """
  out = collections.defaultdict(dict)

  for module_name, name, value in traverse(structure):
    if predicate(module_name, name, value):
      out[module_name][name] = value

  return data_structures.to_haiku_dict(out)


def map(  # pylint: disable=redefined-builtin
    fn: Callable[[str, str, InT], OutT],
    structure: Mapping[str, Mapping[str, InT]],
) -> Mapping[str, Mapping[str, OutT]]:
  """Maps a function to an input structure accordingly.

  >>> params = hk.data_structures.to_immutable_dict(
  ...   {'linear': {'w': 1.0, 'b': 2.0}})
  >>> fn = lambda module_name, name, value: 3 * value if name == 'w' else value
  >>> hk.data_structures.map(fn, params)
  FlatMapping({'linear': FlatMapping({'b': 2.0, 'w': 3.0})})

  Note: returns a new structure not a view.

  Args:
    fn: criterion to be used to map the input data.
      The ``fn`` argument is expected to be a boolean function taking as
      inputs the name of the module, the name of a given entry in the module
      data bundle (e.g. parameter name) and the corresponding data.
    structure: Haiku params or state data structure to be mapped.

  Returns:
    All the input parameters or state as mapped by the input fn.
  """
  out = collections.defaultdict(dict)
  for module_name, name, value in traverse(structure):
    out[module_name][name] = fn(module_name, name, value)
  return data_structures.to_haiku_dict(out)


def merge(*structures: T) -> T:
  """Merges multiple input structures.

  >>> weights = hk.data_structures.to_immutable_dict({'linear': {'w': None}})
  >>> biases = hk.data_structures.to_immutable_dict({'linear': {'b': None}})
  >>> hk.data_structures.merge(weights, biases)
  FlatMapping({'linear': FlatMapping({'w': None, 'b': None})})

  When structures are not disjoint the output will contain the value from the
  last structure for each path:

  >>> weights1 = hk.data_structures.to_immutable_dict({'linear': {'w': 1}})
  >>> weights2 = hk.data_structures.to_immutable_dict({'linear': {'w': 2}})
  >>> hk.data_structures.merge(weights1, weights2)
  FlatMapping({'linear': FlatMapping({'w': 2})})

  Note: returns a new structure not a view.

  Args:
    *structures: One or more structures to merge.

  Returns:
    A single structure with an entry for each path in the input structures.
  """
  out = collections.defaultdict(dict)
  for structure in structures:
    for module_name, name, value in traverse(structure):
      out[module_name][name] = value
  return data_structures.to_haiku_dict(out)
