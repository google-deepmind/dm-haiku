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
"""Data structures used by Haiku."""

import collections
import contextlib
import pprint
import threading
from typing import Any, Callable, Generic, Mapping, Optional, TypeVar, Tuple, Union

from haiku._src import utils
import jax

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
U = TypeVar("U")


class Stack(Generic[T]):
  """Stack supporting push/pop/peek."""

  def __init__(self):
    self._storage = collections.deque()

  def __len__(self):
    return len(self._storage)

  def __iter__(self):
    return iter(reversed(self._storage))

  def clone(self):
    return self.map(lambda v: v)

  def map(self, fn: Callable[[T], U]) -> "Stack[U]":
    s = type(self)()
    for item in self._storage:
      s.push(fn(item))
    return s

  def push(self, elem: T):
    self._storage.append(elem)

  def popleft(self) -> T:
    return self._storage.popleft()

  def pop(self) -> T:
    return self._storage.pop()

  def peek(self, depth=-1) -> T:
    return self._storage[depth]

  @contextlib.contextmanager
  def __call__(self, elem):
    self.push(elem)
    try:
      yield
    finally:
      assert self.pop() is elem


class ThreadLocalStack(Stack[T], threading.local):
  """Thread-local stack."""


# TODO(tomhennigan) Use types.MappingProxyType when we are Python 3 only.
# TODO(lenamartens) Deprecate type
class frozendict(Mapping[K, V]):  # pylint: disable=invalid-name
  """Immutable mapping from keys to values."""

  def __init__(self, *args, **kwargs):
    self._storage = dict(*args, **kwargs)
    self._keys = tuple(sorted(self._storage))
    # Dict values aren't necessarily hashable so we just use the keys.
    self._hash = hash(self._keys)

  def __iter__(self):
    return iter(self._keys)

  def __len__(self):
    return len(self._storage)

  def __getattr__(self, key: K) -> V:
    # NOTE: Strictly speaking this is not part of the dict API, but it is quite
    # convenient to be able to do `params.w` rather than `params['w']`.
    try:
      return self._storage[key]
    except KeyError as e:
      raise AttributeError(e)

  def get(self, key: K, default: Optional[T] = None) -> Union[V, Optional[T]]:
    return self._storage.get(key, default)

  def __getitem__(self, key: K) -> V:
    return self._storage[key]

  def __repr__(self):
    single_line = "{}({{{}}})".format(
        type(self).__name__,
        ", ".join("{!r}: {!r}".format(k, self._storage[k]) for k in self._keys))
    if len(single_line) <= 80:
      return single_line

    return "{}({{\n{},\n}})".format(
        type(self).__name__,
        utils.indent(
            2, ",\n".join(_repr_item(k, self._storage[k]) for k in self._keys)))

  __str__ = __repr__

  def __ne__(self, other):
    # Defined for Python 2 support.
    # https://docs.python.org/2/reference/datamodel.html#object.__ne__
    return not self.__eq__(other)

  def __eq__(self, other):
    if isinstance(other, frozendict):
      return self._storage == other._storage  # pylint: disable=protected-access
    elif isinstance(other, dict):
      # dict is not generally hashable so this comparison is safe.
      return self._storage == other
    else:
      return False

  def __hash__(self):
    return self._hash

  def __reduce__(self):
    return (frozendict, (self._storage,))


def to_immutable_dict(mapping: Mapping[K, V]) -> Mapping[K, V]:
  return _to_immutable(mapping)


def _to_immutable(o: T) -> T:
  if isinstance(o, Mapping) and not isinstance(o, frozendict):
    return frozendict({k: to_immutable_dict(v) for k, v in o.items()})
  return o


# TODO(tomhennigan) Better types here (Mapping[K, V]) -> MutableMapping[K, V]?
def to_mutable_dict(mapping):
  """Turns an immutable frozendict into a mutable dict."""
  return {k: (to_mutable_dict(v) if isinstance(v, frozendict) else v)
          for k, v in mapping.items()}


def _repr_item(k, v):
  k = repr(k) + ": "
  v = pprint.pformat(v)
  return k + utils.indent(len(k), v).lstrip()


jax.tree_util.register_pytree_node(
    frozendict,
    lambda s: (tuple(s.values()), tuple(s.keys())),
    lambda k, xs: frozendict(zip(k, xs)))

# TODO(lenamartens) replace with recursive types when supported (b/109648354)
# Structure = Tuple[Union[Tuple[K], Tuple[K, "Structure"]]]
Structure = Tuple[Tuple[K, Tuple]]
# Index = Mapping[K, Tuple[int, int, Union[Tuple[()], "Index"]]]
Index = Mapping[K, Tuple[int, int, Mapping]]
Leaves = Tuple[Any]
Flat = Tuple[Leaves, Tuple[Structure, Index]]


class FlatMapping(collections.Mapping):
  """Immutable mapping with O(1) flatten and unflatten operation."""

  def __init__(self, flat: Flat):
    """Constructs a flat mapping from a Flat structure.

    Args:
      flat: A tuple containing a flat sequence of values and a tuple
      representing the desired nested structure of the mapping.

    A Flat structure maps to the internal representation of a FlatMapping and is
    the output when calling `flatten` on a FlatMapping.

    This is useful to implement an O(N) map_structure:
    >>> a = FlatMapping.from_mapping("foo": "bar", "baz": {"bat": "qux"})
    >>> leaves, structure = a.flatten()        # O(1)
    >>> leaves = map(lambda x: x + 1, leaves)  # O(N)
    >>> d = FlatMapping((leaves, structure))   # O(1)
    """
    self._leaves, (self._structure, self._index) = flat

  @classmethod
  def from_mapping(cls, m: Mapping[Any, Any]) -> "FlatMapping":
    """Construct a new FlatMapping object from a Mapping.

    Args:
      m: If m is a FlatMapping, the internal structures from m are reused
         to construct a new FlatMapping (this is O(1))
         If m is any other Mapping, m will be deconstructed into
         its Flat representation (this is O(n logn))
    Returns:
      A FlatMapping with the keys and values of the input mapping.
    """
    if isinstance(m, FlatMapping):
      return cls(m.flatten())
    return cls(deconstruct(m))

  def keys(self):
    return [node[0] for node in self._structure]

  def values(self):
    return [self[key] for key in self.keys()]

  def items(self):
    return [(node[0], self[node[0]]) for node in self._structure]

  def __eq__(self, other):
    if not isinstance(other, Mapping):
      return False
    if isinstance(other, FlatMapping):
      leaves, structure = other._leaves, other._structure  # pylint: disable=protected-access
    else:
      leaves, (structure, _) = deconstruct(other)
    return self._leaves == leaves and self._structure == structure

  def __getitem__(self, key):
    leaf_index, structure_index, subindex = self._index[key]
    if subindex:
      # Item at key is a subtree
      if structure_index == len(self._structure) - 1:
        end_index = None  # slice till end
      else:
        next_key = self._structure[structure_index + 1][0]
        end_index, _, _ = self._index[next_key]
      return FlatMapping((self._leaves[leaf_index:end_index],
                          (self._structure[structure_index][1], subindex)))
    else:
      # Item at key is a leaf
      return self._leaves[leaf_index]

  def __iter__(self):
    return iter(self.keys())

  def __len__(self):
    return len(self._structure)

  def flatten(self) -> Flat:
    return self._leaves, (self._structure, self._index)

  def __str__(self):
    s = "{}({{{}}})".format(
        type(self).__name__,
        ", ".join("{!r}: {!r}".format(k, v) for k, v in self.items()))
    return s

  __repr__ = __str__


# pylint: disable=unidiomatic-typecheck
def deconstruct(value: Mapping[K, Any]) -> Flat:
  """Deconstruct a Mapping into the three internal structures of a FlatMapping.

  Args:
    value: A (possibly nested) mapping to be deconstructed into flat structures
  Returns:
    Three structures: a flat tuple containing the leaf values of the mapping,
    a tuple containing the original structure of the mapping and a helper index

  eg.:
  >>> leaves, (structure, index) = deconstruct({foo:{a:1, b:2}, bar: 3})

  Leaves represents the flat values of the structure in key order:
  >>> leaves
  (3, 1, 2)

  Leaf values are any values that are not a dict or FlatMapping

  Structure represents the ordered and nested key space of the structure:
  >>> structure
  (('bar',), ('foo', (('a',), ('b',))))

  Index represents a nested index allowing efficient lookup into leaves:
  >>> index
  {'bar': (0, 0, ()), 'foo': (1, 1, {'a': (0, 0, ()), 'b': (1, 1, ())})}
  """
  if type(value) == FlatMapping:
    return value.flatten()  # pytype: disable=attribute-error
  structure = []
  leaves = []
  index = {}
  leaf_index = 0
  structure_index = 0
  # Needs to be sorted by key to guarantee iteration order on the FlatMapping
  items = sorted(value.items(), key=lambda kv: kv[0])

  for key, value in items:

    # Replacing this type check with isinstanceof is much slower for FlatMapping
    # because it inherits from Mapping
    if type(value) == dict or type(value) == FlatMapping:
      new_leaves, (new_structure, new_index) = deconstruct(value)

      structure.append((key, new_structure))
      leaves.extend(new_leaves)
      index[key] = (leaf_index, structure_index, new_index)
      leaf_index += len(new_leaves)

    else:
      # TODO(lenamartens) support case where value is a Sequence:
      # currently the sequence will be stored in the leaves as is,
      # which makes the leaves structure not flat
      structure.append((key,))
      leaves.append(value)
      index[key] = (leaf_index, structure_index, ())
      leaf_index += 1
    structure_index += 1

  return tuple(leaves), (tuple(structure), index)
# pylint: enable=unidiomatic-typecheck

jax.tree_util.register_pytree_node(
    FlatMapping,
    lambda s: s.flatten(),
    lambda treedef, leaves: FlatMapping((leaves, treedef)))
