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

# pylint: disable=unidiomatic-typecheck
# `isinstance(x, Mapping)` is super expensive, so we avoid it where possible
# since we expect constructing some of these types to be on the critical path
# for users.

import collections
import contextlib
import os
import pprint
import threading
from typing import (Any, Callable, Dict, Generic, Mapping, NamedTuple, Optional,
                    Sequence, TypeVar, Union)

from haiku._src import utils
import jax

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
U = TypeVar("U")
PyTreeDef = type(jax.tree_structure(None))


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


class KeysOnlyKeysView(collections.abc.KeysView):
  """KeysView that does not print values when repr'ing."""

  def __init__(self, mapping):
    super().__init__(mapping)  # pytype: disable=wrong-arg-count
    self._mapping = mapping

  def __repr__(self):
    return f"{type(self).__name__}({list(self._mapping)!r})"

  __str__ = __repr__


def to_immutable_dict(mapping: Mapping[K, V]) -> Mapping[K, V]:
  """Returns an immutable copy of the given mapping."""
  if type(mapping) is FlatMap:
    return mapping
  items = []
  for key, value in mapping.items():
    value_type = type(value)
    if value_type is dict:
      value = to_immutable_dict(value)
    items.append((key, value))
  return FlatMap(items)


# TODO(tomhennigan) Better types here (Mapping[K, V]) -> MutableMapping[K, V]?
def to_mutable_dict(mapping):
  """Turns an immutable FlatMapping into a mutable dict."""
  out = {}
  for key, value in mapping.items():
    value_type = type(value)
    if value_type is FlatMap:
      value = to_mutable_dict(value)
    out[key] = value
  return out


def to_haiku_dict(structure: Mapping[K, V]) -> Mapping[K, V]:
  """Returns a copy of the given two level structure.

  Uses the same mapping type as Haiku will return from ``init`` or ``apply``
  functions.

  Args:
    structure: A two level mapping to copy.

  Returns:
    A new two level mapping with the same contents as the input.
  """
  if os.environ.get("HAIKU_FLATMAPPING", "1").lower() not in ("", "0", "false"):
    return to_immutable_dict(structure)
  return to_dict(structure)


def _copy_structure(tree):
  """Returns a copy of the given structure."""
  leaves, treedef = jax.tree_flatten(tree)
  return jax.tree_unflatten(treedef, leaves)


def _to_dict_recurse(value: Any):
  if isinstance(value, Mapping):
    return {k: _to_dict_recurse(v) for k, v in value.items()}
  else:
    return _copy_structure(value)


def to_dict(mapping: Mapping[str, Mapping[str, T]]) -> Dict[str, Dict[str, T]]:
  """Returns a ``dict`` copy of the given two level structure.

  This method is guaranteed to return a copy of the input structure (e.g. even
  if the input is already a ``dict``).

  Args:
    mapping: A two level mapping as returned by ``init`` functions of Haiku
        transforms.

  Returns:
    A new two level mapping with the same contents as the input.
  """
  return _to_dict_recurse(mapping)


def _repr_item(k, v):
  k = repr(k) + ": "
  v = pprint.pformat(v)
  return k + utils.indent(len(k), v).lstrip()


class FlatComponents(NamedTuple):
  leaves: Sequence[Any]
  structure: PyTreeDef


class FlatMap(Mapping[K, V]):
  """Immutable mapping with O(1) flatten and O(n) unflatten operation.

  Warning: this type is only efficient when used with ``jax.tree_*``. When used
  with ``tree.*`` it has similar performance to ``dict``.

  Note that to prevent common errors immutable shims are returned for any
  nested mappings.
  """
  __slots__ = ("_structure", "_leaves", "_mapping")

  def __init__(self, *args, **kwargs):
    """Accepts FlatComponents or the same arguments as `dict`."""
    if not kwargs and len(args) == 1 and type(args[0]) is FlatComponents:
      leaves, structure = args[0]
      mapping = None

      # When unflattening we cannot assume that the leaves are not pytrees (for
      # example: `jax.tree_map(list, my_map)` would pass a list of lists in
      # as leaves).
      if not jax.tree_util.all_leaves(leaves):
        mapping = jax.tree_unflatten(structure, leaves)
        leaves, structure = jax.tree_flatten(mapping)
    else:
      mapping = dict(*args, **kwargs)
      leaves, structure = jax.tree_flatten(mapping)

    self._structure = structure
    self._leaves = tuple(leaves)
    self._mapping = mapping

  def _to_mapping(self) -> Mapping[K, V]:
    if self._mapping is None:
      self._mapping = jax.tree_unflatten(self._structure, self._leaves)
    return self._mapping

  def keys(self):
    return KeysOnlyKeysView(self._to_mapping())

  def values(self):
    return self._to_mapping().values()

  def items(self):
    return self._to_mapping().items()

  def __eq__(self, other):
    if other is None:
      return False
    t = type(other)
    if t is FlatMap:
      other = other._to_mapping()
    return self._to_mapping() == other

  def __hash__(self):
    return hash((self._structure, self._leaves))

  def __getitem__(self, key: K) -> V:
    return self._to_mapping()[key]

  def __getattr__(self, key):
    raise AttributeError(
        f"`x.{key}` is not supported on FlatMapping, use `x['{key}']` instead.")

  def __iter__(self):
    return iter(self.keys())

  def __len__(self):
    return len(self._to_mapping())

  def __str__(self):
    single_line = "{}({{{}}})".format(
        type(self).__name__,
        ", ".join("{!r}: {!r}".format(k, v) for k, v in self.items()))
    if len(single_line) <= 80:
      return single_line

    return "{}({{\n{},\n}})".format(
        type(self).__name__,
        utils.indent(2, ",\n".join(_repr_item(k, v) for k, v in self.items())))

  __repr__ = __str__

  def __reduce__(self):
    # NOTE: Using FlatMapping (not FlatMap) here for backwards compatibility
    # with old pickles.
    return FlatMapping, (self._to_mapping(),)

  # Workaround for https://github.com/python/typing/issues/498.
  __copy__ = None


jax.tree_util.register_pytree_node(
    FlatMap,
    lambda s: (s._leaves, s._structure),  # pylint: disable=protected-access
    lambda treedef, leaves: FlatMap(FlatComponents(leaves, treedef)))


# This is only needed because some naughty people reach in to Haiku internals
# and use `isinstance(x, haiku._src.data_structures.FlatMapping` (which was
# renamed to FlatMap).
# TODO(tomhennigan): If to_immutable_dict is remove this metaclass can go too.
class FlatMappingMeta(type(FlatMap)):

  def __instancecheck__(cls, instance) -> bool:
    return isinstance(instance, FlatMap)


class FlatMapping(FlatMap, metaclass=FlatMappingMeta):
  """Only called from old checkpoints."""

  def __new__(cls, data):
    return to_haiku_dict(data)

  def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
    del args, kwargs
    assert False, "This should never happen."

#      _                               _           _
#   __| | ___ _ __  _ __ ___  ___ __ _| |_ ___  __| |
#  / _` |/ _ \ '_ \| '__/ _ \/ __/ _` | __/ _ \/ _` |
# | (_| |  __/ |_) | | |  __/ (_| (_| | ||  __/ (_| |
#  \__,_|\___| .__/|_|  \___|\___\__,_|\__\___|\__,_|
#            |_|

# The classes below are untested and maintained for backwards compatibility with
# old checkpoints.


class frozendict(Mapping[K, V]):  # pylint: disable=invalid-name
  """Immutable mapping from keys to values."""
  __slots__ = ("_storage", "_keys", "_hash")

  def __init__(self, *args, **kwargs):
    self._storage = dict(*args, **kwargs)
    self._keys = tuple(sorted(self._storage))
    # Dict values aren't necessarily hashable so we just use the keys.
    self._hash = hash(self._keys)

  def keys(self):
    return KeysOnlyKeysView(self)

  def __iter__(self):
    return iter(self._keys)

  def __len__(self):
    return len(self._storage)

  def __getattr__(self, key):
    raise AttributeError(
        f"x.{key} is not supported on frozendict, use x['{key}'] instead.")

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

jax.tree_util.register_pytree_node(
    frozendict,
    lambda s: (tuple(s.values()), tuple(s.keys())),
    lambda k, xs: frozendict(zip(k, xs)))
