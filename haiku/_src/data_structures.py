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
import types
from typing import Any, Callable, Generic, Mapping, Optional, TypeVar, Tuple, Sequence, Union

from haiku._src import utils
import jax
from jax import tree_util

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


class KeysOnlyKeysView(collections.abc.KeysView):
  """KeysView that does not print values when repr'ing."""

  def __init__(self, mapping):
    super().__init__(mapping)  # pytype: disable=wrong-arg-count
    self._mapping = mapping

  def __repr__(self):
    return f"{type(self).__name__}({list(self._mapping)!r})"

  __str__ = __repr__


# TODO(lenamartens) Deprecate type
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

# PyTreeDef is defined in jaxlib/pytree.cc but not exposed.
PyTreeDef = Any
FlatComponents = Tuple[Sequence[Any], PyTreeDef]


class FlatMapping(Mapping[K, V]):
  """Immutable mapping with O(1) flatten and O(n) unflatten operation."""

  def __init__(self, flat: FlatComponents, check_leaves: bool = True):
    """Constructs a flat mapping from already flattened components.

    Args:
      flat: A tuple containing a flat sequence of values and a PyTreeDef
      representing the output of jax.tree_flatten on a structure.
      check_leaves: Check if all leaves are flat values, and reflatten if not.
      This check is O(n), whereas the normal construction time is O(1).
    """
    leaves, structure = flat

    # TODO(lenamartens): upstream is_leaf check to Jax
    is_leaf = lambda x: type(x) not in tree_util._registry  # pylint: disable=unidiomatic-typecheck  pylint: disable=protected-access
    if check_leaves and not all(map(is_leaf, leaves)):
      tree = jax.tree_unflatten(structure, leaves)
      leaves, structure = jax.tree_flatten(tree)

    self._structure = structure
    self._leaves = tuple(leaves)
    self._mapping = None

  @classmethod
  def from_mapping(cls, m: Mapping[Any, Any]) -> "FlatMapping":
    """Construct a new FlatMapping object from a Mapping.

    Args:
      m: If m is a FlatMapping, the internal structures from m are reused
         to construct a new FlatMapping (this is O(1))
         If m is any other Mapping, m will be flattened (this is O(n logn))
    Returns:
      A FlatMapping with the keys and values of the input mapping.
    """
    if isinstance(m, FlatMapping):
      return cls(m.flatten(), check_leaves=False)
    return cls(jax.tree_flatten(m), check_leaves=False)

  def to_mapping(self) -> Mapping[K, V]:
    if not self._mapping:
      self._mapping = jax.tree_unflatten(self._structure, self._leaves)
    return self._mapping

  def keys(self):
    return KeysOnlyKeysView(self.to_mapping())

  def values(self):
    return self.to_mapping().values()

  def items(self):
    return self.to_mapping().items()

  def __eq__(self, other):
    if not isinstance(other, Mapping):
      return False
    if isinstance(other, FlatMapping):
      leaves, structure = other._leaves, other._structure  # pylint: disable=protected-access
    else:
      leaves, structure = jax.tree_flatten(other)
    return self._leaves == tuple(leaves) and self._structure == structure

  def __getitem__(self, key):
    value = self.to_mapping()[key]
    if isinstance(value, Mapping):
      # Create a read-only version to prevent modification of the returned item:
      # modifying the item will not modify the node in the FlatMapping,
      # which will be confusing.
      value = types.MappingProxyType(value)
    return value

  def __iter__(self):
    return iter(self.keys())

  def __len__(self):
    return len(self.keys())

  def flatten(self) -> FlatComponents:
    return self._leaves, self._structure

  def __str__(self):
    single_line = "{}({{{}}})".format(
        type(self).__name__,
        ", ".join("{!r}: {!r}".format(k, v) for k, v in self.items()))
    if len(single_line) <= 80:
      return single_line

    return "{}({{\n{},\n}})".format(
        type(self).__name__,
        utils.indent(
            2, ",\n".join(_repr_item(k, v) for k, v in self.items())))

  __repr__ = __str__


jax.tree_util.register_pytree_node(
    FlatMapping,
    lambda s: s.flatten(),
    lambda treedef, leaves: FlatMapping((leaves, treedef)))
