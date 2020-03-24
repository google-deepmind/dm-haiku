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
"""Tests for haiku._src.data_structures."""

import copy
import pickle
import threading

from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle
import dill
from haiku._src import data_structures
import jax
import tree

frozendict = data_structures.frozendict
Stack = data_structures.Stack
FlatMapping = data_structures.FlatMapping


class StackTest(absltest.TestCase):

  def test_len(self):
    s = Stack()
    self.assertEmpty(s)
    for i in range(10):
      self.assertLen(s, i)
      s.push(None)
    for i in range(10):
      self.assertLen(s, 10 - i)
      s.pop()
    self.assertEmpty(s)

  def test_push_peek_pop(self):
    s = Stack()
    for i in range(3):
      s.push(i)
    self.assertEqual(s.peek(), 2)
    self.assertEqual(s.peek(-2), 1)
    self.assertEqual(s.peek(-3), 0)
    for i in range(3):
      self.assertEqual(s.peek(), 2 - i)
      self.assertEqual(s.pop(), 2 - i)
    self.assertEmpty(s)

  def test_popleft(self):
    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    self.assertEqual(s.popleft(), 1)
    self.assertEqual(s.pop(), 4)
    self.assertEqual(s.popleft(), 2)
    self.assertEqual(s.pop(), 3)

  def test_call(self):
    s = Stack()
    with s(0):
      self.assertEqual(s.peek(), 0)
      with s(1):
        self.assertEqual(s.peek(), 1)
      self.assertEqual(s.peek(), 0)
    self.assertEmpty(s)

  def test_map(self):
    s1 = Stack()
    s1.push(1)
    s1.push(2)
    s2 = s1.map(lambda x: x + 2)
    self.assertIsNot(s1, s2)
    self.assertEqual(s1.pop(), 2)
    self.assertEqual(s1.pop(), 1)
    self.assertEqual(s2.pop(), 4)
    self.assertEqual(s2.pop(), 3)

  def test_clone(self):
    s1 = Stack()
    for i in range(5):
      s1.push(i)
    s2 = s1.clone()
    assert s1 is not s2
    self.assertEqual([s2.pop() for _ in range(len(s2))], [4, 3, 2, 1, 0])
    self.assertEmpty(s2)
    self.assertEqual([s1.pop() for _ in range(len(s1))], [4, 3, 2, 1, 0])
    self.assertEmpty(s2)


class ThreadLocalStackTest(absltest.TestCase):

  def test_stack_per_thread(self):
    s = data_structures.ThreadLocalStack()
    self.assertEmpty(s)
    s.push(42)

    s_len_second_thread = [None]

    def second_thread():
      self.assertEmpty(s)
      s.push(666)
      s.push(777)
      s_len_second_thread[0] = len(s)

    t = threading.Thread(target=second_thread)
    t.start()
    t.join()

    self.assertEqual(s_len_second_thread[0], 2)
    self.assertEqual(s.pop(), 42)
    self.assertEmpty(s)


class FrozenDictTest(parameterized.TestCase):

  def test_init_from_dict(self):
    o = dict(a=1, b=2)
    f = frozendict(o)
    self.assertEqual(o, f)
    o["a"] = 2
    self.assertEqual(f["a"], 1)
    self.assertNotEqual(o, f)

  def test_getattr(self):
    f = frozendict(a=1, b=2)
    self.assertEqual(f.a, 1)
    self.assertEqual(f.b, 2)

  def test_setattr(self):
    f = frozendict(a=1)
    with self.assertRaises(AttributeError):
      # Existing attr.
      f.a = 4  # pytype: disable=not-writable
    with self.assertRaises(AttributeError):
      # New attr.
      f.c = 4  # pytype: disable=not-writable

  def test_getitem(self):
    f = frozendict(a=1, b=2)
    self.assertEqual(f["a"], 1)
    self.assertEqual(f["b"], 2)

  def test_get(self):
    f = frozendict(a=1, b=2)
    self.assertEqual(f.get("a"), 1)
    self.assertEqual(f.get("b"), 2)
    self.assertEqual(f.get("c"), None)
    self.assertEqual(f.get("d", f), f)

  @parameterized.parameters(jax.tree_util.tree_map, tree.map_structure)
  def test_tree_map(self, tree_map):
    f = frozendict(a=1, b=frozendict(c=2))
    p = tree_map("v: {}".format, f)
    self.assertEqual(type(p), frozendict)
    self.assertEqual(p, {"a": "v: 1", "b": {"c": "v: 2"}})

  def test_eq_hash(self):
    a = frozendict(a=1, b=2)
    b = frozendict(a=1, b=2)
    self.assertEqual(a, b)
    self.assertEqual(hash(a), hash(b))

  @parameterized.named_parameters(
      # ("copy", copy.copy),  # TODO(tomhennigan) Re-enable.
      ("deepcopy", copy.deepcopy),
      ("pickle", lambda v: pickle.loads(pickle.dumps(v)),),
      ("cloudpickle", lambda v: cloudpickle.loads(cloudpickle.dumps(v)),),
      ("dill", lambda v: dill.loads(dill.dumps(v)),),
  )
  def test_copy(self, clone):
    before = frozendict(a=frozendict(b=1, c=2))
    after = clone(before)
    self.assertIsNot(before, after)
    self.assertEqual(before, after)
    self.assertEqual(after, {"a": {"b": 1, "c": 2}})

  def test_keys(self):
    d = frozendict({"key1": "value", "key2": "value2"})
    self.assertEqual(str(d.keys()), "KeysOnlyKeysView(['key1', 'key2'])")
    self.assertEqual(repr(d.keys()), "KeysOnlyKeysView(['key1', 'key2'])")


class FlatMappingTest(parameterized.TestCase):

  def test_init(self):
    # Init from dict
    d = {"foo": {"a": 1}, "bar": 2}
    f = FlatMapping.from_mapping(d)
    self.assertEqual(f, d)

    # Init from FlatMapping
    f2 = FlatMapping.from_mapping(f)
    self.assertEqual(f, f2)

    # Init from dict with nested FlatMapping
    inner = FlatMapping.from_mapping({"a": 1})
    outer = {"foo": inner, "bar": 2}
    nested_flatmapping = FlatMapping.from_mapping(outer)
    self.assertEqual(outer, nested_flatmapping)

    # Init from flat structures
    values, treedef = f.flatten()
    self.assertEqual(FlatMapping((values, treedef)), f)

  def test_get_item(self):
    f_map = FlatMapping.from_mapping({"foo": {"b": [1], "d": {"e": 2}},
                                      "bar": (1,)})
    self.assertEqual(f_map["foo"], {"b": [1], "d": {"e": 2}})
    self.assertEqual(f_map["bar"], (1,))
    with self.assertRaises(KeyError):
      _ = f_map["b"]

    with self.assertRaises(TypeError):
      f_map["foo"]["b"] = 2

  def test_items(self):
    f_map = FlatMapping.from_mapping({"foo": {"b": {"c": 1}, "d": {"e": 2}},
                                      "bar": {"c": 1}})
    items = list(f_map.items())
    self.assertEqual(items[0], ("bar", {"c": 1}))
    self.assertEqual(items[1], ("foo", {"b": {"c": 1}, "d": {"e": 2}}))
    self.assertEqual(items, list(zip(f_map.keys(), f_map.values())))

  def test_tree_functions(self):
    f = FlatMapping.from_mapping({"foo": {"b": {"c": 1}, "d": 2},
                                  "bar": {"c": 1}})

    m = jax.tree_map(lambda x: x + 1, f)
    self.assertEqual(type(m), FlatMapping)
    self.assertEqual(m, {"foo": {"b": {"c": 2}, "d": 3}, "bar": {"c": 2}})

    mm = jax.tree_multimap(lambda x, y: x + y, f, f)
    self.assertEqual(type(mm), FlatMapping)
    self.assertEqual(mm, {"foo": {"b": {"c": 2}, "d": 4}, "bar": {"c": 2}})

    leaves, treedef = jax.tree_flatten(f)
    self.assertEqual(leaves, [1, 1, 2])
    uf = jax.tree_unflatten(treedef, leaves)
    self.assertEqual(type(f), FlatMapping)
    self.assertEqual(f, uf)

  @parameterized.named_parameters(("tuple", tuple), ("list", list),)
  def test_different_sequence_types(self, type_of_sequence):
    f_map = FlatMapping.from_mapping({"foo": type_of_sequence((1, 2)),
                                      "bar": type_of_sequence((3, {"b": 4}))})
    leaves, _ = f_map.flatten()

    self.assertEqual(leaves, (3, 4, 1, 2))
    self.assertEqual(f_map["foo"][0], 1)
    self.assertEqual(f_map["bar"][1]["b"], 4)

  def test_replace_leaves_with_nodes_in_map(self):
    f = FlatMapping.from_mapping({"foo": 1, "bar": 2})

    f_nested = jax.tree_map(lambda x: {"a": (x, x)}, f)
    leaves, _ = f_nested.flatten()

    self.assertEqual(leaves, (2, 2, 1, 1))


class DataStructuresTest(absltest.TestCase):

  def test_to_immutable_dict(self):
    before = {"a": {"b": 1, "c": 2}}
    after = data_structures.to_immutable_dict(before)
    self.assertEqual(before, after)
    self.assertEqual(type(after), frozendict)
    self.assertEqual(type(after["a"]), frozendict)

  def test_to_mutable_dict(self):
    before = frozendict({"a": {"b": 1, "c": 2}})
    after = data_structures.to_mutable_dict(before)
    self.assertEqual(before, after)
    self.assertEqual(type(after), dict)
    self.assertEqual(type(after["a"]), dict)


if __name__ == "__main__":
  absltest.main()
