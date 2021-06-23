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

import collections
import copy
import pickle
import threading

from absl.testing import absltest
from absl.testing import parameterized
import cloudpickle
import dill
from haiku._src import data_structures
from haiku._src import test_utils
import jax
import tree

frozendict = data_structures.frozendict
FlatMap = data_structures.FlatMap
all_picklers = parameterized.parameters(cloudpickle, dill, pickle)


class StackTest(absltest.TestCase):

  cls = data_structures.Stack

  def test_len(self):
    s = self.cls()
    self.assertEmpty(s)
    for i in range(10):
      self.assertLen(s, i)
      s.push(None)
    for i in range(10):
      self.assertLen(s, 10 - i)
      s.pop()
    self.assertEmpty(s)

  def test_push_peek_pop(self):
    s = self.cls()
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
    s = self.cls()
    s.push(1)
    s.push(2)
    s.push(3)
    s.push(4)
    self.assertEqual(s.popleft(), 1)
    self.assertEqual(s.pop(), 4)
    self.assertEqual(s.popleft(), 2)
    self.assertEqual(s.pop(), 3)

  def test_call(self):
    s = self.cls()
    with s(0):
      self.assertEqual(s.peek(), 0)
      with s(1):
        self.assertEqual(s.peek(), 1)
      self.assertEqual(s.peek(), 0)
    self.assertEmpty(s)

  def test_map(self):
    s1 = self.cls()
    s1.push(1)
    s1.push(2)
    s2 = s1.map(lambda x: x + 2)
    self.assertIsNot(s1, s2)
    self.assertEqual(s1.pop(), 2)
    self.assertEqual(s1.pop(), 1)
    self.assertEqual(s2.pop(), 4)
    self.assertEqual(s2.pop(), 3)

  def test_clone(self):
    s1 = self.cls()
    for i in range(5):
      s1.push(i)
    s2 = s1.clone()
    assert s1 is not s2
    self.assertEqual([s2.pop() for _ in range(len(s2))], [4, 3, 2, 1, 0])
    self.assertEmpty(s2)
    self.assertEqual([s1.pop() for _ in range(len(s1))], [4, 3, 2, 1, 0])
    self.assertEmpty(s2)

  def test_exception_safe(self):
    s = self.cls()
    o1 = object()
    o2 = object()
    with s(o1):
      with self.assertRaisesRegex(ValueError, "expected"):
        with s(o2):
          raise ValueError("expected")
      self.assertIs(s.peek(), o1)
    self.assertEmpty(s)


class ThreadLocalStackTest(StackTest):

  cls = data_structures.ThreadLocalStack

  def test_stack_per_thread(self):
    s = self.cls()
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


class FlatMappingTest(parameterized.TestCase):

  def test_init_from_dict(self):
    o = dict(a=1, b=2)
    f = FlatMap(o)
    self.assertEqual(o, f)
    o["a"] = 2
    self.assertEqual(f["a"], 1)
    self.assertNotEqual(o, f)

  def test_getattr(self):
    f = FlatMap(dict(a=1, b=2))
    with self.assertRaisesRegex(AttributeError, "not supported"):
      _ = f.a

  def test_setattr(self):
    f = FlatMap(dict(a=1))
    with self.assertRaises(AttributeError):
      # Existing attr.
      f.a = 4  # pytype: disable=not-writable
    with self.assertRaises(AttributeError):
      # New attr.
      f.c = 4  # pytype: disable=not-writable

  def test_getitem(self):
    f = FlatMap(dict(a=1, b=2))
    self.assertEqual(f["a"], 1)
    self.assertEqual(f["b"], 2)

  def test_getitem_missing(self):
    f = FlatMap({})
    with self.assertRaises(KeyError):
      f["~"]  # pylint: disable=pointless-statement

  def test_getitem_missing_nested(self):
    f = FlatMap({"~": {}})
    with self.assertRaises(KeyError):
      f["~"]["missing"]  # pylint: disable=pointless-statement

  def test_getitem_nested_immutable(self):
    f = data_structures.to_immutable_dict({"a": {"b": "c"}})
    with self.assertRaisesRegex(TypeError, "does not support item assignment"):
      f["a"]["b"] = "d"

  def test_get(self):
    f = FlatMap(dict(a=1, b=2))
    self.assertEqual(f.get("a"), 1)
    self.assertEqual(f.get("b"), 2)
    self.assertIsNone(f.get("c"))
    self.assertEqual(f.get("d", f), f)

  @parameterized.parameters(jax.tree_map, tree.map_structure)
  def test_tree_map(self, tree_map):
    f = FlatMap(dict(a=1, b=dict(c=2)))
    p = tree_map("v: {}".format, f)
    self.assertEqual(type(p), FlatMap)
    self.assertEqual(p._to_mapping(), {"a": "v: 1", "b": {"c": "v: 2"}})

  def test_eq_hash(self):
    a = FlatMap(dict(a=1, b=2))
    b = FlatMap(dict(a=1, b=2))
    self.assertEqual(a, b)
    self.assertEqual(hash(a), hash(b))

  @parameterized.named_parameters(
      ("copy", copy.copy),
      ("deepcopy", copy.deepcopy),
      ("pickle", lambda v: pickle.loads(pickle.dumps(v)),),
      ("cloudpickle", lambda v: cloudpickle.loads(cloudpickle.dumps(v)),),
      ("dill", lambda v: dill.loads(dill.dumps(v)),),
  )
  def test_copy(self, clone):
    before = data_structures.to_immutable_dict(dict(a=dict(b=1, c=2)))
    after = clone(before)
    self.assertIsNot(before, after)
    self.assertEqual(before, after)
    self.assertEqual(after, {"a": {"b": 1, "c": 2}})
    jax.tree_multimap(self.assertEqual, before, after)

  @all_picklers
  @test_utils.with_environ("HAIKU_FLATMAPPING", "0")
  def test_pickle_roundtrip(self, pickler):
    self.assertRoundTripType(pickler, dict)

  @all_picklers
  @test_utils.with_environ("HAIKU_FLATMAPPING", "1")
  def test_pickle_roundtrip_flatmap(self, pickler):
    self.assertRoundTripType(pickler, FlatMap)

  def assertRoundTripType(self, pickler, cls):
    x = FlatMap({})
    y = pickler.loads(pickler.dumps(x))
    self.assertType(y, cls)

  @test_utils.with_environ("HAIKU_FLATMAPPING", "0")
  def test_golden_pickle_load(self):
    self.assertEmptyPickle(dict)

  @test_utils.with_environ("HAIKU_FLATMAPPING", "1")
  def test_golden_pickle_load_flatmap(self):
    self.assertEmptyPickle(FlatMap)

  def assertEmptyPickle(self, cls):
    loaded = self._pickle_load_golden("empty")
    self.assertType(loaded, cls)
    self.assertEmpty(loaded)

  def assertType(self, obj, cls):
    self.assertEqual(type(obj), cls)

  def _pickle_load_golden(self, file):
    with open(f"haiku/_src/testdata/{file}.pkl", "rb") as fp:
      return pickle.load(fp)

  @test_utils.with_environ("HAIKU_FLATMAPPING", "0")
  def test_golden_pickle_nested(self):
    self.assertNestedPickle(dict)

  @test_utils.with_environ("HAIKU_FLATMAPPING", "1")
  def test_golden_pickle_nested_flatmap(self):
    self.assertNestedPickle(FlatMap)

  def assertNestedPickle(self, cls):
    loaded = self._pickle_load_golden("nested")
    self.assertType(loaded, cls)
    self.assertType(loaded["a"], cls)
    self.assertType(loaded["a"]["b"], cls)
    self.assertEqual(loaded, {"a": {"b": {"c": 1}}})

  def test_flatmapping_isinstance(self):
    # Note: This should not work (FlatMapping extends FlatMap not the other way
    # around) however it is needed to support some naughty users who reached in
    # to Haiku internals and depend on the name `data_structures.FlatMapping`.
    o = FlatMap({})
    self.assertIsInstance(o, data_structures.FlatMapping)

  def test_flatmapping_init(self):
    o = data_structures.FlatMapping({})
    self.assertEqual(type(o), data_structures.FlatMap)
    self.assertIsInstance(o, data_structures.FlatMapping)

  def test_deepcopy_still_immutable(self):
    before = FlatMap(dict(a=[1, 2, 3]))
    after = copy.deepcopy(before)
    with self.assertRaises(TypeError):
      before["a"] = [3, 2, 1]  # pytype: disable=unsupported-operands
    self.assertEqual(before["a"], [1, 2, 3])
    self.assertEqual(after["a"], [1, 2, 3])

  def test_keys(self):
    d = FlatMap({"key1": "value", "key2": "value2"})
    self.assertEqual(str(d.keys()), "KeysOnlyKeysView(['key1', 'key2'])")
    self.assertEqual(repr(d.keys()), "KeysOnlyKeysView(['key1', 'key2'])")

  def test_init(self):
    # Init from dict
    d = {"foo": {"a": 1}, "bar": 2}
    f = FlatMap(d)
    self.assertEqual(f, d)

    # Init from FlatMap
    f2 = FlatMap(f)
    self.assertEqual(f, f2)

    # Init from dict with nested FlatMap
    inner = FlatMap({"a": 1})
    outer = {"foo": inner, "bar": 2}
    nested_flatmapping = FlatMap(outer)
    self.assertEqual(outer, nested_flatmapping)

    # Init from flat structures
    values, treedef = jax.tree_flatten(f)
    self.assertEqual(
        FlatMap(data_structures.FlatComponents(values, treedef)), f)

  def test_get_item(self):
    f_map = FlatMap(
        {"foo": {"b": [1], "d": {"e": 2}}, "bar": (1,)})
    self.assertEqual(f_map["foo"], {"b": [1], "d": {"e": 2}})
    self.assertEqual(f_map["bar"], (1,))
    with self.assertRaises(KeyError):
      _ = f_map["b"]

  def test_items(self):
    f_map = FlatMap(
        {"foo": {"b": {"c": 1}, "d": {"e": 2}}, "bar": {"c": 1}})
    items = list(f_map.items())
    self.assertEqual(items[0], ("foo", {"b": {"c": 1}, "d": {"e": 2}}))
    self.assertEqual(items[1], ("bar", {"c": 1}))
    self.assertEqual(items, list(zip(f_map.keys(), f_map.values())))

  def test_tree_functions(self):
    f = FlatMap(
        {"foo": {"b": {"c": 1}, "d": 2}, "bar": {"c": 1}})

    m = jax.tree_map(lambda x: x + 1, f)
    self.assertEqual(type(m), FlatMap)
    self.assertEqual(m, {"foo": {"b": {"c": 2}, "d": 3}, "bar": {"c": 2}})

    mm = jax.tree_multimap(lambda x, y: x + y, f, f)
    self.assertEqual(type(mm), FlatMap)
    self.assertEqual(mm, {"foo": {"b": {"c": 2}, "d": 4}, "bar": {"c": 2}})

    leaves, treedef = jax.tree_flatten(f)
    self.assertEqual(leaves, [1, 1, 2])
    uf = jax.tree_unflatten(treedef, leaves)
    self.assertEqual(type(f), FlatMap)
    self.assertEqual(f, uf)

  def test_flatten_nested_struct(self):
    d = {"foo": {"bar": [1, 2, 3]},
         "baz": {"bat": [4, 5, 6],
                 "qux": [7, [8, 9]]}}
    f = FlatMap(d)
    leaves, treedef = jax.tree_flatten(f)
    self.assertEqual([4, 5, 6, 7, 8, 9, 1, 2, 3], leaves)
    g = jax.tree_unflatten(treedef, leaves)
    self.assertEqual(g, f)
    self.assertEqual(g, d)

  def test_nested_sequence(self):
    f_map = FlatMap(
        {"foo": [1, 2], "bar": [{"a": 1}, 2]})
    leaves, _ = jax.tree_flatten(f_map)

    self.assertEqual(leaves, [1, 2, 1, 2])
    print(f_map["foo"])
    self.assertEqual(f_map["foo"][0], 1)

  @parameterized.named_parameters(("tuple", tuple), ("list", list),)
  def test_different_sequence_types(self, type_of_sequence):
    f_map = FlatMap(
        {"foo": type_of_sequence((1, 2)),
         "bar": type_of_sequence((3, {"b": 4}))})
    leaves, _ = jax.tree_flatten(f_map)

    self.assertEqual(leaves, [3, 4, 1, 2])
    self.assertEqual(f_map["foo"][0], 1)
    self.assertEqual(f_map["bar"][1]["b"], 4)

  def test_replace_leaves_with_nodes_in_map(self):
    f = FlatMap({"foo": 1, "bar": 2})

    f_nested = jax.tree_map(lambda x: {"a": (x, x)}, f)
    leaves, _ = jax.tree_flatten(f_nested)

    self.assertEqual(leaves, [2, 2, 1, 1])

  def test_frozen_builtins_jax_compatibility(self):
    f = FlatMap({"foo": [3, 2], "bar": {"a": 3}})
    mapped_frozen_list = jax.tree_map(lambda x: x+1, f["foo"])
    self.assertEqual(mapped_frozen_list[0], 4)

    mapped_frozen_dict = jax.tree_map(lambda x: x+1, f["bar"])
    self.assertEqual(mapped_frozen_dict["a"], 4)

  def test_tree_transpose(self):
    outerdef = jax.tree_structure(FlatMap({"a": 1, "b": 2}))
    innerdef = jax.tree_structure([1, 2])
    self.assertEqual(
        [FlatMap({"a": 3, "b": 5}), FlatMap({"a": 4, "b": 6})],
        jax.tree_transpose(
            outerdef, innerdef, FlatMap({"a": [3, 4], "b": [5, 6]})))


class DataStructuresTest(parameterized.TestCase):

  @parameterized.parameters(dict, frozendict, FlatMap,
                            lambda x: collections.defaultdict(object, x))
  def test_to_dict(self, cls):
    mapping_in = cls(
        {f"a{i}": cls({f"b{j}": 0 for j in range(2)}) for i in range(10)})
    mapping_out = data_structures.to_dict(mapping_in)
    self.assertEqual(mapping_in, mapping_out)
    self.assertIs(type(mapping_out), dict)
    self.assertIsNot(mapping_in, mapping_out)
    for key in mapping_in:
      self.assertIs(type(mapping_out[key]), dict)
      self.assertIsNot(mapping_in[key], mapping_out[key])

  def test_to_dict_copies_value_structure(self):
    v = [1, 2, 3]
    mapping_in = {"m": {"w": v}}
    mapping_out = data_structures.to_dict(mapping_in)
    self.assertEqual(mapping_in, mapping_out)
    self.assertIsNot(mapping_in["m"]["w"], mapping_out["m"]["w"])
    v.append(4)
    self.assertNotEqual(mapping_in, mapping_out)

  def test_to_dict_recursively_changes_leaf_types(self):
    mapping_in = {"m": {"w": FlatMap(a=FlatMap(b=0))}}
    mapping_out = data_structures.to_dict(mapping_in)
    self.assertEqual(type(mapping_out["m"]["w"]), dict)
    self.assertEqual(type(mapping_out["m"]["w"]["a"]), dict)

  def test_to_immutable_dict(self):
    before = {"a": {"b": 1, "c": 2}}
    after = data_structures.to_immutable_dict(before)
    self.assertEqual(before, after)
    self.assertEqual(type(after), FlatMap)
    self.assertEqual(type(after["a"]), FlatMap)

  def test_to_mutable_dict(self):
    before = FlatMap({"a": {"b": 1, "c": 2}})
    after = data_structures.to_mutable_dict(before)
    self.assertEqual(before, after)
    self.assertEqual(type(after), dict)
    self.assertEqual(type(after["a"]), dict)

if __name__ == "__main__":
  absltest.main()
