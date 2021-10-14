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
"""Tests for haiku._src.filtering."""

import collections
import itertools
import re
import types
from typing import Any, Callable, Sequence, Set, Tuple

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import basic
from haiku._src import data_structures
from haiku._src import filtering
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp


def jax_fn_with_filter(
    jax_fn: Callable[..., Any],
    f: Callable[..., Any],
    predicate: Callable[[str, str, jnp.ndarray], bool],
    **jax_fn_kwargs) -> Callable[..., Any]:
  """Applies a jax functionn to a given function after modifying its signature.

  `jax_fn_with_filter` operates in two steps:
  1   it wraps the input function `f`, which is expect to take as first
      argument a `Params` data structure, with a function taking as first two
      inputs a bipartition of the orginal parameters
  2   the resulting function is transformed with `jax_fn` and wrapped
      by a function supporting `f`'s signature and taking care of partitioning
      the `f`'s `Params` input using `predicate`.

  Args:
    jax_fn: jax function, e.g. `jax.grad` or `jax.jacobian`.
    f: callable to be transformed.
    predicate: predicate to be used to partition `f`'s input parameters.
    **jax_fn_kwargs: kwargs to be forwarded to `jax_fn`.

  Returns:
    Function calling the input jax function on the wrapped `f`.
  """
  def wrapper(p1, p2, *args, **kwargs):
    return f(filtering.merge(p1, p2), *args, **kwargs)
  jaxed_fn = jax_fn(wrapper, **jax_fn_kwargs)

  def fn_with_filter(p, *args, **kwargs):
    p1, p2 = filtering.partition(predicate, p)
    return jaxed_fn(p1, p2, *args, **kwargs)

  return fn_with_filter


def get_net(x):
  def init(v):
    return dict(
        w_init=lambda *args: v * jnp.ones((1, 1)),
        b_init=lambda *args: v * 1.5 * jnp.ones((1,)))
  h = basic.Linear(output_size=1, name="first_layer", **init(1.0))(x)
  h = basic.Linear(output_size=1, name="second_layer", **init(3.0))(h)
  return jnp.mean(h)


def get_names(params) -> Set[str]:
  names = set([])
  for path, module in params.items():
    for name in module.keys():
      names.add("/".join([path, name]))
  return names


def to_set(params) -> Set[Tuple[str, Sequence[float]]]:
  entries = set([])
  for path, module in params.items():
    for key, value in module.items():
      entries.add(
          ("/".join([path, key]), tuple(jax.device_get(value).flatten())))
  return entries


def compile_regex(regex):
  if not isinstance(regex, str):
    regex = "|".join(["(" + r + ")" for r in regex])
  return re.compile(regex)


class FilteringTest(parameterized.TestCase):

  def test_partition(self):

    init_fn, _ = transform.transform(get_net)
    params = init_fn(jax.random.PRNGKey(428), jnp.ones((1, 1)))

    # parse by layer
    first_layer_params, second_layer_params = filtering.partition(
        lambda module_name, *_: module_name == "first_layer",
        params)
    self.assertEqual(
        get_names(first_layer_params),
        set(["first_layer/w", "first_layer/b"]))
    self.assertEqual(
        get_names(second_layer_params),
        set(["second_layer/w", "second_layer/b"]))

    # parse by variable type
    weights, biases = filtering.partition(
        lambda module_name, name, _: name == "w",
        params)  # pytype: disable=wrong-arg-types
    self.assertEqual(
        get_names(weights),
        set(["first_layer/w", "second_layer/w"]))
    self.assertEqual(
        get_names(biases),
        set(["first_layer/b", "second_layer/b"]))

    # Compose regexes
    regex = compile_regex(["first_layer.*", ".*w"])
    matching, not_matching = filtering.partition(
        lambda module_name, name, _: regex.match(f"{module_name}/{name}"),
        params)
    self.assertEqual(
        get_names(matching),
        set(["first_layer/w", "first_layer/b", "second_layer/w"]))
    self.assertEqual(
        get_names(not_matching),
        set(["second_layer/b"]))

    matching, not_matching = filtering.partition(
        lambda mod_name, name, _: mod_name == "first_layer" and name != "w",
        params)
    self.assertEqual(
        get_names(matching),
        set(["first_layer/b"]))
    self.assertEqual(
        get_names(not_matching),
        set(["first_layer/w", "second_layer/w", "second_layer/b"]))

  @parameterized.parameters(*range(1, 8))
  def test_partition_n(self, n):
    cnt = itertools.count()
    fn = lambda m, n, v: next(cnt)
    structure = {f"layer_{i}": {"w": None} for i in range(n)}
    structures = filtering.partition_n(fn, structure, n)
    self.assertLen(structures, n)
    self.assertEqual(filtering.merge(*structures), structure)
    for i, substructure in enumerate(structures):
      expected = {f"layer_{i}": {"w": None}}
      self.assertEqual(substructure, expected)

  def test_partition_n_nested(self):
    nested_structure = {"layer": {"a": [1, 2, 3],
                                  "b": set([object()]),
                                  "c": {"a": "b"}}}
    cnt = itertools.count()
    fn = lambda m, n, v: next(cnt)
    out1, out2, out3 = filtering.partition_n(fn, nested_structure, 3)
    self.assertEqual(out1, {"layer": {"a": nested_structure["layer"]["a"]}})
    self.assertEqual(out2, {"layer": {"b": nested_structure["layer"]["b"]}})
    self.assertEqual(out3, {"layer": {"c": nested_structure["layer"]["c"]}})

  @parameterized.parameters(*range(1, 8))
  def test_partition_n_merge_isomorphism(self, n):
    cnt = itertools.count()
    fn = lambda m, n, v: next(cnt)
    input_structure = {f"layer_{i}": {"w": None} for i in range(n)}
    structures = filtering.partition_n(fn, input_structure, n)
    merged_structure = filtering.merge(*structures)
    self.assertEqual(merged_structure, input_structure)

  @parameterized.parameters(*range(1, 8))
  def test_traverse(self, n):
    structure = {f"layer_{i}": {"w": "wv", "b": "bv"}
                 for i in reversed(range(n))}
    expected = []
    for i in range(n):
      expected.append((f"layer_{i}", "b", "bv"))
      expected.append((f"layer_{i}", "w", "wv"))
    actual = list(filtering.traverse(structure))
    self.assertEqual(expected, actual)

  def test_traverse_nested(self):
    nested_structure = {"layer": {"a": [1, 2, 3],
                                  "b": set([object()]),
                                  "c": {"a": "b"}}}
    expected = [
        ("layer", x, nested_structure["layer"][x]) for x in ("a", "b", "c")
    ]
    actual = list(filtering.traverse(nested_structure))
    self.assertEqual(expected, actual)

  @parameterized.parameters(({}, {}, True),
                            ({"a": {}}, {}, True),
                            ({}, {"a": {}}, True),
                            ({"a": {}}, {"a": {}}, True),
                            ({"a": {"b": 1}}, {"a": {}}, False))
  def test_is_subset(self, structure1, structure2, is_subset):
    if is_subset:
      self.assertTrue(
          filtering.is_subset(subset=structure1, superset=structure2))
    else:
      self.assertFalse(
          filtering.is_subset(subset=structure1, superset=structure2))

  @parameterized.parameters(*range(1, 4))
  def test_is_subset_layers(self, n):
    structure_small = {f"layer_{i}": {"w": "wv", "b": "bv"}
                       for i in reversed(range(n - 1))}
    structure_large = {f"layer_{i}": {"w": "wv", "b": "bv"}
                       for i in reversed(range(n))}
    self.assertTrue(
        filtering.is_subset(subset=structure_small, superset=structure_large))
    self.assertFalse(
        filtering.is_subset(subset=structure_large, superset=structure_small))

  def test_filter(self):
    init_fn, _ = transform.transform(get_net)
    params = init_fn(jax.random.PRNGKey(428), jnp.ones((1, 1)))

    second_layer_params = filtering.filter(
        lambda module_name, *_: module_name == "second_layer",
        params)
    self.assertEqual(
        get_names(second_layer_params),
        set(["second_layer/w", "second_layer/b"]))

    biases = filtering.filter(
        lambda module_name, name, _: name == "b",
        params)  # pytype: disable=wrong-arg-types
    self.assertEqual(
        get_names(biases),
        set(["first_layer/b", "second_layer/b"]))

  def test_transforms_with_filter(self):
    # Note to make sense of test:
    #
    # out = (w0 + b0) * w1 + b1
    #     = w0 * w1 + b0 * w1 + b1
    # doutdw0 = w1
    # doutdw1 = w0 + b0
    # with w0 = 1.0, b0 = 1.5, w1 = 3.0, b1 = 4.5
    init_fn, apply_fn = transform.transform(get_net)
    inputs = jnp.ones((1, 1))
    params = init_fn(jax.random.PRNGKey(428), inputs)

    df_fn = jax_fn_with_filter(
        jax_fn=jax.grad,
        f=apply_fn,
        predicate=lambda module_name, name, _: name == "w")
    df = df_fn(params, None, inputs)
    self.assertEqual(
        to_set(df),
        set([("first_layer/w", (3.0,)), ("second_layer/w", (2.5,))]))

    fn = jax_fn_with_filter(
        jax_fn=jax.value_and_grad,
        f=apply_fn,
        predicate=lambda module_name, name, _: name == "w")
    v = fn(params, None, inputs)
    self.assertEqual(v[0], jnp.array([12.0]))
    self.assertEqual(to_set(df), to_set(v[1]))

    def get_stacked_net(x):
      y = get_net(x)
      return jnp.stack([y, 2.0 * y])
    _, apply_fn = transform.transform(get_stacked_net)
    jf_fn = jax_fn_with_filter(
        jax_fn=jax.jacobian,
        f=apply_fn,
        predicate=lambda module_name, name, _: name == "w")
    jf = jf_fn(params, None, inputs)

    self.assertEqual(
        to_set(jf),
        set([("first_layer/w", (3.0, 6.0)), ("second_layer/w", (2.5, 5.0))]))

  def test_map(self):
    init_fn, _ = transform.transform(get_net)
    params = init_fn(jax.random.PRNGKey(428), jnp.ones((1, 1)))

    # parse by layer
    def map_fn(module_name, name, v):
      del name
      if "first_layer" in module_name:
        return v
      else:
        return 2. * v

    new_params = filtering.map(map_fn, params)
    self.assertLen(jax.tree_leaves(new_params), 4)

    first_layer_params, second_layer_params = filtering.partition(
        lambda module_name, *_: module_name == "first_layer",
        params)
    for mn in first_layer_params:
      for n in first_layer_params[mn]:
        self.assertEqual(params[mn][n], new_params[mn][n])

    for mn in second_layer_params:
      for n in second_layer_params[mn]:
        self.assertEqual(2. * params[mn][n], new_params[mn][n])

  @test_utils.with_environ("HAIKU_FLATMAPPING", None)
  def test_output_type_default(self):
    self.assert_output_type(data_structures.FlatMap)

  @test_utils.with_environ("HAIKU_FLATMAPPING", "0")
  def test_output_type_env_var_0(self):
    self.assert_output_type(dict)

  @test_utils.with_environ("HAIKU_FLATMAPPING", "1")
  def test_output_type_env_var_1(self):
    self.assert_output_type(data_structures.FlatMap)

  @test_utils.with_environ("HAIKU_FLATMAPPING", "0")
  def test_merge_different_mappings(self):
    a = collections.defaultdict(dict)
    a["foo"]["bar"] = 1
    b = {"foo": {"baz": 2}}
    c = types.MappingProxyType({"foo": {"bat": 3}})
    d = filtering.merge(a, b, c)
    self.assertEqual(d, {"foo": {"bar": 1, "baz": 2, "bat": 3}})

  def test_merge_nested(self):
    a = {"layer": {"a": [1, 2, 3]}}
    b = {"layer": {"b": set([object()])}}
    c = {"layer": {"c": {"a": "b"}}}
    actual = filtering.merge(a, b, c)
    expected = {"layer": {"a": a["layer"]["a"],
                          "b": b["layer"]["b"],
                          "c": c["layer"]["c"]}}
    self.assertEqual(expected, actual)

  def assert_output_type(self, out_cls):
    def assert_type_recursive(s):
      self.assertEqual(type(s), out_cls)

    for in_cls in (dict, data_structures.FlatMap):
      with self.subTest(str(in_cls)):
        structure_a = in_cls({"m1": in_cls({"w": None})})
        structure_b = in_cls({"m2": in_cls({"w": None})})
        structure_c = in_cls({f"{i}": in_cls({"w": None}) for i in range(5)})
        assert_type_recursive(
            filtering.filter(lambda m, n, v: True, structure_a))
        assert_type_recursive(filtering.map(lambda m, n, v: v, structure_a))
        assert_type_recursive(filtering.merge(structure_a, structure_b))
        parts = filtering.partition(lambda m, n, v: int(m) > 1, structure_c)
        for part in parts:
          assert_type_recursive(part)
        parts = filtering.partition_n(lambda m, n, v: int(m), structure_c, 5)
        for part in parts:
          assert_type_recursive(part)


if __name__ == "__main__":
  absltest.main()
