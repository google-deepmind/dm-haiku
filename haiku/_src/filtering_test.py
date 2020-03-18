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
"""Tests for haiku._src.filtering."""

import re
from typing import Any, Callable, Sequence, Set, Text, Tuple
from absl.testing import absltest
from haiku._src import basic
from haiku._src import filtering
from haiku._src import transform
from haiku._src.typing import Params
import jax
import jax.numpy as jnp


def jax_fn_with_filter(
    jax_fn: Callable[..., Any],
    f: Callable[..., Any],
    predicate: filtering.Predicate,
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


def get_names(params: Params) -> Set[Text]:
  names = set([])
  for path, module in params.items():
    for name in module.keys():
      names.add("/".join([path, name]))
  return names


def to_set(params: Params) -> Set[Tuple[Text, Sequence[float]]]:
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


class PartitionTest(absltest.TestCase):

  def test_partitioning(self):

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

  def test_matching(self):

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

  def test_transforms_with_filer(self):
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
    df = df_fn(params, inputs)
    self.assertEqual(
        to_set(df),
        set([("first_layer/w", (3.0,)), ("second_layer/w", (2.5,))]))

    fn = jax_fn_with_filter(
        jax_fn=jax.value_and_grad,
        f=apply_fn,
        predicate=lambda module_name, name, _: name == "w")
    v = fn(params, inputs)
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
    jf = jf_fn(params, inputs)

    self.assertEqual(
        to_set(jf),
        set([("first_layer/w", (3.0, 6.0)), ("second_layer/w", (2.5, 5.0))]))

if __name__ == "__main__":
  absltest.main()
