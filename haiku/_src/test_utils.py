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
"""Testing utilities for Haiku."""

import functools
import inspect
import itertools
import types
from typing import Generator, Optional, Sequence, Text, Tuple, TypeVar

from absl.testing import parameterized
from haiku._src import transform
from jax import random

T = TypeVar("T")


def transform_and_run(f=None, seed: Optional[int] = 42, run_apply: bool = True):
  """Transforms the given function and runs init then (optionally) apply.

  Equivalent to:

  >>> def f(x):
  ...   return x

  >>> x = jnp.ones([])
  >>> rng = jax.random.PRNGKey(42)
  >>> f = hk.transform_with_state(f)
  >>> params, state = f.init(rng, x)
  >>> _ = f.apply(params, state, rng, x)

  See :func:`transform` for more details.

  Args:
    f: A test method to transform.
    seed: A seed to pass to init and apply.
    run_apply: Whether to run apply as well as init. Defaults to true.

  Returns:
    A function that transforms f and runs `init` and optionally `apply`.
  """
  if f is None:
    return functools.partial(transform_and_run, seed=seed, run_apply=run_apply)

  @functools.wraps(f)
  def wrapper(*a, **k):
    """Runs init and apply of f."""
    rng = random.PRNGKey(seed) if seed is not None else None
    transformed = transform.transform_with_state(lambda: f(*a, **k))
    params, state = transformed.init(rng)
    if run_apply:
      transformed.apply(params, state, rng)

  return wrapper


def find_internal_python_modules(
    root_module: types.ModuleType,
) -> Sequence[Tuple[Text, types.ModuleType]]:
  """Returns `(name, module)` for all Haiku submodules under `root_module`."""
  modules = set([(root_module.__name__, root_module)])
  visited = set()
  to_visit = [root_module]

  while to_visit:
    mod = to_visit.pop()
    visited.add(mod)

    for name in dir(mod):
      obj = getattr(mod, name)
      if inspect.ismodule(obj) and obj not in visited:
        if obj.__name__.startswith("haiku"):
          to_visit.append(obj)
          modules.add((obj.__name__, obj))

  return sorted(modules)


def find_subclasses(
    root_python_module: types.ModuleType,
    base_class: T,
) -> Generator[T, None, None]:
  """Recursively traverse modules finding subclasses of the given type."""
  seen = set()
  for _, module in find_internal_python_modules(root_python_module):
    for _, value in module.__dict__.items():
      if inspect.isclass(value) and issubclass(value, base_class):
        if value not in seen:
          seen.add(value)
          yield value


def combined_named_parameters(*parameters):
  """Combines multiple ``@parameterized.named_parameters`` compatible sequences.

  >>> foos = ("a_for_foo", "a"), ("b_for_foo", "b")
  >>> bars = ("c_for_bar", "c"), ("d_for_bar", "d")

  >>> @named_parameters(foos)
  ... def testFoo(self, foo):
  ...   assert foo in ("a", "b")

  >>> @combined_named_parameters(foos, bars):
  ... def testFooBar(self, foo, bar):
  ...   assert foo in ("a", "b")
  ...   assert bar in ("c", "d")

  Args:
    *parameters: A sequence of parameters that will be combined and be passed
      into ``parameterized.named_parameters``.

  Returns:
    A test generator to be handled by ``parameterized.TestGeneratorMetaclass``.
  """
  combine = lambda a, b: ("_".join((a[0], b[0])),) + a[1:] + b[1:]
  return parameterized.named_parameters(
      functools.reduce(combine, r) for r in itertools.product(*parameters))


def named_bools(name) -> Sequence[Tuple[Text, bool]]:
  """Returns a pair of booleans suitable for use with ``named_parameters``."""
  return (name, True), ("not_{}".format(name), False)
