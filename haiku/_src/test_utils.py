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
import os
import types
from typing import Callable, Generator, Optional, Sequence, Tuple, TypeVar

from absl.testing import parameterized
from haiku._src import transform
from jax import random

T = TypeVar("T")
Fn = Callable[..., T]


def transform_and_run(f: Optional[Fn] = None,
                      seed: Optional[int] = 42,
                      run_apply: bool = True,
                      jax_transform: Optional[Callable[[Fn], Fn]] = None) -> T:
  r"""Transforms the given function and runs init then (optionally) apply.

  Equivalent to:

  >>> def f(x):
  ...   return x
  >>> x = jnp.ones([])
  >>> rng = jax.random.PRNGKey(42)
  >>> f = hk.transform_with_state(f)
  >>> params, state = f.init(rng, x)
  >>> out = f.apply(params, state, rng, x)

  This function makes it very convenient to unit test Haiku:

  >>> class MyTest(unittest.TestCase):
  ...   @hk.testing.transform_and_run
  ...   def test_linear_output(self):
  ...     mod = hk.Linear(1)
  ...     out = mod(jnp.ones([1, 1]))
  ...     self.assertEqual(out.ndim, 2)

  It can also be combined with ``chex`` to test all pure/jit/pmap versions of a
  function:

  >>> class MyTest(unittest.TestCase):
  ...   @chex.all_variants
  ...   def test_linear_output(self):
  ...     @hk.testing.transform_and_run(jax_transform=self.variant)
  ...     def f(inputs):
  ...       mod = hk.Linear(1)
  ...       return mod(inputs)
  ...     out = f(jnp.ones([1, 1]))
  ...     self.assertEqual(out.ndim, 2)

  And can also be useful in an interactive environment like ipython, Jupyter or
  Google Colaboratory:

  >>> f = lambda x: hk.Bias()(x)
  >>> hk.testing.transform_and_run(f)(jnp.ones([1, 1]))
  DeviceArray([[1.]], dtype=float32)

  See :func:`transform` for more details.

  Args:
    f: A function method to transform.
    seed: A seed to pass to init and apply.
    run_apply: Whether to run apply as well as init. Defaults to true.
    jax_transform: An optional jax transform to apply on the init and apply
      functions.

  Returns:
    A function that :func:`~haiku.transform`\ s ``f`` and runs ``init`` and
    optionally ``apply``.
  """
  if f is None:
    return functools.partial(
        transform_and_run,
        seed=seed,
        run_apply=run_apply,
        jax_transform=jax_transform)

  @functools.wraps(f)
  def wrapper(*a, **k):
    """Runs init and apply of f."""
    rng = random.PRNGKey(seed) if seed is not None else None
    init, apply = transform.transform_with_state(lambda: f(*a, **k))
    if jax_transform:
      init, apply = map(jax_transform, (init, apply))
    params, state = init(rng)
    if run_apply:
      out, state = apply(params, state, rng)
      return out

  return wrapper


def find_internal_python_modules(
    root_module: types.ModuleType,
) -> Sequence[Tuple[str, types.ModuleType]]:
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


def named_bools(name) -> Sequence[Tuple[str, bool]]:
  """Returns a pair of booleans suitable for use with ``named_parameters``."""
  return (name, True), ("not_{}".format(name), False)


def named_range(name, stop: int) -> Sequence[Tuple[str, int]]:
  """Equivalent to `range()` but suitable for use with ``named_parameters``."""
  return tuple(((f"{name}_{i}", i) for i in range(stop)))


def with_environ(key: str, value: Optional[str]):
  """Runs the given test with envrionment variables set."""
  def set_env(new_value):
    if new_value is None:
      if key in os.environ:
        del os.environ[key]
    else:
      os.environ[key] = new_value

  def decorator(f):
    def wrapper(*a, **k):
      value_before = os.environ.get(key, None)
      set_env(value)
      try:
        return f(*a, **k)
      finally:
        set_env(value_before)
    return wrapper

  return decorator
