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

from collections.abc import Callable, Generator, Sequence
import functools
import inspect
import itertools
import os
import types
from typing import Any, TypeVar

from absl.testing import parameterized
from haiku._src import config
from haiku._src import transform
import jax

T = TypeVar("T")
Fn = Callable[..., T]
Key = Any  # NOTE: jax.random.PRNGKey is not actually a type.


def transform_and_run(
    f: Fn | None = None,
    seed: int | None = 42,
    run_apply: bool = True,
    jax_transform: Callable[[Fn], Fn] | None = None,
    *,
    map_rng: Callable[[Key], Key] | None = None,
) -> T:
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
  >>> print(hk.testing.transform_and_run(f)(jnp.ones([1, 1])))
  [[1.]]

  See :func:`transform` for more details.

  To use this with `pmap` (without ``chex``) you need to additionally pass in a
  function to map the init/apply rng keys. For example, if you want every
  instance of your pmap to have the same key:

  >>> def same_key_on_all_devices(key):
  ...   return jnp.broadcast_to(key, (jax.local_device_count(), *key.shape))

  >>> @hk.testing.transform_and_run(jax_transform=jax.pmap,
  ...                               map_rng=same_key_on_all_devices)
  ... def test_something():
  ...   ...

  Or you can use a different key:

  >>> def different_key_on_all_devices(key):
  ...   return jax.random.split(key, jax.local_device_count())

  >>> @hk.testing.transform_and_run(jax_transform=jax.pmap,
  ...                               map_rng=different_key_on_all_devices)
  ... def test_something_else():
  ...   ...

  Args:
    f: A function method to transform.
    seed: A seed to pass to init and apply.
    run_apply: Whether to run apply as well as init. Defaults to true.
    jax_transform: An optional jax transform to apply on the init and apply
      functions.
    map_rng: If set to a non-None value broadcast the init/apply rngs
      broadcast_rng-ways.

  Returns:
    A function that :func:`~haiku.transform`\ s ``f`` and runs ``init`` and
    optionally ``apply``.
  """
  if f is None:
    return functools.partial(
        transform_and_run,
        seed=seed,
        run_apply=run_apply,
        jax_transform=jax_transform,
        map_rng=map_rng)

  @functools.wraps(f)
  def wrapper(*a, **k):
    """Runs init and apply of f."""
    if seed is not None:
      init_rng, apply_rng = (jax.random.PRNGKey(seed),
                             jax.random.PRNGKey(seed + 1))
      if map_rng is not None:
        init_rng, apply_rng = map(map_rng, (init_rng, apply_rng))
    else:
      init_rng, apply_rng = None, None
    init, apply = transform.transform_with_state(lambda: f(*a, **k))
    if jax_transform:
      init, apply = map(jax_transform, (init, apply))
    params, state = init(init_rng)
    if run_apply:
      out, state = apply(params, state, apply_rng)
      return out

  return wrapper


def find_internal_python_modules(
    root_module: types.ModuleType,
) -> Sequence[tuple[str, types.ModuleType]]:
  """Returns `(name, module)` for all Haiku submodules under `root_module`."""
  modules = {(root_module.__name__, root_module)}
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
      if not inspect.isclass(value) or isinstance(value, types.GenericAlias):
        continue
      if issubclass(value, base_class) and value not in seen:
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


def named_bools(name) -> Sequence[tuple[str, bool]]:
  """Returns a pair of booleans suitable for use with ``named_parameters``."""
  return (name, True), (f"not_{name}", False)


def named_range(name, stop: int) -> Sequence[tuple[str, int]]:
  """Equivalent to `range()` but suitable for use with ``named_parameters``."""
  return tuple((f"{name}_{i}", i) for i in range(stop))


def with_environ(key: str, value: str | None):
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


def with_guardrails(f):
  """Runs the given test with JAX guardrails on."""
  @functools.wraps(f)
  def wrapper(*a, **k):
    old = config.check_jax_usage(True)
    try:
      return f(*a, **k)
    finally:
      config.check_jax_usage(old)
  return wrapper


def clone(key):
  """Call jax.random.clone if it is available."""
  if hasattr(jax.random, "clone"):
    # JAX v0.4.26+
    return jax.random.clone(key)
  return key
