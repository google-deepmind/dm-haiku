# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Configuration for Haiku."""

import contextlib
import dataclasses
import threading


@dataclasses.dataclass
class Config:
  check_jax_usage: bool
  module_auto_repr: bool
  restore_flatmap: bool
  rng_reserve_size: int

  @classmethod
  def default(cls) -> "Config":
    return Config(
        check_jax_usage=False,
        module_auto_repr=True,
        restore_flatmap=False,
        rng_reserve_size=1,
    )


def write(config, **overrides):
  for name, value in overrides.items():
    assert hasattr(config, name)
    setattr(config, name, value)

filter_none_values = lambda d: {k: v for k, v in d.items() if v is not None}


# pylint: disable=redefined-outer-name,unused-argument
def context(
    *,
    check_jax_usage: bool | None = None,
    module_auto_repr: bool | None = None,
    restore_flatmap: bool | None = None,
    rng_reserve_size: int | None = None,
):
  """Context manager for setting config options.

  This context manager can be used to override config settings in a given
  context, values that are not explicitly passed as keyword arguments retain
  their current value:

  >>> with hk.config.context(check_jax_usage=True):
  ...   pass

  Args:
    check_jax_usage: Checks that jax transforms and control flow are used
      appropriately in Haiku transformed functions.
    module_auto_repr: Can be used to disable the "to string" functionality that
      is part of Haiku's base contructor.
    restore_flatmap: Whether legacy checkpoints should be restored in the old
      FlatMap datatype (as returned by ``to_immtable_dict``), default is to
      restore these as plain dicts.
    rng_reserve_size: amount of keys to reserve when splitting off a key
      through ``next_rng_key()``, defaults to 1. Reserving larger blocks of keys
      can improve compilation and run-time of your model. Changing the
      reservation size will change RNG keys returned by ``next_rng_key``, and
      will change the generated random numbers.

  Returns:
    Context manager that applies the given configs while active.
  """
  return assign(**filter_none_values(locals()))
# pylint: enable=redefined-outer-name,unused-argument


# pylint: disable=redefined-outer-name,unused-argument,redefined-builtin
def set(
    *,
    check_jax_usage: bool | None = None,
    module_auto_repr: bool | None = None,
    restore_flatmap: bool | None = None,
    rng_reserve_size: int | None = None,
):
  """Sets the given config option(s).

  >>> hk.config.set(module_auto_repr=False)
  >>> hk.Linear(1)
  <...Linear object at ...>
  >>> hk.config.set(module_auto_repr=True)
  >>> hk.Linear(1)
  Linear(output_size=1)

  Args:
    check_jax_usage: Checks that jax transforms and control flow are used
      appropriately in Haiku transformed functions.
    module_auto_repr: Can be used to disable the "to string" functionality that
      is part of Haiku's base contructor.
    restore_flatmap: Whether legacy checkpoints should be restored in the old
      FlatMap datatype (as returned by ``to_immtable_dict``), default is to
      restore these as plain dicts.
    rng_reserve_size: amount of keys to reserve when splitting off a key
      through ``next_rng_key()``, defaults to 1. Reserving larger blocks of keys
      can improve compilation and run-time of your model. Changing the
      reservation size will change RNG keys returned by ``next_rng_key``, and
      will change the generated random numbers.
  """
  write(get_config(), **filter_none_values(locals()))
# pylint: enable=redefined-outer-name,unused-argument,redefined-builtin


@contextlib.contextmanager
def assign(**overrides):
  """Context manager used to override config settings."""
  config = get_config()
  previous = {name: getattr(config, name) for name in overrides}
  write(config, **overrides)
  try:
    yield
  finally:
    write(config, **previous)


def with_config(**overrides):
  """Decorator used to run a wrapped function with overriden config."""
  def decorator(f):
    def wrapper(*args, **kwargs):
      with assign(**overrides):
        return f(*args, **kwargs)
    return wrapper
  return decorator


# We keep a reference to the Config for the importing thread (assumed to be the
# main thread in the process) such that other threads can inherit values set for
# it when they first request the config.
main_thread_config = Config.default()


class ThreadLocalStorage(threading.local):

  def __init__(self):
    super().__init__()
    self.config = Config(**dataclasses.asdict(main_thread_config))

tls = ThreadLocalStorage()
tls.config = main_thread_config


def get_config() -> Config:
  return tls.config


def module_auto_repr(enabled: bool) -> bool:
  """Disables automatically generating an implementation of Module.__repr__.

  By default, Haiku will automatically generate a useful string representation
  of modules for printing. For example:

  >>> print(hk.Linear(1))
  Linear(output_size=1)

  In some cases, objects passed into module constructors may be slow to print,
  for example very nested data structures, or you may be rapidly creating and
  throwing away modules (e.g. in a test) and don't want to pay the overhead of
  converting to string.

  This config option enables users to disable the automatic repr feature
  globally in Haiku:

  >>> previous_value = hk.experimental.module_auto_repr(False)
  >>> print(hk.Linear(1))
  <...Linear object at ...>

  >>> previous_value = hk.experimental.module_auto_repr(True)
  >>> print(hk.Linear(1))
  Linear(output_size=1)

  To disable the feature on a per-subclass basis assign
  ``AUTO_REPR = False`` as a property on your class, for example:

  >>> class NoAutoRepr(hk.Module):
  ...   AUTO_REPR = False
  >>> print(NoAutoRepr())
  <...NoAutoRepr object at ...>

  Args:
    enabled: Boolean indicating whether a module should be enabled.

  Returns:
    The previous value of this config setting.
  """
  config = get_config()
  previous_value, config.module_auto_repr = config.module_auto_repr, enabled
  return previous_value


def check_jax_usage(enabled: bool = True) -> bool:
  """Ensures JAX APIs (e.g. :func:`jax.vmap`) are used correctly with Haiku.

  JAX transforms (like :func:`jax.vmap`) and control flow (e.g.
  :func:`jax.lax.cond`) expect pure functions to be passed in. Some functions
  in Haiku (for example :func:`~haiku.get_parameter`) have side effects and thus
  functions using them are only pure after using :func:`~haiku.transform` (et
  al).

  Sometimes it is convenient to use JAX transforms or control flow before
  transforming your function (for example, to :func:`~haiku.vmap` the
  application of a module) but when doing so you need to be careful to use the
  Haiku overloaded version of the underlying JAX function, which carefully makes
  the function(s) you pass in pure functions before calling the underlying JAX
  function.

  :func:`check_jax_usage` enables checking raw JAX transforms are used
  appropriately inside Haiku transformed functions. Incorrect usage of JAX
  transforms will result in an error.

  Consider the function below, it is not a pure function (a function of its
  inputs with no side effects) because we call into a Haiku API
  (:func:`~haiku.get_parameter`) which during init will create a parameter and
  register it with Haiku.

  >>> def f():
  ...   return hk.get_parameter("some_param", [], init=jnp.zeros)

  We should not use this with JAX APIs like :func:`jax.vmap` (because it is not
  a pure function). :func:`check_jax_usage` allows you to tell Haiku to make
  incorrect usages of JAX APIs an error:

  >>> previous_value = hk.experimental.check_jax_usage(True)
  >>> jax.vmap(f, axis_size=2)()
  Traceback (most recent call last):
    ...
  haiku.JaxUsageError: ...

  Using the Haiku wrapped version works correctly:

  >>> print(hk.vmap(f, axis_size=2, split_rng=False)())
  [0. 0.]

  Args:
    enabled: Boolean indicating whether usage should be checked or not.

  Returns:
    Boolean with the previous value for this setting.
  """
  config = get_config()
  previous_value, config.check_jax_usage = config.check_jax_usage, enabled
  return previous_value


def rng_reserve_size(size: int) -> int:
  """Change amount of RNG keys reserved when calling ``next_rng_key``.

  Args:
    size: amount of keys to reserve when splitting off a key
      through ``next_rng_key()``, defaults to 1. Reserving larger blocks of keys
      can improve compilation and run-time of your model. Changing the
      reservation size will change RNG keys returned by ``next_rng_key``, and
      will change the generated random numbers.
  Returns:
    The previous value of the rng_reserve_size setting.
  """
  if size <= 0:
    raise ValueError(f"RNG reserve size needs to be more than 0, got {size}.")
  config = get_config()
  before, config.rng_reserve_size = config.rng_reserve_size, size
  return before
