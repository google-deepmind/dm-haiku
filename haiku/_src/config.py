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

import dataclasses
import threading


@dataclasses.dataclass
class Config:
  check_jax_usage: bool


def default_config() -> Config:
  return Config(check_jax_usage=False)

# We keep a reference to the Config for the importing thread (assumed to be the
# main thread in the process) such that other threads can inherit values set for
# it when they first request the config.
main_thread_config = default_config()


class ThreadLocalStorage(threading.local):

  def __init__(self):
    super().__init__()
    self.config = Config(**dataclasses.asdict(main_thread_config))

tls = ThreadLocalStorage()
tls.config = main_thread_config


def get_config() -> Config:
  return tls.config


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

  >>> hk.vmap(f, axis_size=2)()
  DeviceArray([0., 0.], dtype=float32)

  Args:
    enabled: Boolean indicating whether usage should be checked or not.

  Returns:
    Boolean with the previous value for this setting.
  """
  config = get_config()
  previous_value, config.check_jax_usage = config.check_jax_usage, enabled
  return previous_value
