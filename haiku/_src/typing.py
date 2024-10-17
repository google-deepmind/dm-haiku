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
"""Haiku types."""

import abc
from collections.abc import Callable, Mapping, MutableMapping, Sequence
import typing
from typing import Any, Protocol, runtime_checkable

import jax

# pytype: disable=module-attr
try:
  # Using PyType's experimental support for forward references.
  Module = typing._ForwardRef("haiku.Module")  # pylint: disable=protected-access
except AttributeError:
  Module = Any
# pytype: enable=module-attr

Initializer = Callable[[Sequence[int], Any], jax.Array]
Params = Mapping[str, Mapping[str, jax.Array]]
MutableParams = MutableMapping[str, MutableMapping[str, jax.Array]]
State = Mapping[str, Mapping[str, jax.Array]]
MutableState = MutableMapping[str, MutableMapping[str, jax.Array]]

# Missing JAX types.
PRNGKey = jax.Array  # pylint: disable=invalid-name


class LiftingModuleType:
  """Parent type of lift.LiftingModule, added here to solve circular dependency."""


class StrictProtocol(Protocol):

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if Protocol not in cls.__bases__:
      base_names = ", ".join(b.__name__ for b in cls.__bases__)
      raise TypeError(
          f"{cls.__name__} is a Protocol and should not be subclassed by "
          "a non-Protocol type. If you intended your subclass to be a "
          "protocol then you need to explicitly additionally extend "
          f"Protocol: `class {cls.__name__}({base_names}, Protocol)`.")


@runtime_checkable
class ModuleProtocol(StrictProtocol, Protocol):
  """Protocol for Module like types."""

  name: str
  module_name: str

  @abc.abstractmethod
  def params_dict(self) -> Mapping[str, jax.Array]:
    raise NotImplementedError

  @abc.abstractmethod
  def state_dict(self) -> Mapping[str, jax.Array]:
    raise NotImplementedError


@runtime_checkable
class SupportsCall(ModuleProtocol, Protocol):
  """Protocol for Module like types that are Callable.

  Being a protocol means you don't need to explicitly extend this type in order
  to support instance checks with it. For example, :class:`Linear` only extends
  :class:`Module`, however since it conforms (e.g. implements ``__call__``) to
  this protocol you can instance check using it::

  >>> assert isinstance(hk.Linear(1), hk.SupportsCall)
  """

  @abc.abstractmethod
  def __call__(self, *args, **kwargs):
    raise NotImplementedError
