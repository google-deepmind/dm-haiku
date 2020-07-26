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

import typing
from typing import Any, Callable, Mapping, Sequence
import jax.numpy as jnp

# pytype: disable=module-attr
try:
  # Using PyType's experimental support for forward references.
  Module = typing._ForwardRef("haiku.Module")  # pylint: disable=protected-access
except AttributeError:
  Module = Any
# pytype: enable=module-attr

Initializer = Callable[[Sequence[int], Any], jnp.ndarray]
Params = Mapping[str, Mapping[str, jnp.ndarray]]
State = Mapping[str, Mapping[str, jnp.ndarray]]

# Missing JAX types.
PRNGKey = jnp.ndarray  # pylint: disable=invalid-name
