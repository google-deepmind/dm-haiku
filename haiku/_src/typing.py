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
"""Haiku types."""

from typing import Any, Callable, Mapping, Text, Sequence, Union
import jax.numpy as jnp

Shape = Sequence[int]
DType = Any
ParamName = Text
Initializer = Callable[[Shape, DType], jnp.ndarray]
Params = Mapping[Text, Mapping[ParamName, jnp.ndarray]]
State = Mapping[Text, Mapping[Text, jnp.ndarray]]
NextCreator = Callable[[ParamName, Shape, DType, Initializer], jnp.ndarray]
ParamCreator = Callable[[NextCreator, ParamName, Shape, DType, Initializer],
                        jnp.ndarray]
Padding = Callable[[int], Sequence[int]]
Paddings = Union[Padding, Sequence[Padding]]

# Missing JAX types.
PRNGKey = jnp.ndarray  # pylint: disable=invalid-name
PRNGSeed = int
