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
"""Haiku is a neural network library for JAX."""

from haiku._src.initializers import Constant
from haiku._src.initializers import Orthogonal
from haiku._src.initializers import RandomNormal
from haiku._src.initializers import RandomUniform
from haiku._src.initializers import TruncatedNormal
from haiku._src.initializers import UniformScaling
from haiku._src.initializers import VarianceScaling
from haiku._src.typing import Initializer


__all__ = (
    "Constant",
    "Orthogonal",
    "Initializer",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "UniformScaling",
    "VarianceScaling",
)
