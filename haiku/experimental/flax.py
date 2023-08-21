# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
# pylint: disable=g-importing-member
"""Tools for working with Haiku and Flax."""

from haiku._src.flax.flax_module import Module
from haiku._src.flax.transform_flax import lift
from haiku._src.flax.utils import flatten_flax_to_haiku

__all__ = (
    'flatten_flax_to_haiku',
    'Module',
    'lift',
)
