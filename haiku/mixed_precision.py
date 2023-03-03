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
# pylint: disable=g-importing-member
"""Utilities for mixed precision."""

from haiku._src.mixed_precision import clear_policy
from haiku._src.mixed_precision import current_policy
from haiku._src.mixed_precision import get_policy
from haiku._src.mixed_precision import push_policy
from haiku._src.mixed_precision import set_policy

__all__ = (
    'clear_policy',
    'current_policy',
    'get_policy',
    'push_policy',
    'set_policy',
)
