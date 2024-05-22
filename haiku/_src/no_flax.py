# Copyright 2024 DeepMind Technologies Limited. All Rights Reserved.
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
"""Module for when flax is missing."""

import sys


class FlaxNotInstalled:
  __slots__ = ()

  def __getattr__(self, name: str) -> None:
    raise ImportError(
        '`haiku.experimental.flax` features require `flax` to be installed.'
    )


def inject_shim_module():
  # c.f. https://mail.python.org/pipermail/python-ideas/2012-May/014969.html
  sys.modules['haiku.experimental.flax'] = FlaxNotInstalled()
