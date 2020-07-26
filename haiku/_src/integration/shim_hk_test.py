# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for Haiku shim imports."""

import importlib
import types

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk


def named_internal_modules():
  for module_name in ("haiku._src", "haiku._src.nets"):
    for name in dir(importlib.import_module(module_name)):
      if not name.startswith("_"):
        submodule_name = module_name + "." + name
        yield submodule_name, importlib.import_module(submodule_name)


class ShimHkTest(parameterized.TestCase):

  @parameterized.named_parameters(*named_internal_modules())
  def test_hk_shim(self, module):
    if not hasattr(module, "hk"):
      self.skipTest(f"No `hk` in {module}")

    shim_hk = module.hk
    for name in dir(shim_hk):
      if name.startswith("_"):
        continue

      shim_value = getattr(shim_hk, name)
      if not hasattr(hk, name):
        raise ValueError(f"`hk.{name}` is not part of the actual Haiku API")

      actual_value = getattr(hk, name)
      if isinstance(shim_value, types.ModuleType):
        assert isinstance(actual_value, types.ModuleType)
      else:
        assert actual_value == shim_value

if __name__ == "__main__":
  absltest.main()
