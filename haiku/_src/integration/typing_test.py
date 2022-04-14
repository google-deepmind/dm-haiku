# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Integration tests for Haiku typing."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors

ModuleFn = descriptors.ModuleFn


class TypingTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  @test_utils.transform_and_run
  def test_protocols(self, module_fn: ModuleFn, shape, dtype):
    del shape, dtype
    module = descriptors.unwrap(module_fn())
    self.assertIsInstance(module, hk.ModuleProtocol)
    # NOTE: All current Haiku builtin modules are callable.
    self.assertIsInstance(module, hk.SupportsCall)

if __name__ == '__main__':
  absltest.main()
