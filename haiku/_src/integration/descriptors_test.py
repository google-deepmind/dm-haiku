# Lint as: python3
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
"""Tests for haiku._src.conformance.descriptors."""

from absl.testing import absltest
import haiku as hk
from haiku._src import test_utils
from haiku._src.integration import descriptors

ALL_MODULES = descriptors.ALL_MODULES
IGNORED_MODULES = descriptors.IGNORED_MODULES


class DescriptorsTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_coverage(self):
    all_modules = frozenset(test_utils.find_subclasses(hk, hk.Module))
    tested_modules = {type(descriptors.unwrap(d.create())) for d in ALL_MODULES}
    self.assertEmpty(all_modules - (tested_modules | IGNORED_MODULES))


if __name__ == '__main__':
  absltest.main()
