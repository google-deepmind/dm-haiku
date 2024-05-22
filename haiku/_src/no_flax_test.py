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

import sys

from absl.testing import absltest
from haiku._src import no_flax


def setUpModule():
  for name in list(sys.modules):
    if name.startswith("flax"):
      # Trigger import errors when someone tries to use flax.
      sys.modules[name] = no_flax.FlaxNotInstalled()
    elif name.startswith("haiku"):
      # Unload any imported haiku modules so we can reload it.
      del sys.modules[name]


def tearDownModule():
  for name in list(sys.modules):
    if name.startswith("flax") or name.startswith("haiku"):
      del sys.modules[name]


class NoFlaxTest(absltest.TestCase):

  def test_import_shim(self):
    # DOES NOT FAIL: Haiku should be importable, even if flax is missing.
    import haiku as hk  # pylint: disable=g-bad-import-order,g-import-not-at-top

    # Errors are only thrown when users try to use flax.
    with self.assertRaisesRegex(ImportError, "require `flax`"):
      hk.experimental.flax.lift  # pylint: disable=pointless-statement


if __name__ == "__main__":
  absltest.main()
