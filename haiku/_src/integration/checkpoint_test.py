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
"""Tests for Haiku checkpointing."""

import json
import os

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import test_utils
from haiku._src.integration import checkpoint_utils
from haiku._src.integration import descriptors

ModuleFn = descriptors.ModuleFn
HOW_TO_REGENERATE = """
You can regenerate checkpoints using the checkpoint_generate utility in this
folder. Set the --base_path flag to the checkpoint folder.
"""


class CheckpointTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(
      descriptors.with_name(descriptors.ALL_MODULES))
  def test_checkpoint_format(self, name, module_fn: ModuleFn, shape, dtype):
    descriptor = descriptors.ModuleDescriptor(name, module_fn, shape, dtype)
    cls = descriptors.module_type(descriptor.create)
    expected = checkpoint_utils.summarize(descriptor)
    file_path = os.path.join(
        "haiku/_src/integration/checkpoints/",
        descriptors.to_file_name(descriptor) + ".json")
    if not os.path.exists(file_path):
      expected_json = json.dumps(expected, indent=2)
      raise ValueError(f"Missing checkpoint file: {file_path}\n\n"
                       f"Expected:\n\n{expected_json}")

    with open(file_path, "r") as fp:
      actual = json.load(fp)

    self.assertEqual(expected, actual, msg=HOW_TO_REGENERATE)

if __name__ == "__main__":
  absltest.main()

