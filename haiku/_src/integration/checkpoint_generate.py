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
"""Generates checkpoint json files."""

import json
import os

from absl import app
from absl import flags
from haiku._src.integration import checkpoint_utils
from haiku._src.integration import descriptors

flags.DEFINE_string("base_dir", None, help="Base directory.")
FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  for descriptor in descriptors.ALL_MODULES:
    cls = descriptors.module_type(descriptor.create)
    file_name = descriptors.to_file_name(descriptor) + ".json"
    summary = checkpoint_utils.summarize(descriptor)
    with open(os.path.join(FLAGS.base_dir, file_name), "w") as fp:
      fp.write(json.dumps(summary, indent=2))
      fp.write("\n")

if __name__ == "__main__":
  flags.mark_flag_as_required("base_dir")
  app.run(main)
