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
"""Asserts all public symbols are covered in the docs."""

from typing import Any, Mapping

import haiku as hk
from haiku._src import test_utils
from sphinx import application
from sphinx import builders
from sphinx import errors

HIDDEN_SYMBOLS = ("haiku.experimental.GetterContext",
                  "haiku.experimental.MethodContext",
                  "haiku.experimental.ParamContext",
                  "haiku.experimental.custom_creator",
                  "haiku.experimental.custom_getter",
                  "haiku.experimental.intercept_methods",
                  "haiku.experimental.lift")


def haiku_public_symbols():
  names = set()
  for module_name, module in test_utils.find_internal_python_modules(hk):
    for name in module.__all__:
      symbol_name = f"{module_name}.{name}"
      if symbol_name not in HIDDEN_SYMBOLS:
        names.add(symbol_name)
  return names


class HaikuCoverageCheck(builders.Builder):
  """Builder that checks all public symbols are included."""

  name = "coverage_check"

  def get_outdated_docs(self) -> str:
    return "coverage_check"

  def write(self, *ignored: Any) -> None:
    pass

  def finish(self) -> None:
    documented_objects = frozenset(self.env.domaindata["py"]["objects"])
    undocumented_objects = haiku_public_symbols() - documented_objects
    if undocumented_objects:
      undocumented_objects = tuple(sorted(undocumented_objects))
      raise errors.SphinxError(
          "All public symbols must be included in our documentation, did you "
          "forget to add an entry to `api.rst`?\n"
          f"Undocumented symbols: {undocumented_objects}")


def setup(app: application.Sphinx) -> Mapping[str, Any]:
  app.add_builder(HaikuCoverageCheck)
  return dict(version="0.0.1", parallel_read_safe=True)
