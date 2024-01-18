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
"""Ensures that code samples in Haiku are accurate."""

import collections
import contextlib
import doctest
import inspect
import itertools
import types
import unittest

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax.linen as nn
import haiku as hk
from haiku._src import test_utils
import jax
import jax.numpy as jnp
import jmp


class DoctestTest(parameterized.TestCase):

  @parameterized.named_parameters(test_utils.find_internal_python_modules(hk))
  def test_doctest(self, module):
    def run_test():
      num_failed, num_attempted = doctest.testmod(
          module,
          optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
          extraglobs={
              "itertools": itertools,
              "chex": chex,
              "collections": collections,
              "contextlib": contextlib,
              "unittest": unittest,
              "hk": hk,
              "jnp": jnp,
              "jax": jax,
              "jmp": jmp,
              "nn": nn,
          })
      tests_symbols = ", ".join(module.__test__.keys())
      if num_attempted == 0:
        logging.info("No doctests in %s", tests_symbols)
      self.assertEqual(num_failed, 0, f"{num_failed} doctests failed")
      logging.info("%s tests passed in %s", num_attempted, tests_symbols)

    # `hk` et al import all dependencies from `src`, however doctest does not
    # test imported deps so we must manually set `__test__` such that imported
    # symbols are tested.
    # See: docs.python.org/3/library/doctest.html#which-docstrings-are-examined
    if not hasattr(module, "__test__") or not module.__test__:
      module.__test__ = {}

    # Many tests expect to be run as part of an `hk.transform`. We loop over all
    # exported symbols and run them in their own `hk.transform` so parameter and
    # module names don't clash.
    for name in module.__all__:
      test_names = []

      value = getattr(module, name)
      if inspect.ismodule(value):
        continue

      # Skip type annotations in Python 3.7.
      if hasattr(value, "__origin__"):
        continue

      # Skip broken tests.
      if name == "flatten_flax_to_haiku":
        self.skipTest("broken test")

      logging.info("Testing name: %r value: %r", name, value)
      if inspect.isclass(value) and not isinstance(value, types.GenericAlias):
        # Find unbound methods on classes, doctest doesn't seem to find them.
        test_names.append(name)
        module.__test__[name] = value

        for attr_name in dir(value):
          attr_value = getattr(value, attr_name)
          if inspect.isfunction(attr_value):
            test_name = name + "_" + attr_name
            test_names.append(test_name)
            module.__test__[test_name] = attr_value
      elif (isinstance(value, str) or inspect.isfunction(value) or
            inspect.ismethod(value) or inspect.isclass(value)):
        test_names.append(name)
        module.__test__[name] = value
      elif hasattr(value, "__doc__"):
        test_names.append(name)
        module.__test__[name] = value.__doc__
      else:
        # This will probably fail, DocTestFinder.find: __test__ values must be
        # strings, functions, methods, classes, or modules
        test_names.append(name)
        module.__test__[name] = value

      init_fn, _ = hk.transform_with_state(run_test)
      rng = jax.random.PRNGKey(42)
      init_fn(rng)

      for test_name in test_names:
        del module.__test__[test_name]


if __name__ == "__main__":
  absltest.main()
