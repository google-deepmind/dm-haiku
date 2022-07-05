# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for replaceable_funcs."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import base
from haiku._src import test_utils
from haiku._src.integration import descriptors
import jax.numpy as jnp

FUNC_TO_DESCRIPTORS = {
    base.get_parameter: descriptors.ONLY_PARAMS_MODULES,
    base.get_state: descriptors.STATEFUL_MODULES,
    base.set_state: descriptors.STATEFUL_MODULES,
    base.next_rng_key: descriptors.NEXT_RNG_KEY_MODULES
}

# pylint: disable=g-complex-comprehension
TEST_CASES = [
    (
        f'{replaceable_function.__name__}_{d.name}',  # func_name+module_name
        d.create,  # module function from descriptor
        d.shape,  # shape of data expected
        d.dtype,  # data type (if specified)
        replaceable_function  # function to replace
    )
    for replaceable_function, descrs in FUNC_TO_DESCRIPTORS.items()
    for d in descrs
]
# pylint: enable=g-complex-comprehension


class ReplaceableFuncsTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(TEST_CASES)
  @test_utils.transform_and_run
  def test_replaceable_funcs(self, module_fn: descriptors.ModuleFn, shape,
                             dtype, replace_fn):

    def my_replacement(*_, **k):
      raise ValueError('my_error')

    replace_fn._replace(my_replacement)  # pytype: disable=attribute-error

    with self.assertRaisesRegex(ValueError, 'my_error'):
      init, _ = hk.transform(module_fn())
      init(717, jnp.ones(shape, dtype))

    replace_fn._reset()  # pytype: disable=attribute-error


if __name__ == '__main__':
  absltest.main()
