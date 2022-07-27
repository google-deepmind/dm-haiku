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
"""Tests for haiku._src.test_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import test_utils
import jax
import jax.numpy as jnp


class TestUtilsTest(parameterized.TestCase):

  @test_utils.transform_and_run(
      jax_transform=jax.pmap,
      map_rng=lambda k: jnp.broadcast_to(k, (1, *k.shape)))
  def test_transform_and_run_pmap(self):
    pass

  @test_utils.transform_and_run(
      jax_transform=lambda f: jax.pmap(f, 'i'),
      map_rng=lambda k: jnp.broadcast_to(k, (1, *k.shape)))
  def test_transform_and_run_pmap_with_axis(self):
    pass

if __name__ == '__main__':
  absltest.main()
