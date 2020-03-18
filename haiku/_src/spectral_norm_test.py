# Lint as: python3
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
"""Tests for haiku._src.spectral_norm."""

from absl.testing import absltest
from haiku._src import basic
from haiku._src import spectral_norm
from haiku._src import test_utils
from haiku._src import transform

import jax.numpy as jnp
import jax.random as random
import numpy as np


class SpectralNormTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_scalar(self):
    sn = spectral_norm.SpectralNorm()
    with self.assertRaisesRegex(ValueError, "not well defined"):
      sn(1.0)

  @test_utils.transform_and_run
  def test_vector(self):
    sn = spectral_norm.SpectralNorm()
    with self.assertRaisesRegex(ValueError, "not well defined"):
      sn(jnp.ones(shape=[5]))

  @test_utils.transform_and_run
  def test_3d_tensor(self):
    sn = spectral_norm.SpectralNorm()
    input_3d = (4.0 * jnp.eye(8, 8))[None, :, :]
    sn(input_3d)
    with self.assertRaisesRegex(ValueError, "Input is 3D but"):
      sn(input_3d, error_on_non_matrix=True)

  @test_utils.transform_and_run
  def test_matrix(self):
    sn = spectral_norm.SpectralNorm()
    # We can easily calculate the first singular value for this matrix.
    input_ = 4.0 * jnp.eye(8, 8)
    sn(input_)
    np.testing.assert_allclose(sn.sigma, 4.0, atol=1e-3)

  @test_utils.transform_and_run
  def test_matrix_multiple_steps(self):
    sn = spectral_norm.SpectralNorm(n_steps=3)
    # We can easily calculate the first singular value for this matrix.
    input_ = 4.0 * jnp.eye(8, 8)
    sn(input_)
    np.testing.assert_allclose(sn.sigma, 4.0, atol=1e-3)

  @test_utils.transform_and_run
  def test_matrix_no_stats(self):
    sn = spectral_norm.SpectralNorm()
    # We can easily calculate the first singular value for this matrix.
    input_ = 4.0 * jnp.eye(8, 8)
    sn(input_, update_stats=False)
    np.testing.assert_allclose(sn.sigma, 1.0)


class SNParamsTreeTest(absltest.TestCase):

  def test_sn_naming_scheme(self):
    sn_name = "this_is_a_wacky_but_valid_name"
    linear_name = "so_is_this"

    def f():
      return basic.Linear(output_size=2, name=linear_name)(jnp.zeros([6, 6]))

    init_fn, _ = transform.transform(f)
    params = init_fn(random.PRNGKey(428))

    def g(x):
      return spectral_norm.SNParamsTree(ignore_regex=".*b", name=sn_name)(x)
    init_fn, _ = transform.transform_with_state(g)
    _, params_state = init_fn(random.PRNGKey(428), params)

    expected_sn_states = [
        "{}/{}__{}".format(sn_name, linear_name, s) for s in ["w"]]
    self.assertSameElements(expected_sn_states, params_state.keys())


if __name__ == "__main__":
  absltest.main()
