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
"""Tests for haiku._src.bias."""

from absl.testing import absltest
from haiku._src import bias
from haiku._src import test_utils
from haiku._src import transform
import jax.numpy as jnp
import numpy as np


class BiasTest(absltest.TestCase):

  @test_utils.transform_and_run
  def test_output_shape(self):
    mod = bias.Bias(output_size=(2 * 2,))
    with self.assertRaisesRegex(ValueError, "Input shape must be [(]-1, 4[)]"):
      mod(jnp.ones([2, 2, 2]))

  @test_utils.transform_and_run
  def test_output_size_valid(self):
    mod = bias.Bias(output_size=(2 * 2,))
    mod(jnp.ones([2, 2 * 2]))

  def test_bias_dims_scalar(self):
    def f():
      mod = bias.Bias(bias_dims=())
      return mod(jnp.ones([1, 2, 3, 4]))
    params = transform.transform(f).init(None)
    self.assertEmpty(params["bias"]["b"].shape)

  @test_utils.transform_and_run
  def test_bias_dims_custom(self):
    b, d1, d2, d3 = range(1, 5)
    def f():
      mod = bias.Bias(bias_dims=[1, 3])
      out = mod(jnp.ones([b, d1, d2, d3]))
      self.assertEqual(mod.bias_shape, (d1, 1, d3))
      return out
    f = transform.transform(f)
    params = f.init(None)
    out = f.apply(params)
    self.assertEqual(params["bias"]["b"].shape, (d1, 1, d3))
    self.assertEqual(out.shape, (b, d1, d2, d3))

  @test_utils.transform_and_run
  def test_bias_dims_negative_out_of_order(self):
    def f():
      mod = bias.Bias(bias_dims=[-1, -2])
      mod(jnp.ones([1, 2, 3]))
      self.assertEqual(mod.bias_shape, (2, 3))
    params = transform.transform(f).init(None)
    self.assertEqual(params["bias"]["b"].shape, (2, 3))

  @test_utils.transform_and_run
  def test_bias_dims_invalid(self):
    mod = bias.Bias(bias_dims=[1, 5])
    with self.assertRaisesRegex(ValueError,
                                "5 .* out of range for input of rank 3"):
      mod(jnp.ones([1, 2, 3]))

  @test_utils.transform_and_run
  def test_b_init_defaults_to_zeros(self):
    mod = bias.Bias()
    x = jnp.ones([1, 1])
    y = mod(x)
    np.testing.assert_allclose(y, x)

  @test_utils.transform_and_run
  def test_b_init_custom(self):
    mod = bias.Bias(b_init=jnp.ones)
    x = jnp.ones([1, 1])
    y = mod(x)
    np.testing.assert_allclose(y, x + 1)

  @test_utils.transform_and_run
  def test_name(self):
    mod = bias.Bias(name="foo")
    self.assertEqual(mod.name, "foo")

  @test_utils.transform_and_run
  def test_multiplier(self):
    mod = bias.Bias(b_init=jnp.ones)
    y = mod(jnp.ones([1, 1]), multiplier=-1)
    np.testing.assert_allclose(jnp.sum(y), 0)

if __name__ == "__main__":
  absltest.main()
