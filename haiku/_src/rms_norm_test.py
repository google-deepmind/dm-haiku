# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for haiku._src.rms_norm."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import initializers
from haiku._src import rms_norm
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp
import numpy as np


class RMSNormTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_connection(self):
    data = jnp.zeros([2, 3, 4, 5])
    norms = []
    for axis in range(4):
      norms.append(rms_norm.RMSNorm(axis=axis)(data))

    norms.append(rms_norm.RMSNorm(axis=slice(1, None))(data))
    norms.append(rms_norm.RMSNorm(axis=slice(2, None))(data))
    norms.append(rms_norm.RMSNorm(axis=slice(1, -1))(data))

    return norms

  def test_bf16(self):
    """For all configurations, ensure bf16 outputs from bf16 inputs."""
    def f(x):
      ln = rms_norm.RMSNorm(axis=-1)
      return ln(x)

    fwd = transform.transform(f)
    data = jnp.zeros([2, 3, 4, 5], dtype=jnp.bfloat16)
    params = fwd.init(jax.random.PRNGKey(428), data)
    bf16_params = jax.tree_map(lambda t: t.astype(jnp.bfloat16), params)
    self.assertEqual(fwd.apply(bf16_params, None, data).dtype, jnp.bfloat16)

  @test_utils.transform_and_run
  def test_simple_case(self):
    layer = rms_norm.RMSNorm([1, 2], eps=0.0)
    inputs = np.full(shape=[2, 3, 3, 5], fill_value=2.0)
    outputs = layer(inputs)
    for x in np.nditer(outputs):
      self.assertEqual(x, 1.0)

  @test_utils.transform_and_run
  def test_simple_case_with_scale(self):
    layer = rms_norm.RMSNorm(
        axis=[1, 2], eps=0.0, scale_init=initializers.Constant(0.5))
    inputs = np.full(shape=[2, 3, 3, 5], fill_value=2.0)
    outputs = layer(inputs)
    for x in np.nditer(outputs):
      self.assertEqual(x, 0.5)

  @test_utils.transform_and_run
  def test_zero_inputs(self):
    layer = rms_norm.RMSNorm([1, 2])
    inputs = np.zeros([2, 3, 3, 5])
    outputs = layer(inputs)
    for x in np.nditer(outputs):
      self.assertEqual(x, 0.0)

  @parameterized.named_parameters(("String", "foo"), ("ListString", ["foo"]))
  @test_utils.transform_and_run
  def test_invalid_axis(self, axis):
    with self.assertRaisesRegex(
        ValueError, "`axis` should be an int, slice or iterable of ints."):
      rms_norm.RMSNorm(axis)

  @test_utils.transform_and_run
  def test_slice_axis(self):
    slice_layer = rms_norm.RMSNorm(slice(1, -1))
    axis_layer = rms_norm.RMSNorm((1, 2))
    inputs = np.random.uniform(size=[3, 4, 4, 5], low=0, high=10)

    slice_outputs = slice_layer(inputs)
    axis_outputs = axis_layer(inputs)

    np.testing.assert_array_equal(slice_outputs, axis_outputs)

if __name__ == "__main__":
  absltest.main()
