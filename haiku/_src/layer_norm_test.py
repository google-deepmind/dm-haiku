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
"""Tests for haiku._src.layer_norm."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import initializers
from haiku._src import layer_norm
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp
import numpy as np


def with_param_axis_error(f):
  @functools.wraps(f)
  def wrapper(*a, **k):
    old = layer_norm.ERROR_IF_PARAM_AXIS_NOT_EXPLICIT
    layer_norm.ERROR_IF_PARAM_AXIS_NOT_EXPLICIT = True
    try:
      return f(*a, **k)
    finally:
      layer_norm.ERROR_IF_PARAM_AXIS_NOT_EXPLICIT = old
  return wrapper


class LayerNormTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_connection(self):
    data = jnp.zeros([2, 3, 4, 5])
    normalize = (
        lambda a: layer_norm.LayerNorm(a, True, True, param_axis=-1)(data))

    normalize(0)
    normalize(1)
    normalize(2)
    normalize(3)
    normalize(slice(1, None))
    normalize(slice(2, None))
    normalize(slice(1, -1))

  @parameterized.parameters(itertools.product([True, False], repeat=3))
  def test_bf16(self, create_scale, create_offset, use_fast_variance):
    """For all configurations, ensure bf16 outputs from bf16 inputs."""

    def f(x):
      ln = layer_norm.LayerNorm(
          axis=-1,
          create_scale=create_scale,
          create_offset=create_offset,
          use_fast_variance=use_fast_variance,
          param_axis=-1)
      return ln(x)

    fwd = transform.transform(f)
    data = jnp.zeros([2, 3, 4, 5], dtype=jnp.bfloat16)
    params = fwd.init(jax.random.PRNGKey(428), data)
    bf16_params = jax.tree.map(lambda t: t.astype(jnp.bfloat16), params)
    self.assertEqual(fwd.apply(bf16_params, None, data).dtype, jnp.bfloat16)

  @parameterized.parameters(True, False)
  @test_utils.transform_and_run
  def test_simple_case(self, use_fast_variance):
    layer = layer_norm.LayerNorm([1, 2],
                                 create_scale=False,
                                 create_offset=False,
                                 use_fast_variance=use_fast_variance,
                                 param_axis=-1)
    inputs = np.ones([2, 3, 3, 5])

    outputs = layer(inputs)
    for x in np.nditer(outputs):
      self.assertEqual(x, 0.0)

  @parameterized.parameters(True, False)
  @test_utils.transform_and_run
  def test_simple_case_var(self, use_fast_variance):
    layer = layer_norm.LayerNorm([1, 2],
                                 create_scale=True,
                                 create_offset=True,
                                 scale_init=initializers.Constant(0.5),
                                 offset_init=initializers.Constant(2.0),
                                 use_fast_variance=use_fast_variance,
                                 param_axis=-1)

    inputs = np.ones([2, 3, 3, 5])

    outputs = layer(inputs)
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  @test_utils.transform_and_run
  def test_simple_case_tensor(self):
    layer = layer_norm.LayerNorm([1, 2],
                                 create_scale=False,
                                 create_offset=False,
                                 param_axis=-1)

    inputs = np.ones([2, 3, 3, 5])
    scale = np.full((5,), 0.5)
    offset = np.full((5,), 2.0)

    outputs = layer(inputs, scale, offset)
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  @parameterized.named_parameters(("String", "foo"), ("ListString", ["foo"]))
  @test_utils.transform_and_run
  def test_invalid_axis(self, axis):
    with self.assertRaisesRegex(
        ValueError, "`axis` should be an int, slice or iterable of ints."):
      layer_norm.LayerNorm(axis, create_scale=False, create_offset=False)

  @test_utils.transform_and_run
  def test_no_scale_and_init_provided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `scale_init` if `create_scale=False`."):
      layer_norm.LayerNorm(
          3, create_scale=False, create_offset=True, scale_init=np.ones)

  @test_utils.transform_and_run
  def test_no_offset_beta_init_provided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `offset_init` if `create_offset=False`."):
      layer_norm.LayerNorm(
          3, create_scale=True, create_offset=False, offset_init=np.zeros)

  @test_utils.transform_and_run
  def test_create_scale_and_scale_provided(self):
    layer = layer_norm.LayerNorm([2], create_scale=True, create_offset=False)

    with self.assertRaisesRegex(
        ValueError, "Cannot pass `scale` at call time if `create_scale=True`."):
      layer(np.ones([2, 3, 4]), scale=np.ones([4]))

  @test_utils.transform_and_run
  def test_create_offset_and_offset_provided(self):
    layer = layer_norm.LayerNorm([2], create_offset=True, create_scale=False)

    with self.assertRaisesRegex(
        ValueError,
        "Cannot pass `offset` at call time if `create_offset=True`."):
      layer(np.ones([2, 3, 4]), offset=np.ones([4]))

  @parameterized.parameters(True, False)
  @test_utils.transform_and_run
  def test_slice_axis(self, use_fast_variance):
    slice_layer = layer_norm.LayerNorm(
        slice(1, -1),
        create_scale=False,
        create_offset=False,
        use_fast_variance=use_fast_variance,
        param_axis=-1)
    axis_layer = layer_norm.LayerNorm((1, 2),
                                      create_scale=False,
                                      create_offset=False,
                                      use_fast_variance=use_fast_variance,
                                      param_axis=-1)

    inputs = np.random.uniform(size=[3, 4, 4, 5], low=0, high=10)
    scale = np.random.normal(size=(5,), loc=1.0)
    offset = np.random.normal(size=(5,))

    slice_outputs = slice_layer(inputs, scale, offset)
    axis_outputs = axis_layer(inputs, scale, offset)

    np.testing.assert_array_equal(slice_outputs, axis_outputs)

  @test_utils.transform_and_run
  def test_connection_instance_norm(self):
    layer = layer_norm.InstanceNorm(create_scale=True, create_offset=True)

    inputs = np.ones([3, 4, 5, 6])
    result = layer(inputs)

    self.assertEqual(result.shape, (3, 4, 5, 6))

  @test_utils.transform_and_run
  def test_param_axis_not_required_for_final_axis(self):
    ln = layer_norm.LayerNorm(-1, True, True)
    x = jnp.ones([3, 4, 5, 6])
    ln(x)
    self.assertEqual(ln.params_dict()["layer_norm/scale"].shape, (6,))
    self.assertEqual(ln.params_dict()["layer_norm/offset"].shape, (6,))

  @test_utils.transform_and_run
  def test_error_prone_param_axis(self):
    # NOTE: This test defends current, potentially error prone behaviour
    # (passing axis!=-1 and not passing param_axis). It will be removed in a
    # future version of Haiku.
    ln = layer_norm.LayerNorm(1, True, True)
    x = jnp.ones([3, 4, 5, 6])
    ln(x)
    self.assertEqual(ln.params_dict()["layer_norm/scale"].shape, (6,))
    self.assertEqual(ln.params_dict()["layer_norm/offset"].shape, (6,))

  @parameterized.parameters(0, 1, 2, ((0, 1),), ((0, 1, 2),), -2, -3, -4,
                            slice(0, 2))
  @test_utils.transform_and_run
  @with_param_axis_error
  def test_param_axis_required_for_non_final_axis(self, axis):
    ln = layer_norm.LayerNorm(axis, True, True)
    x = jnp.ones([3, 4, 5, 6])
    with self.assertRaisesRegex(ValueError, "pass.*param_axis.*in the ctor"):
      ln(x)

  @parameterized.parameters(
      (-1, (6,)),
      (-2, (1, 1, 5, 1)),
      (-3, (1, 4, 1, 1)),
      (-4, (3, 1, 1, 1)),
      (0, (3, 1, 1, 1)),
      (1, (1, 4, 1, 1)),
      (2, (1, 1, 5, 1)),
      (3, (6,)),
  )
  @test_utils.transform_and_run
  def test_param_axis_sets_param_shape(self, param_axis, param_shape):
    ln = layer_norm.LayerNorm(-1, True, True, param_axis=param_axis)
    x = jnp.ones([3, 4, 5, 6])
    ln(x)
    self.assertEqual(ln.params_dict()["layer_norm/scale"].shape, param_shape)
    self.assertEqual(ln.params_dict()["layer_norm/offset"].shape, param_shape)

  @parameterized.parameters(
      ((0, 1, 2), (3, 4, 5, 1)),
      ((-4, -2, -3), (3, 4, 5, 1)),
      ((0, 1), (3, 4, 1, 1)),
      ((0, 3), (3, 1, 1, 6)),
      ((-4, -1), (3, 1, 1, 6)),
      ((-1, -4), (3, 1, 1, 6)),
  )
  @test_utils.transform_and_run
  def test_multiple_param_axis(self, param_axis, param_shape):
    ln = layer_norm.LayerNorm(-1, True, True, param_axis=param_axis)
    x = jnp.ones([3, 4, 5, 6])
    ln(x)
    self.assertEqual(ln.params_dict()["layer_norm/scale"].shape, param_shape)
    self.assertEqual(ln.params_dict()["layer_norm/offset"].shape, param_shape)


if __name__ == "__main__":
  absltest.main()
