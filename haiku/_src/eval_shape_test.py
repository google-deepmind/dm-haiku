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
"""Tests for haiku._src.eval_shape."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import basic
from haiku._src import eval_shape
from haiku._src import stateful
from haiku._src import test_utils
from haiku._src import transform
import jax
import jax.numpy as jnp


class EvalShapeTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_fast_eval_shape_dropout(self):
    f = lambda rng, x: basic.dropout(rng, 0.5, x)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones([1])
    y_slow = jax.eval_shape(f, rng, x)
    y_fast = eval_shape.fast_eval_shape(f, rng, x)
    self.assertEqual(y_slow, y_fast)

  def test_fast_eval_shape_fold_in(self):
    f = lambda rng, x: jax.random.fold_in(rng, 1)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones([1])
    y_slow = jax.eval_shape(f, rng, x)
    y_fast = eval_shape.fast_eval_shape(f, rng, x)
    self.assertEqual(y_slow, y_fast)

  def test_fast_eval_shape_already_transformed(self):
    f = transform.transform(lambda x: basic.Linear(20)(x))  # pylint: disable=unnecessary-lambda
    rng = jax.random.PRNGKey(0)
    x = jnp.ones([1, 12])
    # init_fn
    y_slow = jax.eval_shape(f.init, rng, x)
    y_fast = eval_shape.fast_eval_shape(f.init, rng, x)
    self.assertEqual(y_slow, y_fast)
    self.assertEqual(
        y_slow, {'linear': {'w': jax.ShapeDtypeStruct((12, 20), jnp.float32),
                            'b': jax.ShapeDtypeStruct((20,), jnp.float32)}})
    # apply_fn
    y_slow = jax.eval_shape(f.apply, y_slow, rng, x)
    y_fast = eval_shape.fast_eval_shape(f.apply, y_fast, rng, x)
    self.assertEqual(y_slow, y_fast)

  def test_fast_eval_shape_within_transform(self):
    def f(x):
      m = basic.Linear(20)
      y_slow = stateful.eval_shape(m, x)
      y_fast = eval_shape.fast_eval_shape(m, x)
      self.assertEqual(y_slow, y_fast)
      return m(x)

    f = transform.transform(f)
    rng = jax.random.PRNGKey(0)
    x = jnp.ones([1, 12])
    params = jax.eval_shape(f.init, rng, x)
    self.assertEqual(
        params, {'linear': {'w': jax.ShapeDtypeStruct((12, 20), jnp.float32),
                            'b': jax.ShapeDtypeStruct((20,), jnp.float32)}})
    jax.eval_shape(f.apply, params, rng, x)

if __name__ == '__main__':
  absltest.main()
