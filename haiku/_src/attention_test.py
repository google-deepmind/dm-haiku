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
"""Tests for haiku._src.attention."""

from absl.testing import absltest
from absl.testing import parameterized

from haiku._src import attention
from haiku._src import initializers
from haiku._src import test_utils
from haiku._src import transform

import jax
import jax.numpy as jnp


class MultiHeadAttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("batch = 1 & seq len = 1", 1, 1, 3, 5, 7, 11, 13),
      ("batch = 1 & seq len > 1", 1, 2, 3, 5, 7, 11, 13),
      ("batch > 1 & seq len > 1", 2, 3, 5, 7, 11, 13, 17),
  )
  @test_utils.transform_and_run
  def test_shapes_batch(
      self, batch_size, seq_len, embed_size, d_key, num_heads, d_value, d_out):
    query = key = value = jnp.zeros((batch_size, seq_len, embed_size))
    mha = attention.MultiHeadAttention(
        key_size=d_key, num_heads=num_heads, value_size=d_value,
        model_size=d_out, w_init_scale=1.0)(query, key, value)
    self.assertEqual(mha.shape, (batch_size, seq_len, d_out))

  @parameterized.named_parameters(
      ("seq len = 1", 1, 2, 3, 5, 7, 11),
      ("seq len > 1", 2, 3, 5, 7, 11, 13),
  )
  @test_utils.transform_and_run
  def test_shapes_single(
      self, seq_len, embed_size, d_key, num_heads, d_value, d_out):
    query = key = value = jnp.zeros((seq_len, embed_size))
    mha = attention.MultiHeadAttention(
        key_size=d_key, num_heads=num_heads, value_size=d_value,
        model_size=d_out, w_init_scale=1.0)(query, key, value)
    self.assertEqual(mha.shape, (seq_len, d_out))

  @test_utils.transform_and_run
  def test_mask_arg(self):
    seq_len = 3
    embed_size = 2
    model_size = 15
    query = key = value = jnp.zeros((seq_len, embed_size))
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    causal_mask = causal_mask[None, :, :]

    mha = attention.MultiHeadAttention(
        key_size=7, num_heads=11, value_size=13,
        model_size=model_size, w_init_scale=1.0)(
            query, key, value, mask=causal_mask)
    self.assertEqual(mha.shape, (seq_len, model_size))

  @test_utils.transform_and_run
  def test_different_seq_lengths(self):
    query = jnp.zeros((2, 3))
    key = value = jnp.zeros((5, 3))
    mha = attention.MultiHeadAttention(
        key_size=7, num_heads=11, value_size=13,
        model_size=15, w_init_scale=1.0)(query, key, value)
    self.assertEqual(mha.shape, (2, 15))

  @test_utils.transform_and_run
  def test_default_sizes(self):
    mha = attention.MultiHeadAttention(
        key_size=3, num_heads=5, w_init_scale=1.0)
    self.assertEqual(mha.value_size, mha.key_size)
    self.assertEqual(mha.model_size, mha.key_size * mha.num_heads)

  def test_vmap(self):
    def f(query, key, value):
      return attention.MultiHeadAttention(
          key_size=3, num_heads=5, w_init_scale=1.0)(query, key, value)
    rng = jax.random.PRNGKey(42)
    init_rng, apply_rng, vmap_rng = jax.random.split(rng, num=3)
    init, apply = transform.transform(f)
    # Transform as single-instance function:
    query = key = value = jnp.zeros((7, 11))
    params = init(init_rng, query, key, value)
    y = apply(params, apply_rng, query, key, value)
    self.assertEqual(y.shape, (7, 15,))
    # Use vmap to get batched function:
    vapply = jax.vmap(apply, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    query = key = value = jnp.zeros((13, 7, 11))  # prepend batch axis
    rngs = jax.random.split(vmap_rng, 13)  # give each instance its own rng
    y = vapply(params, rngs, query, key, value)
    self.assertEqual(y.shape, (13, 7, 15))

  @test_utils.transform_and_run
  def test_w_init(self):

    with self.assertRaisesRegex(ValueError, "provide a weight initializer"):
      attention.MultiHeadAttention(2, 3)
    with self.assertRaisesRegex(ValueError, "provide only `w_init`"):
      attention.MultiHeadAttention(
          2, 3, w_init_scale=5, w_init=initializers.Constant(0))

    w_init = initializers.Constant(3)
    mha1 = attention.MultiHeadAttention(2, 3, w_init=w_init)
    self.assertIs(mha1.w_init, w_init)

    mha2 = attention.MultiHeadAttention(2, 3, w_init_scale=5)
    self.assertIsInstance(mha2.w_init, initializers.VarianceScaling)

  @test_utils.transform_and_run
  def test_b_init(self):

    w_init = initializers.Constant(3)
    b_init = initializers.Constant(4)
    mha1 = attention.MultiHeadAttention(2, 3, w_init=w_init, b_init=b_init)
    self.assertIs(mha1.b_init, b_init)

  @parameterized.named_parameters(
      ("with_bias_true", True, 2),
      ("with_bias_false", False, 1),
  )
  def test_with_bias(self, with_bias, expected_params):
    def f(key, query, value):
      w_init = initializers.Constant(3)
      mha1 = attention.MultiHeadAttention(2, 3, w_init=w_init,
                                          with_bias=with_bias)
      return mha1(key, query, value)

    rng = jax.random.PRNGKey(42)
    init, _ = transform.transform(f)
    query = key = jnp.zeros((5, 3))
    value = jnp.zeros((5, 10))
    params = init(rng, key, query, value)
    for module_params in params.values():
      self.assertLen(module_params, expected_params)


if __name__ == "__main__":
  absltest.main()
