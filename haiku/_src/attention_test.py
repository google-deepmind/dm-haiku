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

  @parameterized.named_parameters(
      ("unbatched_queries", (), (2,), (2,), False),
      ("singleton_queries", (1,), (2,), (2,), False),
      ("value_batch", (1,), (1,), (2,), False),
      ("multiple_batch_axes", (2, 1), (1, 3), (2, 3), False),
      ("masked_batch_axes", (2, 1), (1, 3), (2, 3), True),
      ("broadcast_keys_and_values", (2,), (1,), (1,), True),
  )
  def test_broadcast_batch_dims(
      self, query_batch, key_batch, value_batch, use_mask
  ):
    def f(query, key, value, mask):
      return attention.MultiHeadAttention(
          key_size=3,
          num_heads=2,
          value_size=4,
          model_size=5,
          w_init=initializers.VarianceScaling(1.0),
      )(query, key, value, mask)

    init, apply = transform.transform(f)
    keys = jax.random.split(jax.random.PRNGKey(42), 4)
    query = jax.random.normal(keys[0], query_batch + (3, 6))
    key = jax.random.normal(keys[1], key_batch + (4, 6))
    value = jax.random.normal(keys[2], value_batch + (4, 7))
    batch_shape = jnp.broadcast_shapes(query_batch, key_batch, value_batch)
    expanded = tuple(
        jnp.broadcast_to(x, batch_shape + x.shape[-2:])
        for x in (query, key, value)
    )
    mask = None
    if use_mask:
      mask = jnp.broadcast_to(
          jnp.tril(jnp.ones((3, 4), dtype=bool)), batch_shape + (1, 3, 4)
      )
    params = init(keys[3], *expanded, mask)
    expected = apply(params, None, *expanded, mask)
    self.assertEqual(expected.shape, batch_shape + (3, 5))
    for apply_fn in (apply, jax.jit(apply)):
      actual = apply_fn(params, None, query, key, value, mask)
      self.assertEqual(actual.shape, expected.shape)
      self.assertTrue(jnp.allclose(actual, expected, atol=1e-6))

    # Initializing with broadcast inputs must produce the same parameter shapes.
    broadcast_params = init(keys[3], query, key, value, mask)
    for actual, expected_param in zip(
        jax.tree_util.tree_leaves(broadcast_params),
        jax.tree_util.tree_leaves(params),
    ):
      self.assertEqual(actual.shape, expected_param.shape)
      self.assertTrue(jnp.array_equal(actual, expected_param))

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
