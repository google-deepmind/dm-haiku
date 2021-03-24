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
from haiku._src import test_utils

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
  def test_default_sizes(self):
    mha = attention.MultiHeadAttention(
      key_size=3, num_heads=5, w_init_scale=1.0)
    self.assertEqual(mha.query_size, mha.key_size)
    self.assertEqual(mha.value_size, mha.key_size)
    self.assertEqual(mha.model_size, mha.key_size * mha.num_heads)


if __name__ == "__main__":
  absltest.main()
