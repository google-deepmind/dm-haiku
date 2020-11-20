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

import numpy as np


class MultiHeadAttentionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("batch & seq len = 1", 1, 1, 3, 5, 7),
      ("batch & seq len > 1", 2, 3, 5, 7, 11),
  )
  @test_utils.transform_and_run
  def test_shapes(self, batch_size, seq_len, embed_size, d_key, num_heads):
    query = key = value = np.zeros((batch_size, seq_len, embed_size))
    mha = attention.MultiHeadAttention(d_key, num_heads, 1.0)(query, key, value)
    self.assertEqual(mha.shape, (batch_size, seq_len, d_key * num_heads))


if __name__ == "__main__":
  absltest.main()
