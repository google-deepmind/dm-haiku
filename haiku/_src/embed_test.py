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
"""Tests for haiku._src.embed."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

from haiku._src import embed
from haiku._src import test_utils

import jax.numpy as jnp
import numpy as np


_EMBEDDING_MATRIX = np.asarray([
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5, 0.5],
    [0.1, 0.2, 0.3, 0.4]
])


_1D_IDS = [0, 2]  # pylint: disable=invalid-name
_2D_IDS = [[0, 2], [2, 2]]  # pylint: disable=invalid-name
_3D_IDS = [[[0, 2], [2, 2]], [[1, 1], [0, 2]]]  # pylint: disable=invalid-name


class EmbedTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(["ARRAY_INDEX", "ONE_HOT"],
                        [_1D_IDS, _2D_IDS, _3D_IDS]))
  @test_utils.transform_and_run
  def test_lookup(self, lookup_style, inp_ids):
    emb = embed.Embed(embedding_matrix=_EMBEDDING_MATRIX,
                      lookup_style=lookup_style)
    np.testing.assert_allclose(
        emb(inp_ids),
        jnp.asarray(_EMBEDDING_MATRIX)[jnp.asarray(inp_ids)])
    self.assertEqual(
        list(emb(inp_ids).shape),
        list(jnp.asarray(_EMBEDDING_MATRIX)[jnp.asarray(inp_ids)].shape))

  @parameterized.parameters("ARRAY_INDEX", "ONE_HOT")
  @test_utils.transform_and_run
  def test_default_creation(self, lookup_style):
    emb = embed.Embed(vocab_size=6, embed_dim=12, lookup_style=lookup_style)
    self.assertEqual(emb(_1D_IDS).shape, (2, 12))

  @test_utils.transform_and_run
  def test_no_creation_args(self):
    with self.assertRaisesRegex(ValueError, "must be supplied either with an"):
      embed.Embed()

  @test_utils.transform_and_run
  def test_inconsistent_creation_args(self):
    with self.assertRaisesRegex(ValueError, "supplied but the `vocab_size`"):
      embed.Embed(embedding_matrix=_EMBEDDING_MATRIX, vocab_size=4)
    with self.assertRaisesRegex(ValueError, "supplied but the `embed_dim`"):
      embed.Embed(embedding_matrix=_EMBEDDING_MATRIX, embed_dim=5)

  @test_utils.transform_and_run
  def test_embed_dtype_check(self):
    emb = embed.Embed(
        embedding_matrix=_EMBEDDING_MATRIX, lookup_style="ARRAY_INDEX")
    with self.assertRaisesRegex(
        ValueError,
        "hk.Embed's __call__ method must take an array of integer dtype but "
        "was called with an array of float32"):
      emb([1.0, 2.0])

  @test_utils.transform_and_run
  def test_embed_invalid_lookup(self):
    lookup_style = "FOO"
    emb = embed.Embed(embedding_matrix=_EMBEDDING_MATRIX, lookup_style="FOO")
    with self.assertRaisesRegex(
        ValueError, f"{lookup_style} is not a valid enum "
        f"in EmbedLookupStyle."):
      emb(_1D_IDS)

  @test_utils.transform_and_run
  def test_embed_property_check(self):
    lookup_style = "ONE_HOT"
    emb = embed.Embed(
        embedding_matrix=_EMBEDDING_MATRIX, lookup_style=lookup_style)

    self.assertEqual(emb.vocab_size, 3)
    self.assertEqual(emb.embed_dim, 4)
    np.testing.assert_allclose(
        emb.embeddings,
        jnp.asarray([[0., 0., 0., 0.], [0.5, 0.5, 0.5, 0.5],
                     [0.1, 0.2, 0.3, 0.4]]))


if __name__ == "__main__":
  absltest.main()
