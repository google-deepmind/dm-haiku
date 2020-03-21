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
"""Modules for performing embedding lookups in Haiku."""

import enum

from haiku._src import base
from haiku._src import basic
from haiku._src import module
import haiku._src.initializers as hk_init
import jax.numpy as jnp


class EmbedLookupStyle(enum.Enum):
  """How to return the embedding matrices given IDs."""
  ARRAY_INDEX = 1
  ONE_HOT = 2


class Embed(module.Module):
  """Module for embedding tokens in a low-dimensional space."""

  def __init__(self,
               vocab_size=None,
               embed_dim=None,
               embedding_matrix=None,
               w_init=None,
               lookup_style=EmbedLookupStyle.ARRAY_INDEX.name,
               name=None):
    """Constructs an Embed module.

    Args:
      vocab_size: int or None: the number of unique tokens to embed. If not
        provided, an existing vocabulary matrix from which vocab_size can be
        inferred must be provided as `existing_vocab`.
      embed_dim: int or None. Number of dimensions to assign to each embedding.
        If an existing vocabulary matrix initializes the module, this should not
        be provided as it will be inferred.
      embedding_matrix: A matrix-like object equivalent in size to
        [vocab_size, embed_dim]. If given, it is used as the initial value for
        the embedding matrix and neither vocab_size or embed_dim need be given.
        If they are given, their values are checked to be consistent with the
        dimensions of embedding_matrix.
      w_init: An initializer for the embeddings matrix. As a default,
        embeddings are initialized via a truncated normal distribution.
      lookup_style: One of the enum values of EmbedLookupStyle determining how
        to access the value of the embbeddings given an ID. Regardless the input
        should be a dense array of integer values representing ids. This setting
        changes how internally this module maps those ides to embeddings. The
        result is the same, but the speed and memory tradeoffs are different.
        It default to using numpy-style array indexing. This value is only the
        default for the module, and at any given invocation can be overriden
        in the __call__ method.
      name: string. Name for this module.

    Raise:
      ValueError: If none of embed_dim, embedding_matrix and vocab_size are
        supplied, or if embedding_matrix is supplied and embed_dim or vocab_size
        is not consistent with the supplied matrix.
    """
    super(Embed, self).__init__(name=name)
    if embedding_matrix is None and not (vocab_size and embed_dim):
      raise ValueError(
          "hk.Embed must be supplied either with an initial `embedding_matrix` "
          "or with `embed_dim` and `vocab_size`.")
    if embedding_matrix is not None:
      embedding_matrix = jnp.asarray(embedding_matrix)
      if vocab_size and embedding_matrix.shape[0] != vocab_size:
        raise ValueError(
            "An `embedding_matrix` was supplied but the `vocab_size` of {vs} "
            "was not consistent with its shape {emb_shape}.".format(
                vs=vocab_size, emb_shape=embedding_matrix.shape))
      if embed_dim and embedding_matrix.shape[1] != embed_dim:
        raise ValueError(
            "An `embedding_matrix` was supplied but the `embed_dim` of {ed} "
            "was not consistent with its shape {emb_shape}.".format(
                ed=embed_dim, emb_shape=embedding_matrix.shape))
      self._embedding = base.get_parameter(
          "embeddings", shape=embedding_matrix.shape,
          init=lambda _, __: embedding_matrix)
    else:
      w_init = w_init or hk_init.TruncatedNormal()
      self._embedding = base.get_parameter(
          "embeddings", shape=[vocab_size, embed_dim], init=w_init)

    self._vocab_size = vocab_size or embedding_matrix.shape[0]
    self._embed_dim = embed_dim or embedding_matrix.shape[1]
    self._lookup_style = lookup_style

  def __call__(self, ids, lookup_style=None):
    """Lookup embeddings.

    Looks up an embedding vector for each value in `ids`. All ids must be within
    [0, vocab_size) to prevent NaNs from propagating.

    Args:
      ids: Tensor of dtype int64.
      lookup_style: Overrides the lookup_style given in the constructor.

    Returns:
      Tensor of tf.shape(ids) + [embedding_dim] and dtype float32.

    Raises:
      ValueError: If lookup_style is not an enum in EmbedLookupStyle or if `ids`
        is not an integer array.
    """
    lookup_style = lookup_style or self._lookup_style
    ids = jnp.asarray(ids)
    if not jnp.issubdtype(ids.dtype, jnp.integer):
      raise ValueError("hk.Embed's __call__ method must take an array of "
                       "integer dtype but was called with an array of "
                       "{dtype}".format(dtype=ids.dtype))

    if lookup_style == EmbedLookupStyle.ARRAY_INDEX.name:
      return self._embedding[ids]
    elif lookup_style == EmbedLookupStyle.ONE_HOT.name:
      one_hot_ids = basic.one_hot(ids, self._vocab_size)[..., None]
      return (self._embedding * one_hot_ids).sum(axis=-2)
    else:
      raise ValueError(
          "{s} is not a valid enum in EmbedLookupStyle.".format(s=lookup_style))

  @property
  def vocab_size(self):
    """Size of input vocabulary."""
    return self._vocab_size

  @property
  def embed_dim(self):
    """Size of embedding vectors."""
    return self._embed_dim

  @property
  def embeddings(self):
    """Returns the Variable containing embeddings."""
    return self._embedding
