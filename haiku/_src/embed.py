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
import types
from typing import Optional

from haiku._src import base
from haiku._src import initializers
from haiku._src import module
import jax
import jax.numpy as jnp

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.get_parameter = base.get_parameter
hk.Module = module.Module
hk.initializers = initializers
del base, module, initializers


class EmbedLookupStyle(enum.Enum):
  """How to return the embedding matrices given IDs."""
  ARRAY_INDEX = 1
  ONE_HOT = 2


class Embed(hk.Module):
  """Module for embedding tokens in a low-dimensional space."""

  def __init__(
      self,
      vocab_size: Optional[int] = None,
      embed_dim: Optional[int] = None,
      embedding_matrix: Optional[jnp.ndarray] = None,
      w_init: Optional[hk.initializers.Initializer] = None,
      # TODO(tomhennigan) Support EmbedLookupStyle or string.
      lookup_style: str = EmbedLookupStyle.ARRAY_INDEX.name,
      name: Optional[str] = None,
  ):
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
    super().__init__(name=name)
    if embedding_matrix is None and not (vocab_size and embed_dim):
      raise ValueError(
          "hk.Embed must be supplied either with an initial `embedding_matrix` "
          "or with `embed_dim` and `vocab_size`.")
    if embedding_matrix is not None:
      embedding_matrix = jnp.asarray(embedding_matrix)
      if vocab_size and embedding_matrix.shape[0] != vocab_size:
        raise ValueError(
            "An `embedding_matrix` was supplied but the `vocab_size` of "
            f"{vocab_size} was not consistent with its shape "
            f"{embedding_matrix.shape}.")
      if embed_dim and embedding_matrix.shape[1] != embed_dim:
        raise ValueError(
            "An `embedding_matrix` was supplied but the `embed_dim` of "
            f"{embed_dim} was not consistent with its shape "
            f"{embedding_matrix.shape}.")
      self.embeddings = hk.get_parameter("embeddings", embedding_matrix.shape,
                                         init=lambda _, __: embedding_matrix)
    else:
      w_init = w_init or hk.initializers.TruncatedNormal()
      self.embeddings = hk.get_parameter("embeddings", [vocab_size, embed_dim],
                                         init=w_init)

    self.vocab_size = vocab_size or embedding_matrix.shape[0]
    self.embed_dim = embed_dim or embedding_matrix.shape[1]
    self.lookup_style = lookup_style

  def __call__(
      self,
      ids: jnp.ndarray,
      lookup_style: Optional[str] = None,
  ) -> jnp.ndarray:
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
    lookup_style = lookup_style or self.lookup_style
    # TODO(tomhennigan) Consider removing asarray here.
    ids = jnp.asarray(ids)
    if not jnp.issubdtype(ids.dtype, jnp.integer):
      raise ValueError("hk.Embed's __call__ method must take an array of "
                       "integer dtype but was called with an array of "
                       f"{ids.dtype}")

    if lookup_style == EmbedLookupStyle.ARRAY_INDEX.name:
      return self.embeddings[ids]
    elif lookup_style == EmbedLookupStyle.ONE_HOT.name:
      one_hot_ids = jax.nn.one_hot(ids, self.vocab_size)[..., None]
      return (self.embeddings * one_hot_ids).sum(axis=-2)
    else:
      raise ValueError(
          f"{lookup_style} is not a valid enum in EmbedLookupStyle.")
