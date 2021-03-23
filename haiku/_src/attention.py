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
"""(Multi-Head) Attention module to be used in a Transformer architecture."""

import types
from typing import Optional

from haiku._src import basic
from haiku._src import initializers
from haiku._src import module
import jax
import jax.numpy as jnp

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.Module = module.Module
hk.Linear = basic.Linear
hk.transparent = module.transparent
hk.initializers = initializers
del basic, module, initializers


class MultiHeadAttention(hk.Module):
  """Multi-headed attention mechanism.

  As described in the vanilla Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      w_init_scale: float,
      query_size: Optional[int] = None,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.query_size = query_size or key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size * num_heads
    self.w_init = hk.initializers.VarianceScaling(w_init_scale)

  def __call__(
      self,
      query: jnp.ndarray,
      key: jnp.ndarray,
      value: jnp.ndarray,
      mask: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Compute (optionally masked) MHA with batched queries, keys & values."""
    query_heads = self._linear_projection(query, self.query_size, "query")
    key_heads = self._linear_projection(key, self.key_size, "key")
    value_heads = self._linear_projection(value, self.value_size, "value")

    attention_logits = jnp.einsum("bthd,bThd->bhtT", query_heads, key_heads)
    sqrt_key_size = jnp.sqrt(self.key_size).astype(key.dtype)
    attention_logits = attention_logits / sqrt_key_size
    if mask is not None:
      attention_logits -= 1e10 * (1. - mask)

    attention_weights = jax.nn.softmax(attention_logits)
    attention = jnp.einsum("bhtT,bThd->bthd", attention_weights, value_heads)
    # Concatenate attention matrix of all heads into a single vector.
    attention_vec = jnp.reshape(attention, (*query.shape[:2], -1))

    return hk.Linear(self.model_size, w_init=self.w_init)(attention_vec)

  @hk.transparent
  def _linear_projection(
      self,
      x: jnp.ndarray,
      head_size: int,
      name: Optional[str] = None
  ) -> jnp.ndarray:
    y = hk.Linear(self.num_heads * head_size, w_init=self.w_init, name=name)(x)
    return y.reshape((*x.shape[:2], self.num_heads, head_size))
