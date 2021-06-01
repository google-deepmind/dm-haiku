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
"""Haiku implementation of VQ-VAE https://arxiv.org/abs/1711.00937."""

import types
from typing import Any, Optional

from haiku._src import base
from haiku._src import initializers
from haiku._src import module
from haiku._src import moving_averages

import jax
import jax.numpy as jnp

# If forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.get_parameter = base.get_parameter
hk.get_state = base.get_state
hk.set_state = base.set_state
hk.initializers = initializers
hk.ExponentialMovingAverage = moving_averages.ExponentialMovingAverage
hk.Module = module.Module
del base, initializers, module, moving_averages


class VectorQuantizer(hk.Module):
  """Haiku module representing the VQ-VAE layer.

  Implements the algorithm presented in
  "Neural Discrete Representation Learning" by van den Oord et al.
  https://arxiv.org/abs/1711.00937

  Input any tensor to be quantized. Last dimension will be used as space in
  which to quantize. All other dimensions will be flattened and will be seen
  as different examples to quantize.

  The output tensor will have the same shape as the input.

  For example a tensor with shape ``[16, 32, 32, 64]`` will be reshaped into
  ``[16384, 64]`` and all ``16384`` vectors (each of ``64`` dimensions)  will be
  quantized independently.

  Attributes:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms (see
      equation 4 in the paper - this variable is Beta).
  """

  def __init__(
      self,
      embedding_dim: int,
      num_embeddings: int,
      commitment_cost: float,
      dtype: Any = jnp.float32,
      name: Optional[str] = None,
  ):
    """Initializes a VQ-VAE module.

    Args:
      embedding_dim: dimensionality of the tensors in the quantized space.
        Inputs to the modules must be in this format as well.
      num_embeddings: number of vectors in the quantized space.
      commitment_cost: scalar which controls the weighting of the loss terms
        (see equation 4 in the paper - this variable is Beta).
      dtype: dtype for the embeddings variable, defaults to ``float32``.
      name: name of the module.
    """
    super().__init__(name=name)
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost

    self._embedding_shape = [embedding_dim, num_embeddings]
    self._embedding_dtype = dtype

  @property
  def embeddings(self):
    initializer = hk.initializers.VarianceScaling(distribution="uniform")
    return hk.get_parameter(
        "embeddings",
        self._embedding_shape,
        self._embedding_dtype,
        init=initializer)

  def __call__(self, inputs, is_training):
    """Connects the module to some inputs.

    Args:
      inputs: Tensor, final dimension must be equal to ``embedding_dim``. All
        other leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data.

    Returns:
      dict: Dictionary containing the following keys and values:
        * ``quantize``: Tensor containing the quantized version of the input.
        * ``loss``: Tensor containing the loss to optimize.
        * ``perplexity``: Tensor containing the perplexity of the encodings.
        * ``encodings``: Tensor containing the discrete encodings, ie which
          element of the quantized space each input element was mapped to.
        * ``encoding_indices``: Tensor containing the discrete encoding indices,
          ie which element of the quantized space each input element was mapped
          to.
    """
    flat_inputs = jnp.reshape(inputs, [-1, self.embedding_dim])

    distances = (
        jnp.sum(jnp.square(flat_inputs), 1, keepdims=True) -
        2 * jnp.matmul(flat_inputs, self.embeddings) +
        jnp.sum(jnp.square(self.embeddings), 0, keepdims=True))

    encoding_indices = jnp.argmax(-distances, 1)
    encodings = jax.nn.one_hot(encoding_indices,
                               self.num_embeddings,
                               dtype=distances.dtype)

    # NB: if your code crashes with a reshape error on the line below about a
    # Tensor containing the wrong number of values, then the most likely cause
    # is that the input passed in does not have a final dimension equal to
    # self.embedding_dim. Ideally we would catch this with an Assert but that
    # creates various other problems related to device placement / TPUs.
    encoding_indices = jnp.reshape(encoding_indices, inputs.shape[:-1])
    quantized = self.quantize(encoding_indices)

    e_latent_loss = jnp.mean(
        jnp.square(jax.lax.stop_gradient(quantized) - inputs))
    q_latent_loss = jnp.mean(
        jnp.square(quantized - jax.lax.stop_gradient(inputs)))
    loss = q_latent_loss + self.commitment_cost * e_latent_loss

    # Straight Through Estimator
    quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
    avg_probs = jnp.mean(encodings, 0)
    perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

    return {
        "quantize": quantized,
        "loss": loss,
        "perplexity": perplexity,
        "encodings": encodings,
        "encoding_indices": encoding_indices,
        "distances": distances,
    }

  def quantize(self, encoding_indices):
    """Returns embedding tensor for a batch of indices."""
    w = self.embeddings.swapaxes(1, 0)
    w = jax.device_put(w)  # Required when embeddings is a NumPy array.
    return w[(encoding_indices,)]


class VectorQuantizerEMA(hk.Module):
  r"""Haiku module representing the VQ-VAE layer.

  Implements a slightly modified version of the algorithm presented in
  "Neural Discrete Representation Learning" by van den Oord et al.
  https://arxiv.org/abs/1711.00937

  The difference between :class:`VectorQuantizerEMA` and
  :class:`VectorQuantizer` is that this module uses
  :class:`~haiku.ExponentialMovingAverage`\ s to update the embedding vectors
  instead of an auxiliary loss. This has the advantage that the embedding
  updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
  ...) used for the encoder, decoder and other parts of the architecture. For
  most experiments the EMA version trains faster than the non-EMA version.

  Input any tensor to be quantized. Last dimension will be used as space in
  which to quantize. All other dimensions will be flattened and will be seen
  as different examples to quantize.

  The output tensor will have the same shape as the input.

  For example a tensor with shape ``[16, 32, 32, 64]`` will be reshaped into
  ``[16384, 64]`` and all ``16384`` vectors (each of 64 dimensions)  will be
  quantized independently.

  Attributes:
    embedding_dim: integer representing the dimensionality of the tensors in
      the quantized space. Inputs to the modules must be in this format as well.
    num_embeddings: integer, the number of vectors in the quantized space.
    commitment_cost: scalar which controls the weighting of the loss terms
      (see equation 4 in the paper).
    decay: float, decay for the moving averages.
    epsilon: small float constant to avoid numerical instability.
  """

  def __init__(
      self,
      embedding_dim,
      num_embeddings,
      commitment_cost,
      decay,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      cross_replica_axis: Optional[str] = None,
      name: Optional[str] = None,
  ):
    """Initializes a VQ-VAE EMA module.

    Args:
      embedding_dim: integer representing the dimensionality of the tensors in
        the quantized space. Inputs to the modules must be in this format as
        well.
      num_embeddings: integer, the number of vectors in the quantized space.
      commitment_cost: scalar which controls the weighting of the loss terms
        (see equation 4 in the paper - this variable is Beta).
      decay: float between 0 and 1, controls the speed of the Exponential Moving
        Averages.
      epsilon: small constant to aid numerical stability, default ``1e-5``.
      dtype: dtype for the embeddings variable, defaults to ``float32``.
      cross_replica_axis: If not ``None``, it should be a string representing
        the axis name over which this module is being run within a
        :func:`jax.pmap`. Supplying this argument means that cluster statistics
        and the perplexity are calculated across all replicas on that axis.
      name: name of the module.
    """
    super().__init__(name=name)
    if not 0 <= decay <= 1:
      raise ValueError("decay must be in range [0, 1]")

    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.decay = decay
    self.commitment_cost = commitment_cost
    self.epsilon = epsilon
    self.cross_replica_axis = cross_replica_axis

    self._embedding_shape = [embedding_dim, num_embeddings]
    self._dtype = dtype

    self._ema_cluster_size = hk.ExponentialMovingAverage(
        decay=self.decay, name="ema_cluster_size")
    self._ema_dw = hk.ExponentialMovingAverage(decay=self.decay, name="ema_dw")

  @property
  def embeddings(self):
    initializer = hk.initializers.VarianceScaling(distribution="uniform")
    return hk.get_state(
        "embeddings", self._embedding_shape, self._dtype, init=initializer)

  @property
  def ema_cluster_size(self):
    self._ema_cluster_size.initialize([self.num_embeddings], self._dtype)
    return self._ema_cluster_size

  @property
  def ema_dw(self):
    self._ema_dw.initialize(self._embedding_shape, self._dtype)
    return self._ema_dw

  def __call__(self, inputs, is_training):
    """Connects the module to some inputs.

    Args:
      inputs: Tensor, final dimension must be equal to ``embedding_dim``. All
        other leading dimensions will be flattened and treated as a large batch.
      is_training: boolean, whether this connection is to training data. When
        this is set to ``False``, the internal moving average statistics will
        not be updated.

    Returns:
      dict: Dictionary containing the following keys and values:
        * ``quantize``: Tensor containing the quantized version of the input.
        * ``loss``: Tensor containing the loss to optimize.
        * ``perplexity``: Tensor containing the perplexity of the encodings.
        * ``encodings``: Tensor containing the discrete encodings, ie which
          element of the quantized space each input element was mapped to.
        * ``encoding_indices``: Tensor containing the discrete encoding indices,
          ie which element of the quantized space each input element was mapped
          to.
    """
    flat_inputs = jnp.reshape(inputs, [-1, self.embedding_dim])
    embeddings = self.embeddings

    distances = (
        jnp.sum(jnp.square(flat_inputs), 1, keepdims=True) -
        2 * jnp.matmul(flat_inputs, embeddings) +
        jnp.sum(jnp.square(embeddings), 0, keepdims=True))

    encoding_indices = jnp.argmax(-distances, 1)
    encodings = jax.nn.one_hot(encoding_indices,
                               self.num_embeddings,
                               dtype=distances.dtype)

    # NB: if your code crashes with a reshape error on the line below about a
    # Tensor containing the wrong number of values, then the most likely cause
    # is that the input passed in does not have a final dimension equal to
    # self.embedding_dim. Ideally we would catch this with an Assert but that
    # creates various other problems related to device placement / TPUs.
    encoding_indices = jnp.reshape(encoding_indices, inputs.shape[:-1])
    quantized = self.quantize(encoding_indices)
    e_latent_loss = jnp.mean(
        jnp.square(jax.lax.stop_gradient(quantized) - inputs))

    if is_training:
      cluster_size = jnp.sum(encodings, axis=0)
      if self.cross_replica_axis:
        cluster_size = jax.lax.psum(
            cluster_size, axis_name=self.cross_replica_axis)
      updated_ema_cluster_size = self.ema_cluster_size(cluster_size)

      dw = jnp.matmul(flat_inputs.T, encodings)
      if self.cross_replica_axis:
        dw = jax.lax.psum(dw, axis_name=self.cross_replica_axis)
      updated_ema_dw = self.ema_dw(dw)

      n = jnp.sum(updated_ema_cluster_size)
      updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                  (n + self.num_embeddings * self.epsilon) * n)

      normalised_updated_ema_w = (
          updated_ema_dw / jnp.reshape(updated_ema_cluster_size, [1, -1]))

      hk.set_state("embeddings", normalised_updated_ema_w)
      loss = self.commitment_cost * e_latent_loss

    else:
      loss = self.commitment_cost * e_latent_loss

    # Straight Through Estimator
    quantized = inputs + jax.lax.stop_gradient(quantized - inputs)
    avg_probs = jnp.mean(encodings, 0)
    if self.cross_replica_axis:
      avg_probs = jax.lax.pmean(avg_probs, axis_name=self.cross_replica_axis)
    perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

    return {
        "quantize": quantized,
        "loss": loss,
        "perplexity": perplexity,
        "encodings": encodings,
        "encoding_indices": encoding_indices,
        "distances": distances,
    }

  def quantize(self, encoding_indices):
    """Returns embedding tensor for a batch of indices."""
    w = self.embeddings.swapaxes(1, 0)
    w = jax.device_put(w)  # Required when embeddings is a NumPy array.
    return w[(encoding_indices,)]
