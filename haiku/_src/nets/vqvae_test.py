# Lint as: python3
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
"""Tests for haiku._src.nets.vqvae."""

from absl.testing import absltest
from absl.testing import parameterized

from haiku._src import stateful
from haiku._src import test_utils
from haiku._src.nets import vqvae
import jax
import jax.numpy as jnp
import numpy as np


class VqvaeTest(parameterized.TestCase):

  @parameterized.parameters((vqvae.VectorQuantizer, {
      'embedding_dim': 4,
      'num_embeddings': 8,
      'commitment_cost': 0.25
  }), (vqvae.VectorQuantizerEMA, {
      'embedding_dim': 6,
      'num_embeddings': 13,
      'commitment_cost': 0.5,
      'decay': 0.1
  }))
  @test_utils.transform_and_run
  def testConstruct(self, constructor, kwargs):
    vqvae_module = constructor(**kwargs)
    # Batch of input vectors to quantize
    inputs_np = np.random.randn(100, kwargs['embedding_dim']).astype(np.float32)
    inputs = jnp.array(inputs_np)

    # Set is_training to False, otherwise for the EMA case just evaluating the
    # forward pass will change the embeddings, meaning that some of our computed
    # closest embeddings will be incorrect.
    vq_output = vqvae_module(inputs, is_training=False)

    # Output shape is correct
    self.assertEqual(vq_output['quantize'].shape, inputs.shape)

    vq_output_np = jax.tree_map(lambda t: t, vq_output)
    embeddings_np = vqvae_module.embeddings

    self.assertEqual(embeddings_np.shape,
                     (kwargs['embedding_dim'], kwargs['num_embeddings']))

    # Check that each input was assigned to the embedding it is closest to.
    distances = ((inputs_np**2).sum(axis=1, keepdims=True) -
                 2 * np.dot(inputs_np, embeddings_np) +
                 (embeddings_np**2).sum(axis=0, keepdims=True))
    closest_index = np.argmax(-distances, axis=1)
    # On TPU, distances can be different by ~1% due to precision. This can cause
    # the distanc to the closest embedding to flip, leading to a difference
    # in the encoding indices tensor. First we check that the continuous
    # distances are reasonably close, and then we only allow N differences in
    # the encodings. For batch of 100, N == 3 seems okay (passed 1000x tests).
    np.testing.assert_allclose(distances, vq_output_np['distances'], atol=4e-2)
    num_differences_in_encodings = (closest_index !=
                                    vq_output_np['encoding_indices']).sum()
    num_differences_allowed = 3
    self.assertLessEqual(num_differences_in_encodings, num_differences_allowed)

  @parameterized.parameters((vqvae.VectorQuantizer, {
      'embedding_dim': 4,
      'num_embeddings': 8,
      'commitment_cost': 0.25
  }), (vqvae.VectorQuantizerEMA, {
      'embedding_dim': 6,
      'num_embeddings': 13,
      'commitment_cost': 0.5,
      'decay': 0.1
  }))
  @test_utils.transform_and_run
  def testShapeChecking(self, constructor, kwargs):
    vqvae_module = constructor(**kwargs)
    wrong_shape_input = np.random.randn(100, kwargs['embedding_dim'] * 2)
    with self.assertRaisesRegex(TypeError, 'total size must be unchanged'):
      vqvae_module(
          jnp.array(wrong_shape_input.astype(np.float32)), is_training=False)

  @parameterized.parameters((vqvae.VectorQuantizer, {
      'embedding_dim': 4,
      'num_embeddings': 8,
      'commitment_cost': 0.25
  }), (vqvae.VectorQuantizerEMA, {
      'embedding_dim': 6,
      'num_embeddings': 13,
      'commitment_cost': 0.5,
      'decay': 0.1
  }))
  @test_utils.transform_and_run
  def testNoneBatch(self, constructor, kwargs):
    """Check that vqvae can be built on input with a None batch dimension."""
    vqvae_module = constructor(**kwargs)
    inputs = jnp.zeros([0, 5, 5, kwargs['embedding_dim']])
    vqvae_module(inputs, is_training=False)

  @parameterized.parameters({'use_jit': True, 'dtype': jnp.float32},
                            {'use_jit': True, 'dtype': jnp.float64},
                            {'use_jit': False, 'dtype': jnp.float32},
                            {'use_jit': False, 'dtype': jnp.float64})
  @test_utils.transform_and_run
  def testEmaUpdating(self, use_jit, dtype):
    if jax.local_devices()[0].platform == 'tpu' and dtype == jnp.float64:
      self.skipTest('F64 not supported by TPU')

    embedding_dim = 6
    np_dtype = np.float64 if dtype is jnp.float64 else np.float32
    decay = np.array(0.1, dtype=np_dtype)
    vqvae_module = vqvae.VectorQuantizerEMA(
        embedding_dim=embedding_dim,
        num_embeddings=7,
        commitment_cost=0.5,
        decay=decay,
        dtype=dtype)

    if use_jit:
      vqvae_f = stateful.jit(vqvae_module, static_argnums=1)
    else:
      vqvae_f = vqvae_module

    batch_size = 16

    prev_embeddings = vqvae_module.embeddings

    # Embeddings should change with every forwards pass if is_training == True.
    for _ in range(10):
      inputs = np.random.rand(batch_size, embedding_dim).astype(dtype)
      vqvae_f(inputs, True)
      current_embeddings = vqvae_module.embeddings
      self.assertFalse((prev_embeddings == current_embeddings).all())
      prev_embeddings = current_embeddings

    # Forward passes with is_training == False don't change anything
    for _ in range(10):
      inputs = np.random.rand(batch_size, embedding_dim).astype(dtype)
      vqvae_f(inputs, False)
      current_embeddings = vqvae_module.embeddings
      self.assertTrue((current_embeddings == prev_embeddings).all())


if __name__ == '__main__':
  absltest.main()
