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
"""Variational Autoencoder example on binarized MNIST dataset."""

from typing import Any, Generator, Mapping, Tuple, NamedTuple, Sequence

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds


flags.DEFINE_integer("batch_size", 128, "Size of the batch to train on.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer.")
flags.DEFINE_integer("training_steps", 5000, "Number of training steps to run.")
flags.DEFINE_integer("eval_frequency", 100, "How often to evaluate the model.")
FLAGS = flags.FLAGS


OptState = Any
PRNGKey = jnp.ndarray
Batch = Mapping[str, np.ndarray]

MNIST_IMAGE_SHAPE: Sequence[int] = (28, 28, 1)


def load_dataset(split: str, batch_size: int) -> Generator[Batch, None, None]:
  ds = tfds.load("binarized_mnist", split=split, shuffle_files=True)
  ds = ds.shuffle(buffer_size=10 * batch_size)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=5)
  ds = ds.repeat()
  return tfds.as_numpy(ds)


class Encoder(hk.Module):
  """Encoder model."""

  def __init__(self, hidden_size: int = 512, latent_size: int = 10):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size

  def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = hk.Flatten()(x)
    x = hk.Linear(self._hidden_size)(x)
    x = jax.nn.relu(x)

    mean = hk.Linear(self._latent_size)(x)
    log_stddev = hk.Linear(self._latent_size)(x)
    stddev = jnp.exp(log_stddev)

    return mean, stddev


class Decoder(hk.Module):
  """Decoder model."""

  def __init__(
      self,
      hidden_size: int = 512,
      output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
  ):
    super().__init__()
    self._hidden_size = hidden_size
    self._output_shape = output_shape

  def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
    z = hk.Linear(self._hidden_size)(z)
    z = jax.nn.relu(z)

    logits = hk.Linear(jnp.prod(self._output_shape))(z)
    logits = jnp.reshape(logits, (-1, *self._output_shape))

    return logits


class VAEOutput(NamedTuple):
  image: jnp.ndarray
  mean: jnp.ndarray
  stddev: jnp.ndarray
  logits: jnp.ndarray


class VariationalAutoEncoder(hk.Module):
  """Main VAE model class, uses Encoder & Decoder under the hood."""

  def __init__(
      self,
      hidden_size: int = 512,
      latent_size: int = 10,
      output_shape: Sequence[int] = MNIST_IMAGE_SHAPE,
  ):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size
    self._output_shape = output_shape

  def __call__(self, x: jnp.ndarray) -> VAEOutput:
    x = x.astype(jnp.float32)
    mean, stddev = Encoder(self._hidden_size, self._latent_size)(x)
    z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
    logits = Decoder(self._hidden_size, self._output_shape)(z)

    p = jax.nn.sigmoid(logits)
    image = jax.random.bernoulli(hk.next_rng_key(), p)

    return VAEOutput(image, mean, stddev, logits)


def binary_cross_entropy(x: jnp.ndarray, logits: jnp.ndarray) -> jnp.ndarray:
  """Calculate binary (logistic) cross-entropy from distribution logits.

  Args:
    x: input variable tensor, must be of same shape as logits
    logits: log odds of a Bernoulli distribution, i.e. log(p/(1-p))

  Returns:
    A scalar representing binary CE for the given Bernoulli distribution.
  """
  if x.shape != logits.shape:
    raise ValueError("inputs x and logits must be of the same shape")

  x = jnp.reshape(x, (x.shape[0], -1))
  logits = jnp.reshape(logits, (logits.shape[0], -1))

  return -jnp.sum(x * logits - jnp.logaddexp(0.0, logits), axis=-1)


def kl_gaussian(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
  r"""Calculate KL divergence between given and standard gaussian distributions.

  KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
           = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
           = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)

  Args:
    mean: mean vector of the first distribution
    var: diagonal vector of covariance matrix of the first distribution

  Returns:
    A scalar representing KL divergence of the two Gaussian distributions.
  """
  return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + mean**2, axis=-1)


def main(_):
  model = hk.transform(lambda x: VariationalAutoEncoder()(x), apply_rng=True)  # pylint: disable=unnecessary-lambda
  optimizer = optix.adam(FLAGS.learning_rate)

  @jax.jit
  def loss_fn(params: hk.Params, rng_key: PRNGKey, batch: Batch) -> jnp.ndarray:
    """ELBO loss: E_p[log(e)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""
    outputs: VAEOutput = model.apply(params, rng_key, batch["image"])

    log_likelihood = -binary_cross_entropy(batch["image"], outputs.logits)
    kl = kl_gaussian(outputs.mean, outputs.stddev**2)
    loss = log_likelihood - kl

    return -jnp.mean(loss)

  @jax.jit
  def update(
      params: hk.Params,
      rng_key: PRNGKey,
      opt_state: OptState,
      batch: Batch,
  ) -> Tuple[hk.Params, OptState]:
    """Single SGD update step."""
    grads = jax.grad(loss_fn)(params, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optix.apply_updates(params, updates)
    return new_params, new_opt_state

  rng_seq = hk.PRNGSequence(42)
  params = model.init(next(rng_seq), np.zeros((1, *MNIST_IMAGE_SHAPE)))
  opt_state = optimizer.init(params)

  train_ds = load_dataset(tfds.Split.TRAIN, FLAGS.batch_size)
  valid_ds = load_dataset(tfds.Split.TEST, FLAGS.batch_size)

  for step in range(FLAGS.training_steps):
    params, opt_state = update(params, next(rng_seq), opt_state, next(train_ds))

    if step % FLAGS.eval_frequency == 0:
      val_loss = loss_fn(params, next(rng_seq), next(valid_ds))
      logging.info("STEP: %5d; Validation ELBO: %.3f", step, -val_loss)


if __name__ == "__main__":
  app.run(main)
