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
"""Variational Autoencoder example on binarized MNIST dataset.

See "Auto-encoding variational Bayes" (Kingma & Welling, 2014) [0].

[0]https://arxiv.org/abs/1312.6114
"""

from collections.abc import Iterator, Sequence
import dataclasses
from typing import NamedTuple

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds


@dataclasses.dataclass
class Config:
  batch_size: int = 128
  learning_rate: float = 1e-3
  training_steps: int = 5000
  eval_every: int = 100
  seed: int = 0


class Batch(NamedTuple):
  image: jax.Array  # [B, H, W, C]


def load_dataset(split: str, batch_size: int, seed: int) -> Iterator[Batch]:
  ds = (
      tfds.load("binarized_mnist", split=split)
      .shuffle(buffer_size=10 * batch_size, seed=seed)
      .batch(batch_size)
      .prefetch(buffer_size=5)
      .repeat()
      .as_numpy_iterator()
  )
  return map(lambda x: Batch(x["image"]), ds)


@dataclasses.dataclass
class Encoder(hk.Module):
  """Encoder model."""

  latent_size: int
  hidden_size: int = 512

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Encodes an image as an isotropic Guassian latent code."""
    x = hk.Flatten()(x)
    x = hk.Linear(self.hidden_size)(x)
    x = jax.nn.relu(x)

    mean = hk.Linear(self.latent_size)(x)
    log_stddev = hk.Linear(self.latent_size)(x)
    stddev = jnp.exp(log_stddev)

    return mean, stddev


@dataclasses.dataclass
class Decoder(hk.Module):
  """Decoder model."""

  output_shape: Sequence[int]
  hidden_size: int = 512

  def __call__(self, z: jax.Array) -> jax.Array:
    """Decodes a latent code into Bernoulli log-odds over an output image."""
    z = hk.Linear(self.hidden_size)(z)
    z = jax.nn.relu(z)

    logits = hk.Linear(np.prod(self.output_shape))(z)
    logits = jnp.reshape(logits, (-1, *self.output_shape))

    return logits


class VAEOutput(NamedTuple):
  image: jax.Array
  mean: jax.Array
  variance: jax.Array
  logits: jax.Array


@dataclasses.dataclass
class VariationalAutoEncoder(hk.Module):
  """Main VAE model class."""

  encoder: Encoder
  decoder: Decoder

  def __call__(self, x: jax.Array) -> VAEOutput:
    """Forward pass of the variational autoencoder."""
    x = x.astype(jnp.float32)
    mean, stddev = self.encoder(x)
    z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
    logits = self.decoder(z)

    p = jax.nn.sigmoid(logits)
    image = jax.random.bernoulli(hk.next_rng_key(), p)

    return VAEOutput(image, mean, jnp.square(stddev), logits)


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState
  rng_key: jax.Array


def main(_):

  flags.FLAGS.alsologtostderr = True
  config = Config()

  @hk.transform
  def model(x):
    vae = VariationalAutoEncoder(
        encoder=Encoder(latent_size=10),
        decoder=Decoder(output_shape=x.shape[1:]),
    )
    return vae(x)

  @jax.jit
  def loss_fn(params, rng_key, batch: Batch) -> jax.Array:
    """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""

    # Run the model on the inputs.
    _, mean, var, logits = model.apply(params, rng_key, batch.image)

    # Bernoulli log-likelihood (assumes `image` is binarised).
    log_likelihood = jnp.einsum(
        "b...->b", batch.image * logits - jnp.logaddexp(0., logits))

    # KL divergence between Gaussians N(mean, std) and N(0, 1).
    kl = 0.5 * jnp.sum(-jnp.log(var) - 1. + var + jnp.square(mean), axis=-1)

    # Loss is the negative evidence lower-bound.
    return -jnp.mean(log_likelihood - kl)

  optimizer = optax.adam(config.learning_rate)

  @jax.jit
  def update(state: TrainingState, batch: Batch) -> TrainingState:
    """Performs a single SGD step."""
    rng_key, next_rng_key = jax.random.split(state.rng_key)
    gradients = jax.grad(loss_fn)(state.params, rng_key, batch)
    updates, new_opt_state = optimizer.update(gradients, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return TrainingState(new_params, new_opt_state, next_rng_key)

  # Load datasets.
  train_dataset = load_dataset("train", config.batch_size, config.seed)
  eval_datasets = {
      "train": load_dataset("train", config.batch_size, config.seed),
      "valid": load_dataset("validation", config.batch_size, config.seed),
  }

  # Initialise the training state.
  initial_rng_key = jax.random.PRNGKey(config.seed)
  initial_params = model.init(initial_rng_key, next(train_dataset).image)
  initial_opt_state = optimizer.init(initial_params)
  state = TrainingState(initial_params, initial_opt_state, initial_rng_key)

  # Run training and evaluation.
  for step in range(config.training_steps):
    state = update(state, next(train_dataset))

    if step % config.eval_every == 0:
      for split, ds in eval_datasets.items():
        loss = loss_fn(state.params, state.rng_key, next(ds))
        logging.info({
            "step": step,
            "split": split,
            "elbo": -jax.device_get(loss).item(),
        })


if __name__ == "__main__":
  app.run(main)
