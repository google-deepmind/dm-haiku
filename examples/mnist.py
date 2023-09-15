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
"""A minimal MNIST classifier example."""

from collections.abc import Iterator
from typing import NamedTuple

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

NUM_CLASSES = 10  # MNIST has 10 classes (hand-written digits).


class Batch(NamedTuple):
  image: np.ndarray  # [B, H, W, 1]
  label: np.ndarray  # [B]


class TrainingState(NamedTuple):
  params: hk.Params
  avg_params: hk.Params
  opt_state: optax.OptState


def net_fn(images: jax.Array) -> jax.Array:
  """Standard LeNet-300-100 MLP network."""
  x = images.astype(jnp.float32) / 255.
  mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(300), jax.nn.relu,
      hk.Linear(100), jax.nn.relu,
      hk.Linear(NUM_CLASSES),
  ])
  return mlp(x)


def load_dataset(
    split: str,
    *,
    shuffle: bool,
    batch_size: int,
) -> Iterator[Batch]:
  """Loads the MNIST dataset."""
  ds, ds_info = tfds.load("mnist:3.*.*", split=split, with_info=True)
  ds.cache()
  if shuffle:
    ds = ds.shuffle(ds_info.splits[split].num_examples, seed=0)
  ds = ds.repeat()
  ds = ds.batch(batch_size)
  ds = ds.map(lambda x: Batch(**x))
  return iter(tfds.as_numpy(ds))


def main(_):
  # First, make the network and optimiser.
  network = hk.without_apply_rng(hk.transform(net_fn))
  optimiser = optax.adam(1e-3)

  def loss(params: hk.Params, batch: Batch) -> jax.Array:
    """Cross-entropy classification loss, regularised by L2 weight decay."""
    batch_size, *_ = batch.image.shape
    logits = network.apply(params, batch.image)
    labels = jax.nn.one_hot(batch.label, NUM_CLASSES)

    l2_regulariser = 0.5 * sum(
        jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))

    return -log_likelihood / batch_size + 1e-4 * l2_regulariser

  @jax.jit
  def evaluate(params: hk.Params, batch: Batch) -> jax.Array:
    """Evaluation metric (classification accuracy)."""
    logits = network.apply(params, batch.image)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == batch.label)

  @jax.jit
  def update(state: TrainingState, batch: Batch) -> TrainingState:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(state.params, batch)
    updates, opt_state = optimiser.update(grads, state.opt_state)
    params = optax.apply_updates(state.params, updates)
    # Compute avg_params, the exponential moving average of the "live" params.
    # We use this only for evaluation (cf. https://doi.org/10.1137/0330046).
    avg_params = optax.incremental_update(
        params, state.avg_params, step_size=0.001)
    return TrainingState(params, avg_params, opt_state)

  # Make datasets.
  train_dataset = load_dataset("train", shuffle=True, batch_size=1_000)
  eval_datasets = {
      split: load_dataset(split, shuffle=False, batch_size=10_000)
      for split in ("train", "test")
  }

  # Initialise network and optimiser; note we draw an input to get shapes.
  initial_params = network.init(
      jax.random.PRNGKey(seed=0), next(train_dataset).image)
  initial_opt_state = optimiser.init(initial_params)
  state = TrainingState(initial_params, initial_params, initial_opt_state)

  # Training & evaluation loop.
  for step in range(3001):
    if step % 100 == 0:
      # Periodically evaluate classification accuracy on train & test sets.
      # Note that each evaluation is only on a (large) batch.
      for split, dataset in eval_datasets.items():
        accuracy = np.array(evaluate(state.avg_params, next(dataset))).item()
        print({"step": step, "split": split, "accuracy": f"{accuracy:.3f}"})

    # Do SGD on a batch of training examples.
    state = update(state, next(train_dataset))

if __name__ == "__main__":
  app.run(main)
