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
r"""Trains a transformer for language modeling on a small text dataset.

This example serves to demonstrate:
  - A clean Haiku transformer implementation.
  - An example minimal training loop around it.

This example runs on ASCII text files.
We have not tuned the hyperparameters at all.

Example, using Karpathy's tiny_shakespeare dataset:
$ wget -O /tmp/shakespeare.txt \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
$ python3 examples/transformer/train.py \
    --dataset_path=/tmp/shakespeare.txt --alsologtostderr
"""

import time
from typing import Any, MutableMapping, NamedTuple, Tuple

from absl import app
from absl import flags
from absl import logging
import haiku as hk
from examples.transformer import dataset
from examples.transformer import model
import jax
import jax.numpy as jnp
import numpy as np
import optax

DATASET_PATH = flags.DEFINE_string(
    'dataset_path', None, help='Path to raw dataset file', required=True)

# Training hyperparameters.
BATCH_SIZE = 2
SEQUENCE_LENGTH = 64
LEARNING_RATE = 3e-4
GRAD_CLIP_VALUE = 1
LOG_EVERY = 50
MAX_STEPS = 10**6
SEED = 0

# Model hyperparameters.
NUM_LAYERS = 6
NUM_HEADS = 8  # Number of attention heads.
MODEL_SIZE = 128
KEY_SIZE = 32
DROPOUT_RATE = 0.1

# Helpful type aliases.
_Batch = dataset.Batch
_Metrics = MutableMapping[str, Any]


class TrainingState(NamedTuple):
  """Container for the training state."""
  params: hk.Params
  opt_state: optax.OptState
  rng: jnp.DeviceArray
  step: jnp.DeviceArray


def main(_):

  # Create the model.
  def forward(tokens: jnp.ndarray) -> jnp.ndarray:
    lm = model.LanguageModel(
        model_size=MODEL_SIZE,
        vocab_size=dataset.VOCAB_SIZE,
        pad_token=dataset.PAD_TOKEN,
        transformer=model.Transformer(
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            key_size=KEY_SIZE,
            dropout_rate=DROPOUT_RATE,
        ),
    )
    return lm(tokens)

  # Create the optimiser.
  optimiser = optax.chain(
      optax.clip_by_global_norm(GRAD_CLIP_VALUE),
      optax.adam(LEARNING_RATE, b1=0.9, b2=0.99),
  )

  # Create the loss.
  @hk.transform
  def loss_fn(data: _Batch) -> jnp.ndarray:
    """Computes the (scalar) LM loss on `data` w.r.t. params."""
    logits = forward(data.inputs)
    targets = jax.nn.one_hot(data.targets, dataset.VOCAB_SIZE)
    assert logits.shape == targets.shape

    mask = jnp.greater(data.inputs, 0)
    log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    return -jnp.sum(log_likelihood * mask) / jnp.sum(mask)  # NLL per token.

  @jax.jit
  def update(state: TrainingState, data) -> Tuple[TrainingState, _Metrics]:
    """Does an SGD step and returns metrics."""
    rng, new_rng = jax.random.split(state.rng)
    loss_and_grad_fn = jax.value_and_grad(loss_fn.apply)
    loss, gradients = loss_and_grad_fn(state.params, rng, data)

    updates, new_opt_state = optimiser.update(gradients, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(
        params=new_params,
        opt_state=new_opt_state,
        rng=new_rng,
        step=state.step + 1,
    )

    metrics = {
        'step': state.step,
        'loss': loss,
    }
    return new_state, metrics

  @jax.jit
  def init(rng: jnp.ndarray, data) -> TrainingState:
    rng, init_rng = jax.random.split(rng)
    initial_params = loss_fn.init(init_rng, data)
    initial_opt_state = optimiser.init(initial_params)
    return TrainingState(
        params=initial_params,
        opt_state=initial_opt_state,
        rng=rng,
        step=np.array(0),
    )

  # Create the dataset.
  with open(DATASET_PATH.value, mode='r') as file:
    train_dataset = dataset.load_ascii_dataset(
        corpus=file.read(),
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH,
    )

  # Initialise the model parameters.
  rng = jax.random.PRNGKey(SEED)
  data = next(train_dataset)
  state = init(rng, data)

  # Start training (note we don't include any explicit eval in this example).
  prev_time = time.time()
  for step in range(MAX_STEPS):
    data = next(train_dataset)
    state, metrics = update(state, data)
    # We use JAX runahead to mask data preprocessing and JAX dispatch overheads.
    # Using values from state/metrics too often will block the runahead and can
    # cause these overheads to become more prominent.
    if step % LOG_EVERY == 0:
      steps_per_sec = LOG_EVERY / (time.time() - prev_time)
      prev_time = time.time()
      metrics |= {'steps_per_sec': steps_per_sec}
      logging.info({k: float(v) for k, v in metrics.items()})


if __name__ == '__main__':
  app.run(main)
