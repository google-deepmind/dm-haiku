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
"""Character-level language modelling with a recurrent network in JAX."""

from typing import Any, NamedTuple

from absl import app
from absl import flags
from absl import logging

import haiku as hk
from examples.rnn import dataset
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

TRAIN_BATCH_SIZE = flags.DEFINE_integer('train_batch_size', 32, '')
EVAL_BATCH_SIZE = flags.DEFINE_integer('eval_batch_size', 1000, '')
SEQUENCE_LENGTH = flags.DEFINE_integer('sequence_length', 128, '')
HIDDEN_SIZE = flags.DEFINE_integer('hidden_size', 256, '')
SAMPLE_LENGTH = flags.DEFINE_integer('sample_length', 128, '')
LEARNING_RATE = flags.DEFINE_float('learning_rate', 1e-3, '')
TRAINING_STEPS = flags.DEFINE_integer('training_steps', 100_000, '')
EVALUATION_INTERVAL = flags.DEFINE_integer('evaluation_interval', 100, '')
SAMPLING_INTERVAL = flags.DEFINE_integer('sampling_interval', 100, '')
SEED = flags.DEFINE_integer('seed', 42, '')


class LoopValues(NamedTuple):
  tokens: jnp.ndarray
  state: Any
  rng_key: jnp.ndarray


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState


def make_network() -> hk.RNNCore:
  """Defines the network architecture."""
  model = hk.DeepRNN([
      lambda x: jax.nn.one_hot(x, num_classes=dataset.NUM_CHARS),
      hk.LSTM(HIDDEN_SIZE.value),
      jax.nn.relu,
      hk.LSTM(HIDDEN_SIZE.value),
      hk.nets.MLP([HIDDEN_SIZE.value, dataset.NUM_CHARS]),
  ])
  return model


def make_optimizer() -> optax.GradientTransformation:
  """Defines the optimizer."""
  return optax.adam(LEARNING_RATE.value)


def sequence_loss(batch: dataset.Batch) -> jnp.ndarray:
  """Unrolls the network over a sequence of inputs & targets, gets loss."""
  # Note: this function is impure; we hk.transform() it below.
  core = make_network()
  sequence_length, batch_size = batch['input'].shape
  initial_state = core.initial_state(batch_size)
  logits, _ = hk.dynamic_unroll(core, batch['input'], initial_state)
  log_probs = jax.nn.log_softmax(logits)
  one_hot_labels = jax.nn.one_hot(batch['target'], num_classes=logits.shape[-1])
  return -jnp.sum(one_hot_labels * log_probs) / (sequence_length * batch_size)


@jax.jit
def update(state: TrainingState, batch: dataset.Batch) -> TrainingState:
  """Does a step of SGD given inputs & targets."""
  _, optimizer = make_optimizer()
  _, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
  gradients = jax.grad(loss_fn)(state.params, batch)
  updates, new_opt_state = optimizer(gradients, state.opt_state)
  new_params = optax.apply_updates(state.params, updates)
  return TrainingState(params=new_params, opt_state=new_opt_state)


def sample(
    rng_key: jnp.ndarray,
    context: jnp.ndarray,
    sample_length: int,
) -> jnp.ndarray:
  """Draws samples from the model, given an initial context."""
  # Note: this function is impure; we hk.transform() it below.
  assert context.ndim == 1  # No batching for now.
  core = make_network()

  def body_fn(t: int, v: LoopValues) -> LoopValues:
    token = v.tokens[t]
    next_logits, next_state = core(token, v.state)
    key, subkey = jax.random.split(v.rng_key)
    next_token = jax.random.categorical(subkey, next_logits, axis=-1)
    new_tokens = v.tokens.at[t + 1].set(next_token)
    return LoopValues(tokens=new_tokens, state=next_state, rng_key=key)

  logits, state = hk.dynamic_unroll(core, context, core.initial_state(None))
  key, subkey = jax.random.split(rng_key)
  first_token = jax.random.categorical(subkey, logits[-1])
  tokens = np.zeros(sample_length, dtype=np.int32)
  tokens = tokens.at[0].set(first_token)
  initial_values = LoopValues(tokens=tokens, state=state, rng_key=key)
  values: LoopValues = lax.fori_loop(0, sample_length, body_fn, initial_values)

  return values.tokens


def main(_):
  flags.FLAGS.alsologtostderr = True

  # Make training dataset.
  train_data = dataset.load(
      tfds.Split.TRAIN,
      batch_size=TRAIN_BATCH_SIZE.value,
      sequence_length=SEQUENCE_LENGTH.value)

  # Make evaluation dataset(s).
  eval_data = {  # pylint: disable=g-complex-comprehension
      split: dataset.load(
          split,
          batch_size=EVAL_BATCH_SIZE.value,
          sequence_length=SEQUENCE_LENGTH.value)
      for split in [tfds.Split.TRAIN, tfds.Split.TEST]
  }

  # Make loss, sampler, and optimizer.
  params_init, loss_fn = hk.without_apply_rng(hk.transform(sequence_loss))
  _, sample_fn = hk.without_apply_rng(hk.transform(sample))
  opt_init, _ = make_optimizer()

  loss_fn = jax.jit(loss_fn)
  sample_fn = jax.jit(sample_fn, static_argnums=[3])

  # Initialize training state.
  rng = hk.PRNGSequence(SEED.value)
  initial_params = params_init(next(rng), next(train_data))
  initial_opt_state = opt_init(initial_params)
  state = TrainingState(params=initial_params, opt_state=initial_opt_state)

  # Training loop.
  for step in range(TRAINING_STEPS.value + 1):
    # Do a batch of SGD.
    train_batch = next(train_data)
    state = update(state, train_batch)

    # Periodically generate samples.
    if step % SAMPLING_INTERVAL.value == 0:
      context = train_batch['input'][:, 0]  # First element of training batch.
      assert context.ndim == 1
      rng_key = next(rng)
      samples = sample_fn(state.params, rng_key, context, SAMPLE_LENGTH.value)

      prompt = dataset.decode(context)
      continuation = dataset.decode(samples)

      logging.info('Prompt: %s', prompt)
      logging.info('Continuation: %s', continuation)

    # Periodically evaluate training and test loss.
    if step % EVALUATION_INTERVAL.value == 0:
      for split, ds in eval_data.items():
        eval_batch = next(ds)
        loss = loss_fn(state.params, eval_batch)
        logging.info({
            'step': step,
            'loss': float(loss),
            'split': split,
        })


if __name__ == '__main__':
  app.run(main)
