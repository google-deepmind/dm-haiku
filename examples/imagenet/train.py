# python3
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
"""ResNet50 on ImageNet2012."""

import contextlib
import functools
from typing import Iterable, Mapping, Tuple

from absl import app
from absl import flags
from absl import logging
import haiku as hk
from haiku.examples.imagenet import dataset
import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as np
import tree

# Hyper parameters.
SPLITS = ('TRAIN', 'TRAIN_AND_VALID', 'VALID', 'TEST')
flags.DEFINE_integer('eval_batch_size', 1000, help='')
flags.DEFINE_enum('eval_split', 'TEST', SPLITS, help='')
flags.DEFINE_float('model_bn_decay', 0.9, help='')
flags.DEFINE_bool('model_resnet_v2', True, help='')
flags.DEFINE_float('optimizer_momentum', 0.9, help='')
flags.DEFINE_bool('optimizer_use_nesterov', True, help='')
flags.DEFINE_integer('train_device_batch_size', 32, help='')
flags.DEFINE_integer('train_eval_every', -1, help='')
flags.DEFINE_integer('train_init_random_seed', 42, help='')
flags.DEFINE_integer('train_log_every', 1000, help='')
flags.DEFINE_enum('train_split', 'TRAIN_AND_VALID', SPLITS, help='')
flags.DEFINE_float('train_weight_decay', 1e-4, help='')
FLAGS = flags.FLAGS

# Types.
OptState = Tuple[optix.TraceState, optix.ScaleByScheduleState, optix.ScaleState]
Scalars = Mapping[str, jnp.ndarray]


def _forward(
    batch: dataset.Batch,
    is_training: bool,
) -> jnp.ndarray:
  """Forward application of the resnet."""
  net = hk.nets.ResNet50(1000,
                         resnet_v2=FLAGS.model_resnet_v2,
                         bn_config={'decay_rate': FLAGS.model_bn_decay})
  return net(batch['images'], is_training=is_training)

# Transform our forwards function into a pair of pure functions.
forward = hk.transform_with_state(_forward)


def lr_schedule(step: jnp.ndarray) -> jnp.ndarray:
  """Linear scaling rule optimized for 90 epochs."""
  train_split = dataset.Split.from_string(FLAGS.train_split)

  # See Section 5.1 of https://arxiv.org/pdf/1706.02677.pdf.
  total_batch_size = FLAGS.train_device_batch_size * jax.device_count()
  steps_per_epoch = train_split.num_examples / total_batch_size

  current_epoch = step / steps_per_epoch  # type: float
  lr = (0.1 * total_batch_size) / 256
  lr_linear_till = 5
  boundaries = jnp.array((30, 60, 80)) * steps_per_epoch
  values = jnp.array([1., 0.1, 0.01, 0.001]) * lr

  index = jnp.sum(boundaries < step)
  lr = jnp.take(values, index)
  return lr * jnp.minimum(1., current_epoch / lr_linear_till)


def make_optimizer():
  """SGD with nesterov momentum and a custom lr schedule."""
  return optix.chain(optix.trace(decay=FLAGS.optimizer_momentum,
                                 nesterov=FLAGS.optimizer_use_nesterov),
                     optix.scale_by_schedule(lr_schedule),
                     optix.scale(-1))


def l2_loss(params: Iterable[jnp.ndarray]) -> jnp.ndarray:
  return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


def softmax_cross_entropy(
    *,
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
  return -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)


def loss_fn(
    params: hk.Params,
    state: hk.State,
    batch: dataset.Batch,
) -> Tuple[jnp.ndarray, hk.State]:
  """Computes a regularized loss for the given batch."""
  logits, state = forward.apply(params, state, None, batch, is_training=True)
  labels = hk.one_hot(batch['labels'], 1000)
  cat_loss = jnp.mean(softmax_cross_entropy(logits=logits, labels=labels))
  l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params)
               if 'batchnorm' not in mod_name]
  reg_loss = FLAGS.train_weight_decay * l2_loss(l2_params)
  loss = cat_loss + reg_loss
  return loss, state


@functools.partial(jax.pmap, axis_name='i')
def train_step(
    params: hk.Params,
    state: hk.State,
    opt_state: OptState,
    batch: dataset.Batch,
) -> Tuple[hk.Params, hk.State, OptState, Scalars]:
  """Applies an update to parameters and returns new state."""
  (loss, state), grads = (
      jax.value_and_grad(loss_fn, has_aux=True)(params, state, batch))

  # Taking the mean across all replicas to keep params in sync.
  grads = jax.lax.pmean(grads, axis_name='i')

  # Compute and apply updates via our optimizer.
  updates, opt_state = make_optimizer().update(grads, opt_state)
  params = optix.apply_updates(params, updates)

  # Scalars to log (note: we log the mean across all hosts/devices).
  scalars = {'train_loss': loss}
  scalars = jax.lax.pmean(scalars, axis_name='i')

  return params, state, opt_state, scalars


def make_initial_state(
    rng: jnp.ndarray,
    batch: dataset.Batch,
) -> Tuple[hk.Params, hk.State, OptState]:
  """Computes the initial network state."""
  params, state = forward.init(rng, batch, is_training=True)
  opt_state = make_optimizer().init(params)
  return params, state, opt_state


# NOTE: We use `jit` not `pmap` here because we want to ensure that we see all
# eval data once and this is not easily satisfiable with pmap (e.g. n=3).
# TODO(tomhennigan) Find a solution to allow pmap of eval.
@jax.jit
def eval_batch(
    params: hk.Params,
    state: hk.State,
    batch: dataset.Batch,
) -> jnp.ndarray:
  """Evaluates a batch."""
  logits, _ = forward.apply(params, state, None, batch, is_training=False)
  predicted_label = jnp.argmax(logits, axis=-1)
  correct = jnp.sum(jnp.equal(predicted_label, batch['labels']))
  return correct.astype(jnp.float32)


def evaluate(
    split: dataset.Split,
    params: hk.Params,
    state: hk.State,
) -> Scalars:
  """Evaluates the model at the given params/state."""
  # Params/state are sharded per-device during training. We just need the copy
  # from the first device (since we do not pmap evaluation at the moment).
  params, state = jax.tree_map(lambda x: x[0], (params, state))
  test_dataset = dataset.load(split, batch_dims=[FLAGS.eval_batch_size])
  correct = jnp.array(0)
  total = 0
  for batch in test_dataset:
    correct += eval_batch(params, state, next(test_dataset))
    total += batch['images'].shape[0]
  return {'top_1_acc': correct.item() / total}


@contextlib.contextmanager
def time_activity(activity_name: str):
  logging.info(f'[Timing] {activity_name} start.')
  yield
  logging.info(f'[Timing] {activity_name} finished.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_split = dataset.Split.from_string(FLAGS.train_split)
  eval_split = dataset.Split.from_string(FLAGS.eval_split)

  # The total batch size is the batch size accross all hosts and devices. In a
  # multi-host training setup each host will only see a batch size of
  # `total_train_batch_size / jax.host_count()`.
  total_train_batch_size = FLAGS.train_device_batch_size * jax.device_count()
  num_train_steps = (train_split.num_examples * 90) // total_train_batch_size

  local_device_count = jax.local_device_count()
  train_dataset = dataset.load(
      train_split,
      batch_dims=[local_device_count, FLAGS.train_device_batch_size])

  # For initialization we need the same random key on each device.
  rng = jax.random.PRNGKey(FLAGS.train_init_random_seed)
  rng = jnp.broadcast_to(rng, (local_device_count,) + rng.shape)
  # Initialization requires an example input.
  batch = next(train_dataset)
  params, state, opt_state = jax.pmap(make_initial_state)(rng, batch)

  eval_every = FLAGS.train_eval_every
  log_every = FLAGS.train_log_every

  with time_activity('train'):
    for step_num in range(num_train_steps):
      # Take a single training step.
      params, state, opt_state, train_scalars = (
          train_step(params, state, opt_state, next(train_dataset)))

      # By default we do not evaluate during training, but you can configure
      # this with a flag.
      if eval_every > 0 and step_num and step_num % eval_every == 0:
        with time_activity('eval during train'):
          eval_scalars = evaluate(eval_split, params, state)
        logging.info(f'[Eval {step_num}/{num_train_steps}] {eval_scalars}')

      # Log progress at fixed intervals.
      if step_num and step_num % log_every == 0:
        train_scalars = jax.tree_map(lambda v: np.mean(v).item(),
                                     jax.device_get(train_scalars))
        logging.info(f'[Train {step_num}/{num_train_steps}] {train_scalars}')

  # Once training has finished we run eval one more time to get final results.
  with time_activity('final eval'):
    eval_scalars = evaluate(eval_split, params, state)
  logging.info(f'[Eval FINAL]: {eval_scalars}')

if __name__ == '__main__':
  app.run(main)
