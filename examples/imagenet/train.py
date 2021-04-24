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
import timeit
from typing import Iterable, Mapping, NamedTuple, Tuple

from absl import app
from absl import flags
from absl import logging
import haiku as hk
from examples.imagenet import dataset
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import tree

# Hyper parameters.
SPLITS = ('TRAIN', 'TRAIN_AND_VALID', 'VALID', 'TEST')
flags.DEFINE_integer('eval_batch_size', 1000, help='')
flags.DEFINE_enum('eval_split', 'TEST', SPLITS, help='')
flags.DEFINE_float('model_bn_decay', 0.9, help='')
flags.DEFINE_bool('model_resnet_v2', True, help='')
flags.DEFINE_float('optimizer_momentum', 0.9, help='')
flags.DEFINE_bool('optimizer_use_nesterov', True, help='')
flags.DEFINE_integer('train_device_batch_size', 128, help='')
flags.DEFINE_integer('train_eval_every', -1, help='')
flags.DEFINE_integer('train_init_random_seed', 42, help='')
flags.DEFINE_integer('train_log_every', 100, help='')
flags.DEFINE_integer('train_epochs', 90, help='')
flags.DEFINE_integer('train_lr_warmup_epochs', 5, help='')
flags.DEFINE_float('train_lr_init', 0.1, help='')
flags.DEFINE_float('train_smoothing', .1, lower_bound=0, upper_bound=1, help='')
flags.DEFINE_enum('train_split', 'TRAIN_AND_VALID', SPLITS, help='')
flags.DEFINE_float('train_weight_decay', 1e-4, help='')
flags.DEFINE_string('mp_policy', 'p=f32,c=f32,o=f32', help='')
flags.DEFINE_string('mp_bn_policy', 'p=f32,c=f32,o=f32', help='')
flags.DEFINE_enum('mp_scale_type', 'NoOp', ['NoOp', 'Static', 'Dynamic'],
                  help='')
flags.DEFINE_float('mp_scale_value', 2 ** 15, help='')
flags.DEFINE_bool('mp_skip_nonfinite', False, help='')
flags.DEFINE_bool('dataset_transpose', False, help='')
flags.DEFINE_bool('dataset_zeros', False, help='')
FLAGS = flags.FLAGS

Scalars = Mapping[str, jnp.ndarray]


class TrainState(NamedTuple):
  params: hk.Params
  state: hk.State
  opt_state: optax.OptState
  loss_scale: jmp.LossScale

get_policy = lambda: jmp.get_policy(FLAGS.mp_policy)
get_bn_policy = lambda: jmp.get_policy(FLAGS.mp_bn_policy)


def get_initial_loss_scale() -> jmp.LossScale:
  cls = getattr(jmp, f'{FLAGS.mp_scale_type}LossScale')
  return cls(FLAGS.mp_scale_value) if cls is not jmp.NoOpLossScale else cls()


def _forward(
    batch: dataset.Batch,
    is_training: bool,
) -> jnp.ndarray:
  """Forward application of the resnet."""
  images = batch['images']
  if FLAGS.dataset_transpose:
    # See note in dataset.py if you are curious about this.
    images = jnp.transpose(images, (3, 0, 1, 2))  # HWCN -> NHWC
  net = hk.nets.ResNet50(1000,
                         resnet_v2=FLAGS.model_resnet_v2,
                         bn_config={'decay_rate': FLAGS.model_bn_decay})
  return net(images, is_training=is_training)

# Transform our forwards function into a pair of pure functions.
forward = hk.transform_with_state(_forward)


def lr_schedule(step: jnp.ndarray) -> jnp.ndarray:
  """Cosine learning rate schedule."""
  train_split = dataset.Split.from_string(FLAGS.train_split)

  total_batch_size = FLAGS.train_device_batch_size * jax.device_count()
  steps_per_epoch = train_split.num_examples / total_batch_size
  warmup_steps = FLAGS.train_lr_warmup_epochs * steps_per_epoch
  training_steps = FLAGS.train_epochs * steps_per_epoch

  lr = FLAGS.train_lr_init * total_batch_size / 256
  scaled_step = (jnp.maximum(step - warmup_steps, 0) /
                 (training_steps - warmup_steps))
  lr *= 0.5 * (1.0 + jnp.cos(jnp.pi * scaled_step))
  if warmup_steps:
    lr *= jnp.minimum(step / warmup_steps, 1.0)
  return lr


def make_optimizer() -> optax.GradientTransformation:
  """SGD with nesterov momentum and a custom lr schedule."""
  return optax.chain(
      optax.trace(
          decay=FLAGS.optimizer_momentum,
          nesterov=FLAGS.optimizer_use_nesterov),
      optax.scale_by_schedule(lr_schedule), optax.scale(-1))


def l2_loss(params: Iterable[jnp.ndarray]) -> jnp.ndarray:
  return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


def loss_fn(
    params: hk.Params,
    state: hk.State,
    loss_scale: jmp.LossScale,
    batch: dataset.Batch,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, hk.State]]:
  """Computes a regularized loss for the given batch."""
  logits, state = forward.apply(params, state, None, batch, is_training=True)
  labels = jax.nn.one_hot(batch['labels'], 1000)
  if FLAGS.train_smoothing:
    labels = optax.smooth_labels(labels, FLAGS.train_smoothing)
  loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()
  l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params)
               if 'batchnorm' not in mod_name]
  loss = loss + FLAGS.train_weight_decay * l2_loss(l2_params)
  return loss_scale.scale(loss), (loss, state)


@functools.partial(jax.pmap, axis_name='i', donate_argnums=(0,))
def train_step(
    train_state: TrainState,
    batch: dataset.Batch,
) -> Tuple[TrainState, Scalars]:
  """Applies an update to parameters and returns new state."""
  params, state, opt_state, loss_scale = train_state
  grads, (loss, new_state) = (
      jax.grad(loss_fn, has_aux=True)(params, state, loss_scale, batch))

  # Grads are in "param_dtype" (likely F32) here. We cast them back to the
  # compute dtype such that we do the all-reduce below in the compute precision
  # (which is typically lower than the param precision).
  policy = get_policy()
  grads = policy.cast_to_compute(grads)
  grads = loss_scale.unscale(grads)

  # Taking the mean across all replicas to keep params in sync.
  grads = jax.lax.pmean(grads, axis_name='i')

  # We compute our optimizer update in the same precision as params, even when
  # doing mixed precision training.
  grads = policy.cast_to_param(grads)

  # Compute and apply updates via our optimizer.
  updates, new_opt_state = make_optimizer().update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)

  if FLAGS.mp_skip_nonfinite:
    grads_finite = jmp.all_finite(grads)
    loss_scale = loss_scale.adjust(grads_finite)
    new_params, new_state, new_opt_state = jmp.select_tree(
        grads_finite,
        (new_params, new_state, new_opt_state),
        (params, state, opt_state))

  # Scalars to log (note: we log the mean across all hosts/devices).
  scalars = {'train_loss': loss, 'loss_scale': loss_scale.loss_scale}
  if FLAGS.mp_skip_nonfinite:
    scalars['grads_finite'] = grads_finite
  state, scalars = jmp.cast_to_full((state, scalars))
  scalars = jax.lax.pmean(scalars, axis_name='i')
  train_state = TrainState(new_params, new_state, new_opt_state, loss_scale)
  return train_state, scalars


def initial_state(rng: jnp.ndarray, batch: dataset.Batch) -> TrainState:
  """Computes the initial network state."""
  params, state = forward.init(rng, batch, is_training=True)
  opt_state = make_optimizer().init(params)
  loss_scale = get_initial_loss_scale()
  return TrainState(params, state, opt_state, loss_scale)


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
  if split.num_examples % FLAGS.eval_batch_size:
    raise ValueError(f'Eval batch size {FLAGS.eval_batch_size} must be a '
                     f'multiple of {split} num examples {split.num_examples}')

  # Params/state are sharded per-device during training. We just need the copy
  # from the first device (since we do not pmap evaluation at the moment).
  params, state = jax.tree_map(lambda x: x[0], (params, state))
  test_dataset = dataset.load(split,
                              is_training=False,
                              batch_dims=[FLAGS.eval_batch_size],
                              transpose=FLAGS.dataset_transpose,
                              zeros=FLAGS.dataset_zeros)
  correct = jnp.array(0)
  total = 0
  for batch in test_dataset:
    correct += eval_batch(params, state, batch)
    total += batch['labels'].shape[0]
  assert total == split.num_examples, total
  return {'top_1_acc': correct.item() / total}


@contextlib.contextmanager
def time_activity(activity_name: str):
  logging.info('[Timing] %s start.', activity_name)
  start = timeit.default_timer()
  yield
  duration = timeit.default_timer() - start
  logging.info('[Timing] %s finished (Took %.2fs).', activity_name, duration)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  FLAGS.alsologtostderr = True

  train_split = dataset.Split.from_string(FLAGS.train_split)
  eval_split = dataset.Split.from_string(FLAGS.eval_split)

  # The total batch size is the batch size accross all hosts and devices. In a
  # multi-host training setup each host will only see a batch size of
  # `total_train_batch_size / jax.host_count()`.
  total_train_batch_size = FLAGS.train_device_batch_size * jax.device_count()
  num_train_steps = (
      (train_split.num_examples * FLAGS.train_epochs) // total_train_batch_size)

  local_device_count = jax.local_device_count()
  train_dataset = dataset.load(
      train_split,
      is_training=True,
      batch_dims=[local_device_count, FLAGS.train_device_batch_size],
      dtype=get_policy().compute_dtype,
      transpose=FLAGS.dataset_transpose,
      zeros=FLAGS.dataset_zeros)

  # Assign mixed precision policies to modules. Note that when training in f16
  # we keep BatchNorm in  full precision. When training with bf16 you can often
  # use bf16 for BatchNorm.
  mp_policy = get_policy()
  bn_policy = get_bn_policy().with_output_dtype(mp_policy.compute_dtype)
  # NOTE: The order we call `set_policy` doesn't matter, when a method on a
  # class is called the policy for that class will be applied, or it will
  # inherit the policy from its parent module.
  hk.mixed_precision.set_policy(hk.BatchNorm, bn_policy)
  hk.mixed_precision.set_policy(hk.nets.ResNet50, mp_policy)

  if jax.default_backend() == 'gpu':
    # TODO(tomhennigan): This could be removed if XLA:GPU's allocator changes.
    train_dataset = dataset.double_buffer(train_dataset)

  # For initialization we need the same random key on each device.
  rng = jax.random.PRNGKey(FLAGS.train_init_random_seed)
  rng = jnp.broadcast_to(rng, (local_device_count,) + rng.shape)
  # Initialization requires an example input.
  batch = next(train_dataset)
  train_state = jax.pmap(initial_state)(rng, batch)

  # Print a useful summary of the execution of our module.
  summary = hk.experimental.tabulate(train_step)(train_state, batch)
  for line in summary.split('\n'):
    logging.info(line)

  eval_every = FLAGS.train_eval_every
  log_every = FLAGS.train_log_every

  with time_activity('train'):
    for step_num in range(num_train_steps):
      # Take a single training step.
      with jax.profiler.StepTraceAnnotation('train', step_num=step_num):
        batch = next(train_dataset)
        train_state, train_scalars = train_step(train_state, batch)

      # By default we do not evaluate during training, but you can configure
      # this with a flag.
      if eval_every > 0 and step_num and step_num % eval_every == 0:
        with time_activity('eval during train'):
          eval_scalars = evaluate(eval_split,
                                  train_state.params, train_state.state)
        logging.info('[Eval %s/%s] %s', step_num, num_train_steps, eval_scalars)

      # Log progress at fixed intervals.
      if step_num and step_num % log_every == 0:
        train_scalars = jax.tree_map(lambda v: np.mean(v).item(),
                                     jax.device_get(train_scalars))
        logging.info('[Train %s/%s] %s',
                     step_num, num_train_steps, train_scalars)

  # Once training has finished we run eval one more time to get final results.
  with time_activity('final eval'):
    eval_scalars = evaluate(eval_split, train_state.params, train_state.state)
  logging.info('[Eval FINAL]: %s', eval_scalars)

if __name__ == '__main__':
  dataset.check_versions()
  app.run(main)
