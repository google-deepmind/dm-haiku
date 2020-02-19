# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""MNIST classifier example."""

from typing import Any, Tuple

from absl import app
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

OptState = Any
NUM_DIGITS = 10  # MNIST.


def net_fn(x: np.ndarray) -> jnp.DeviceArray:
  """Simple MLP network."""
  mlp = hk.Sequential([
      lambda x: x.astype(jnp.float32) / 255.,
      hk.Flatten(),
      hk.Linear(100),
      jax.nn.relu,
      hk.Linear(100),
      jax.nn.relu,
      hk.Linear(NUM_DIGITS),
      jax.nn.log_softmax,
  ])

  return mlp(x)


def get_datasets(train_batch_size: int, eval_batch_size: int):
  """Creates MNIST datasets as iterators of NumPy arrays."""
  ds = lambda s: tfds.load("mnist:3.*.*", split=s, as_supervised=True).cache()

  train_ds = ds(tfds.Split.TRAIN).repeat().shuffle(1000).batch(train_batch_size)
  train_eval_ds = ds(tfds.Split.TRAIN).repeat().batch(eval_batch_size)
  test_eval_ds = ds(tfds.Split.TEST).repeat().batch(eval_batch_size)

  return [tfds.as_numpy(d) for d in (train_ds, test_eval_ds, train_eval_ds)]


def main(_):

  # Make the network and optimiser.
  net = hk.transform(net_fn)
  opt_init, opt_update = optix.adam(1e-4)

  # Training loss (cross-entropy).
  @jax.jit
  def loss(
      params: hk.Params,
      inputs: np.ndarray,
      targets: np.ndarray,
  ) -> jnp.DeviceArray:
    assert targets.dtype == np.int32
    batch_size = inputs.shape[0]
    log_probs = net.apply(params, inputs)
    return -jnp.sum(hk.one_hot(targets, NUM_DIGITS) * log_probs) / batch_size

  # Evaluation metric (classification accuracy).
  @jax.jit
  def accuracy(
      params: hk.Params,
      inputs: np.ndarray,
      targets: np.ndarray,
  ) -> jnp.DeviceArray:
    predictions = net.apply(params, inputs)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == targets)

  @jax.jit
  def update(
      params: hk.Params,
      opt_state: OptState,
      inputs: np.ndarray,
      targets: np.ndarray,
  ) -> Tuple[hk.Params, OptState]:
    """Learning rule (stochastic gradient descent)."""
    _, gradient = jax.value_and_grad(loss)(params, inputs, targets)
    updates, opt_state = opt_update(gradient, opt_state)
    new_params = optix.apply_updates(params, updates)
    return new_params, opt_state

  # We maintain avg_params, the exponential moving average of the "live" params.
  # avg_params is used only for evaluation.
  # For more, see: https://doi.org/10.1137/0330046
  @jax.jit
  def ema_update(
      avg_params: hk.Params,
      new_params: hk.Params,
      epsilon: float = 0.99,
  ) -> hk.Params:
    return jax.tree_multimap(lambda p1, p2: (1 - epsilon) * p1 + epsilon * p2,
                             avg_params, new_params)

  # Make datasets.
  train, test_eval, train_eval = get_datasets(
      train_batch_size=32, eval_batch_size=1000)

  # Initialize network and optimiser; note we draw an input to get shapes.
  params = avg_params = net.init(jax.random.PRNGKey(42), next(train)[0])
  opt_state = opt_init(params)

  # Train/eval loop.
  for step in range(20000):
    if step % 1000 == 0:
      # Periodically evaluate classification accuracy on train & test sets.
      inputs, targets = next(train_eval)
      train_accuracy = accuracy(avg_params, inputs, targets)
      inputs, targets = next(test_eval)
      test_accuracy = accuracy(avg_params, inputs, targets)
      print(f"[Step {step}] Train accuracy: {train_accuracy}.")
      print(f"[Step {step}] Test accuracy: {test_accuracy}.")

    # Do SGD on a batch of training examples.
    inputs, targets = next(train)
    params, opt_state = update(params, opt_state, inputs, targets)
    avg_params = ema_update(avg_params, params)


if __name__ == "__main__":
  app.run(main)
