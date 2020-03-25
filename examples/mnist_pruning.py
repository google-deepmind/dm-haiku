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
"""MNIST classifier with pruning as in https://arxiv.org/abs/1710.01878 ."""

import functools
from typing import Any, Callable, Generator, Mapping, Sequence, Text, Tuple

from absl import app
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

OptState = Any
Batch = Mapping[str, np.ndarray]
Predicate = Callable[[Text, Text, jnp.ndarray], bool]
PredicateMap = Mapping[Predicate, jnp.ndarray]
ModuleSparsity = Sequence[Tuple[Predicate, jnp.ndarray]]


def _topk_mask(value: jnp.ndarray, density_fraction: float) -> jnp.ndarray:
  """Return a mask with 1s marking the top fraction of value.

  Note: This routine takes care to make sure that ties are handled without
  bias toward smaller indices.  This can be a problem when pruning large
  embedding matrices, or global pruning where all parameters in the model
  are concatenated together and pruned at once.

  Args:
    value: An array. Must contain sortable values (i.e. not complex).
    density_fraction: A float. What fraction of value should be kept.

  Returns:
    A mask containing 1s where the topk elements of value are. k is
    determined based on density_fraction and the size of value.
  """

  def _topk_mask_internal(value, density_fraction):
    assert value.ndim == 1
    indices = jnp.argsort(value)
    k = jnp.round(density_fraction * jnp.size(value)).astype(jnp.int32)
    mask = jnp.greater_equal(np.arange(value.size), value.size - k)
    mask = jax.ops.index_update(jnp.zeros_like(mask), indices, mask)
    return mask.astype(np.int32)

  # shuffle value so that identical values aren't always pruned
  # with a bias to lower indices
  orig_shape = value.shape
  value = jnp.reshape(value, -1)
  shuffled_indices = jax.random.shuffle(
      jax.random.PRNGKey(42), jnp.arange(0, jnp.size(value), dtype=jnp.int32))

  shuffled_mask = _topk_mask_internal(value[shuffled_indices], density_fraction)
  mask = jax.ops.index_update(
      jnp.zeros_like(shuffled_mask), shuffled_indices, shuffled_mask)
  mask = jnp.reshape(mask, orig_shape)
  return mask


def zhugupta_func(progress: jnp.ndarray) -> jnp.ndarray:
  """From 'To Prune or Not To Prune."""
  return 1. - (1. - progress)**3


def _create_partitions(
    module_sparsity: ModuleSparsity, params: hk.Params
) -> Tuple[Sequence[hk.Params], Sequence[jnp.ndarray], hk.Params]:
  """Partition params based on sparsity_predicate_map.

  Args:
    module_sparsity:  A Sequence of (Predicate, float) pairs.  Predicate
      functions take module_name, name, value as arguments. The floats are the
      sparsity level to apply to leaves matching Predicate.
    params: A Haiku param.

  Returns:
    A tuple containing:
      - A list of len(module_sparsity), where each element is a disjoint subset
        of the `params` to be pruned.
      - A list of len(module_sparsity) where each element is the sparsity level.
      - The remaining elements of `params` not being pruned such that the union
        of the first list and this element contains the elements of `params`.
  """
  list_of_trees = []
  sparsity_list = []

  tail = params
  # Greedily match so that no parameter can be matched more than once
  for predicate, sparsity in module_sparsity:
    head, tail = hk.data_structures.partition(predicate, tail)
    list_of_trees.append(head)
    sparsity_list.append(sparsity)

  return list_of_trees, sparsity_list, tail


def sparsity_ignore(m: Text, n: Text, v: jnp.ndarray) -> bool:
  """Any parameter matching these conditions should generally not be pruned."""
  # n == 'b' when param is a bias
  return n == "b" or v.ndim == 1 or "batchnorm" in m or "batch_norm" in m


@functools.partial(jax.jit, static_argnums=2)
def apply_mask(params: hk.Params, masks: hk.Params,
               module_sparsity: ModuleSparsity) -> hk.Params:
  """Apply existing masks to params based on sparsity_predicate_map.

  Some of params may not be masked depending on the content of
  module_sparsity.  masks must have the same structure as implied by
  module_sparsity.

  Args:
    params: Tree to mask, can be a superset of masks.
    masks: Tree of masks to apply to params.  This must match the result of
      applying module_sparsity to params.
    module_sparsity: A dictionary mapping predicates to sparsity levels. Any
      leaf matching a predicate key will be pruned to the resulting sparsity
      level.

  Returns:
    A tree of masked params.
  """
  params_to_prune, _, params_no_prune = _create_partitions(
      module_sparsity, params)
  pruned_params = []
  for value, mask in zip(params_to_prune, masks):
    pruned_params.append(
        jax.tree_util.tree_multimap(lambda x, y: x * y, value, mask))
  params = hk.data_structures.merge(*pruned_params, params_no_prune)
  return params


@functools.partial(jax.jit, static_argnums=2)
def update_mask(params: hk.Params, sparsity_fraction: float,
                module_sparsity: ModuleSparsity) -> jnp.ndarray:
  """Generate masks based on module_sparsity and sparsity_fraction."""
  params_to_prune, sparsities, _ = _create_partitions(module_sparsity, params)
  masks = []

  def map_fn(x: jnp.ndarray, sparsity: float) -> jnp.ndarray:
    return _topk_mask(jnp.abs(x), 1. - sparsity * sparsity_fraction)

  for tree, sparsity in zip(params_to_prune, sparsities):
    map_fn_sparsity = jax.tree_util.Partial(map_fn, sparsity=sparsity)
    mask = jax.tree_util.tree_map(map_fn_sparsity, tree)
    masks.append(mask)
  return masks


@jax.jit
def get_sparsity(params: hk.Params):
  """Calculate the total sparsity and tensor-wise sparsity of params."""
  total_params = sum(jnp.size(x) for x in jax.tree_leaves(params))
  total_nnz = sum(jnp.sum(x != 0.) for x in jax.tree_leaves(params))
  leaf_sparsity = jax.tree_map(lambda x: jnp.sum(x == 0) / jnp.size(x), params)
  return total_params, total_nnz, leaf_sparsity


def net_fn(batch: Batch) -> jnp.ndarray:
  """Standard LeNet-300-100 MLP network."""
  x = batch["image"].astype(jnp.float32) / 255.
  mlp = hk.Sequential([
      hk.Flatten(),
      hk.Linear(300), jax.nn.relu,
      hk.Linear(100), jax.nn.relu,
      hk.Linear(10),
  ])
  return mlp(x)


def load_dataset(
    split: str,
    *,
    is_training: bool,
    batch_size: int,
) -> Generator[Batch, None, None]:
  """Loads the dataset as a generator of batches."""
  ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return tfds.as_numpy(ds)


def main(_):
  # Make the network and optimiser.
  net = hk.transform(net_fn)
  opt = optix.adam(1e-3)

  # Define layerwise sparsities
  def module_matching(s):

    def match_func(m, n, k):
      return m.endswith(s) and not sparsity_ignore(m, n, k)

    return match_func

  module_sparsity = ((module_matching("linear"), 0.98),
                     (module_matching("linear_1"), 0.9))

  # Training loss (cross-entropy).
  @jax.jit
  def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, batch)
    labels = hk.one_hot(batch["label"], 10)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent + 1e-4 * l2_loss

  # Evaluation metric (classification accuracy).
  @jax.jit
  def accuracy(params: hk.Params, batch: Batch) -> jnp.ndarray:
    predictions = net.apply(params, batch)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

  @jax.jit
  def get_updates(
      params: hk.Params,
      opt_state: OptState,
      batch: Batch,
  ) -> Tuple[hk.Params, OptState]:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(params, batch)
    updates, opt_state = opt.update(grads, opt_state)
    return updates, opt_state

  # We maintain avg_params, the exponential moving average of the "live" params.
  # avg_params is used only for evaluation.
  # For more, see: https://doi.org/10.1137/0330046
  @jax.jit
  def ema_update(
      avg_params: hk.Params,
      new_params: hk.Params,
      epsilon: float = 0.001,
  ) -> hk.Params:
    return jax.tree_multimap(lambda p1, p2: (1 - epsilon) * p1 + epsilon * p2,
                             avg_params, new_params)

  # Make datasets.
  train = load_dataset("train", is_training=True, batch_size=100)
  train_eval = load_dataset("train", is_training=False, batch_size=10000)
  test_eval = load_dataset("test", is_training=False, batch_size=10000)

  # Implemenation note: It is possible to avoid pruned_params and just use
  # a single params which progressively gets pruned.  The updates also don't
  # need to masked in such an implementation.  The current implementation
  # attempts to mimic the way the current TF implementation which allows for
  # previously inactivated connections to become active again if active values
  # drop below their value.

  # Initialize network and optimiser; note we draw an input to get shapes.
  pruned_params = params = avg_params = net.init(
      jax.random.PRNGKey(42), next(train))

  masks = update_mask(params, 0., module_sparsity)
  opt_state = opt.init(params)

  # Train/eval loop.
  for step in range(10001):
    if step % 1000 == 0:
      # Periodically evaluate classification accuracy on train & test sets.
      avg_params = apply_mask(avg_params, masks, module_sparsity)
      train_accuracy = accuracy(avg_params, next(train_eval))
      test_accuracy = accuracy(avg_params, next(test_eval))
      print(f"[Step {step}] Train / Test accuracy: "
            f"{train_accuracy:.3f} / {test_accuracy:.3f}.")
      total_params, total_nnz, per_layer_sparsities = get_sparsity(avg_params)
      print(f"Non-zero params / Total: {total_nnz} / {total_params}; "
            f"Total Sparsity: {1. - total_nnz / total_params:.3f}")

    # Do SGD on a batch of training examples.
    pruned_params = apply_mask(params, masks, module_sparsity)
    updates, opt_state = get_updates(pruned_params, opt_state, next(train))
    # applying a straight-through estimator here (that is not masking
    # the updates) leads to much worse performance.
    updates = apply_mask(updates, masks, module_sparsity)
    params = optix.apply_updates(params, updates)
    # we start pruning at iteration 1000 and end at iteration 8000
    progress = min(max((step - 1000.) / 8000., 0.), 1.)
    if step % 200 == 0:
      sparsity_fraction = zhugupta_func(progress)
      masks = update_mask(params, sparsity_fraction, module_sparsity)
    avg_params = ema_update(avg_params, params)
  print(per_layer_sparsities)


if __name__ == "__main__":
  app.run(main)
