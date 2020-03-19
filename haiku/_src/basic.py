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
"""Basic Haiku modules and functions."""

import functools
from typing import Any, Callable, Iterable, Optional, Text

from haiku._src import base
from haiku._src import initializers
from haiku._src import module
import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import tree


# Utility and activation functions.
def one_hot(indices, num_classes, dtype=jnp.float32):
  """Returns a one-hot version of indices.

  Args:
    indices: A tensor of indices.
    num_classes: Number of classes in the one-hot dimension.
    dtype: The dtype.

  Returns:
    The one-hot tensor. If indices' shape is [A, B, ...], shape is
      [A, B, ... num_classes].
  """

  return jnp.array(
      indices[..., jnp.newaxis] == jnp.arange(num_classes), dtype=dtype)


def multinomial(rng, logits, num_samples):
  """Draws samples from a multinomial distribution.

  Args:
    rng: A JAX PRNGKey.
    logits: Unnormalized log-probabilities, of shape [batch_size, categories] or
      [categories].
    num_samples: Number of samples to draw.

  Returns:
    Chosen categories, of shape [batch_size, num_samples] or [num_samples].
  """
  # NOTE(tycai): Currently, tf.multinomial uses CDF for non-XLA CPU only.
  # We may want to switch to the Gumbel trick as used in XLA.
  if len(logits.shape) > 2 or not logits.shape:
    raise ValueError("Logits must be rank-1 or rank-2.")
  probs = jax.nn.softmax(logits)
  probs = jnp.cumsum(probs, axis=-1)
  # Special-case num_samples == 1 due to TPU padding, as in TF2XLA.
  # https://github.com/tensorflow/tensorflow/blob/b1608511d5a50d05825c4025b0c347e8689a241f/tensorflow/compiler/tf2xla/kernels/categorical_op.cc#L79
  if num_samples == 1:
    a = jax.random.uniform(rng, logits.shape[:-1] + (1,))
    out = jnp.argmin(a > probs, axis=-1)
    return out[..., None]
  else:
    a = jax.random.uniform(rng, (num_samples,) + logits.shape[:-1] + (1,))
    out = jnp.argmin(a > probs, axis=-1)
    return jnp.transpose(out)


# Common modules.
class Sequential(module.Module):
  """Sequentially calls the given list of layers.

  Note that `Sequential` is limited in the range of possible architectures
  it can handle. This is a deliberate design decision; `Sequential` is only
  meant to be used for the simple case of fusing together modules/ops where
  the input of a particular module/op is the output of the previous one.

  Another restriction is that it is not possible to have extra arguments in the
  `__call__` method that are passed to the constituents of the module - for
  example, if there is a `BatchNorm` module in `Sequential` and the user wishes
  to switch the `is_training` flag. If this is the desired use case, the
  recommended solution is to subclass `snt.Module` and implement `__call__`:

      >>> class CustomModule(hk.Module):
      ...   def __init__(self, name=None):
      ...     super().__init__(name=name)
      ...     self.conv2d = hk.Conv2D(32, 4, 2)
      ...     self.bn = hk.BatchNorm(True, True, 0.9)
      ...
      ...   def __call__(self, inputs, is_training):
      ...     outputs = self.conv2d(inputs)
      ...     outputs = self.bn(outputs, is_training=is_training)
      ...     outputs = jax.nn.relu(outputs)
      ...     return outputs
  """

  def __init__(self,
               layers: Iterable[Callable[..., Any]] = None,
               name: Optional[Text] = None):
    super(Sequential, self).__init__(name=name)
    self.layers = layers

  def __call__(self, inputs, *args, **kwargs):
    """Connects all layers. *args and **kwargs are passed to the first layer."""
    outputs = inputs
    for i, layer in enumerate(self.layers):
      if i == 0:
        outputs = layer(outputs, *args, **kwargs)
      else:
        outputs = layer(outputs)
    return outputs


class Linear(module.Module):
  """Linear module."""

  def __init__(self,
               output_size: int,
               with_bias: bool = True,
               w_init: Optional[base.Initializer] = None,
               b_init: Optional[base.Initializer] = None,
               name: Optional[Text] = None):
    """Constructs the Linear module.

    Args:
      output_size: Output dimensionality.
      with_bias: Whether to add a bias to the output.
      w_init: Optional initializer for weights. By default, uses random values
        from truncated normal, with stddev `1 / sqrt(fan_in)`. See
        https://arxiv.org/abs/1502.03167v3.
      b_init: Optional initializer for bias. By default, zero.
      name: Name of the module.
    """
    super(Linear, self).__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros

  def __call__(self, inputs):
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    self.input_size = inputs.shape[-1]
    default_stddev = 1. / np.sqrt(self.input_size)
    w_init = self.w_init or initializers.TruncatedNormal(stddev=default_stddev)

    w = base.get_parameter("w", [self.input_size, self.output_size],
                           inputs.dtype, init=w_init)
    out = jnp.dot(inputs, w)
    if self.with_bias:
      b = base.get_parameter("b", [self.output_size], inputs.dtype,
                             init=self.b_init)
      out += jnp.broadcast_to(b, out.shape)
    return out


def ndim_at_least(x, num_dims):
  if x is None:
    return False
  x = jnp.asarray(x)
  return len(x.shape) >= num_dims


def arbitrary_mergeable_leaf(min_num_dims, args, kwargs):
  for a in tree.flatten(args):
    if ndim_at_least(a, min_num_dims):
      return a
  for k in tree.flatten(kwargs):
    if ndim_at_least(k, min_num_dims):
      return k
  # Couldn't find a satisfactory leaf.
  return None


def merge_leading_dims(x, num_dims):
  """Merge leading dimensions."""
  # Don't merge if there aren't dimensions to merge.
  if not ndim_at_least(x, num_dims):
    return x

  new_shape = (jnp.prod(x.shape[:num_dims]),) + x.shape[num_dims:]
  return jnp.reshape(x, new_shape)


def split_leading_dim(x, to_dim):
  if x is None:
    return None
  new_shape = to_dim + x.shape[1:]
  return jnp.reshape(x, new_shape)


class BatchApply(object):
  """Temporarily merges leading dimensions of input tensors.

  Merges the leading dimensions of a tensor into a single dimension, runs the
  given callable, then splits the leading dimension of the result to match the
  input.

  Input arrays whose rank is smaller than the number of dimensions to collapse
  are passed unmodified.

  This may be useful for applying a module to each timestep of e.g. a
  [Time, Batch, ...] array.

  For some `f`s and platforms, this may be more efficient than `jax.vmap`,
  especially when combined with other transformations like `jax.grad`.
  """

  def __init__(self, f, num_dims=2):
    """Constructs a BatchApply module.

    Args:
      f: The callable to be applied to the reshaped array.
      num_dims: The number of dimensions to merge.
    """
    self._f = f
    self.num_dims = num_dims

  def __call__(self, *args, **kwargs):
    example = arbitrary_mergeable_leaf(self.num_dims, args, kwargs)
    if example is None:
      msg = "BatchApply requires at least one input with ndim >= {}."
      raise ValueError(msg.format(self.num_dims))
    merge = lambda x: merge_leading_dims(x, self.num_dims)
    split = lambda x: split_leading_dim(x, example.shape[:self.num_dims])
    args = tree.map_structure(merge, args)
    kwargs = tree.map_structure(merge, kwargs)
    outputs = self._f(*args, **kwargs)
    return tree.map_structure(split, outputs)


def expand_apply(f, axis=0):
  """Wraps f to temporarily add a size-1 axis to its inputs.

  Syntactic sugar for:
  ```
      in = tree.map_structure(lambda t: np.expand_dims(t, axis=axis), in)
      out = f(in)
      out = tree.map_structure(lambda t: np.squeeze(t, axis=axis), out)
  ```

  This may be useful for applying a function built for [Time, Batch, ...] arrays
  to a single timestep.

  Args:
    f: The callable to be applied to the expanded inputs.
    axis: Where to add the extra axis.

  Returns:
    f, wrapped as described above.
  """
  if axis not in [0, -1]:
    raise ValueError("expand_apply currently only supports axis=0 or axis=-1.")

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    expand = lambda t: jnp.expand_dims(t, axis=axis)
    args = tree.map_structure(expand, args)
    kwargs = tree.map_structure(expand, kwargs)
    outputs = f(*args, **kwargs)
    return tree.map_structure(lambda t: jnp.squeeze(t, axis=axis), outputs)

  return wrapper


def dropout(rng, rate, x):
  """Randomly drop units in the input at a given rate.

  See: http://www.cs.toronto.edu/~hinton/absps/dropout.pdf

  Args:
    rng: The RNGKey.
    rate: Probability that each element of x is discarded. Must be a scalar in
      the range `[0, 1)`.
    x: The value to be dropped out.
  Returns:
    x, but dropped out and scaled by 1 / (1 - rate).
  """
  if rate < 0 or rate >= 1:
    raise ValueError("rate must be in [0, 1).")
  keep_rate = 1.0 - rate
  keep = jax.random.bernoulli(rng, keep_rate, shape=x.shape)
  return keep * x / keep_rate


def to_module(f):
  """Converts a function into a module.

  Sample usage:
  >>> def add_bias(x):
  ...   b = hk.get_parameter("b", [], init=hk.initializers.RandomNormal())
  ...   return x + b
  >>> Bias = hk.to_module(add_bias)
  >>> def net(x, y):
  ...   b = Bias()
  ...   # Bias x and y by the same amount.
  ...   return b(x) * b(y)

  Args:
    f: The function to convert.

  Returns:
    A module class which runs runs `f` when called.
  """

  class Wrapper(module.Module):

    def __init__(self, name=None):
      if name is None:
        name = f.__name__
      super(Wrapper, self).__init__(name=name)

    def __call__(self, *a, **k):
      return f(*a, **k)

  return Wrapper
