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

from collections.abc import Callable, Iterable, Sequence
import functools
from typing import Any
from haiku._src import base
from haiku._src import initializers
from haiku._src import module
from haiku._src import typing
from haiku._src.typing import PRNGKey
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


# If you are forking replace this block with `import haiku as hk`.
# pylint: disable=invalid-name
class hk:
  get_parameter = base.get_parameter
  initializers = initializers
  Module = module.Module
  SupportsCall = typing.SupportsCall
# pylint: enable=invalid-name
del base, module, initializers, typing


# Utility and activation functions.
def one_hot(x, num_classes, dtype=jnp.float32):
  """Returns a one-hot version of indices.

  DEPRECATED: Use ``jax.nn.one_hot(x, num_classes).astype(dtype)`` instead.

  Args:
    x: A tensor of indices.
    num_classes: Number of classes in the one-hot dimension.
    dtype: The dtype.

  Returns:
    The one-hot tensor. If indices' shape is [A, B, ...], shape is
      [A, B, ... num_classes].
  """
  return jax.nn.one_hot(x, num_classes).astype(dtype)


def multinomial(rng, logits, num_samples):
  """Draws samples from a multinomial distribution.

  DEPRECATED: Use ``jax.random.categorical`` instead.

  Args:
    rng: A JAX PRNGKey.
    logits: Unnormalized log-probabilities, where last dimension is categories.
    num_samples: Number of samples to draw.

  Returns:
    Chosen categories, of shape ``logits.shape[:-1] + (num_samples,)``.
  """
  if num_samples != 1:
    shape = (num_samples,) + logits.shape[:-1]
  else:
    shape = None

  samples = jax.random.categorical(rng, logits, shape=shape)

  # Return expected shape.
  if num_samples != 1:
    return jnp.moveaxis(samples, 0, -1)
  else:
    return samples[..., None]


# Common modules.
class Sequential(hk.Module):
  """Sequentially calls the given list of layers.

  Note that :class:`Sequential` is limited in the range of possible
  architectures it can handle. This is a deliberate design decision;
  :class:`Sequential` is only meant to be used for the simple case of fusing
  together modules/ops where the input of a particular module/op is the output
  of the previous one.

  Another restriction is that it is not possible to have extra arguments in the
  :meth:`__call__` method that are passed to the constituents of the module -
  for example, if there is a :class:`BatchNorm` module in :class:`Sequential`
  and the user wishes to switch the ``is_training`` flag. If this is the desired
  use case, the recommended solution is to subclass :class:`Module` and
  implement ``__call__``:

      >>> class CustomModule(hk.Module):
      ...   def __call__(self, x, is_training):
      ...     x = hk.Conv2D(32, 4, 2)(x)
      ...     x = hk.BatchNorm(True, True, 0.9)(x, is_training)
      ...     x = jax.nn.relu(x)
      ...     return x
  """

  def __init__(
      self,
      layers: Iterable[Callable[..., Any]],
      name: str | None = None,
  ):
    super().__init__(name=name)
    self.layers = tuple(layers)

  def __call__(self, inputs, *args, **kwargs):
    """Calls all layers sequentially."""
    out = inputs
    for i, layer in enumerate(self.layers):
      if i == 0:
        out = layer(out, *args, **kwargs)
      else:
        out = layer(out)
    return out


class Linear(hk.Module):
  """Linear module."""

  def __init__(
      self,
      output_size: int,
      with_bias: bool = True,
      w_init: hk.initializers.Initializer | None = None,
      b_init: hk.initializers.Initializer | None = None,
      name: str | None = None,
  ):
    """Constructs the Linear module.

    Args:
      output_size: Output dimensionality.
      with_bias: Whether to add a bias to the output.
      w_init: Optional initializer for weights. By default, uses random values
        from truncated normal, with stddev ``1 / sqrt(fan_in)``. See
        https://arxiv.org/abs/1502.03167v3.
      b_init: Optional initializer for bias. By default, zero.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.input_size = None
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    self.b_init = b_init or jnp.zeros

  def __call__(
      self,
      inputs: jax.Array,
      *,
      precision: lax.Precision | None = None,
  ) -> jax.Array:
    """Computes a linear transform of the input."""
    if not inputs.shape:
      raise ValueError("Input must not be scalar.")

    input_size = self.input_size = inputs.shape[-1]
    output_size = self.output_size
    dtype = inputs.dtype

    w_init = self.w_init
    if w_init is None:
      stddev = 1. / np.sqrt(self.input_size)
      w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    w = hk.get_parameter("w", [input_size, output_size], dtype, init=w_init)

    out = jnp.dot(inputs, w, precision=precision)

    if self.with_bias:
      b = hk.get_parameter("b", [self.output_size], dtype, init=self.b_init)
      b = jnp.broadcast_to(b, out.shape)
      out = out + b

    return out


def ndim_at_least(x, num_dims):
  if not (isinstance(x, jax.Array) or isinstance(x, np.ndarray)):
    x = jnp.asarray(x)
  return x.ndim >= num_dims


def arbitrary_mergeable_leaf(min_num_dims, args, kwargs):
  for a in jax.tree.leaves(args):
    if ndim_at_least(a, min_num_dims):
      return a
  for k in jax.tree.leaves(kwargs):
    if ndim_at_least(k, min_num_dims):
      return k
  # Couldn't find a satisfactory leaf.
  return None


def merge_leading_dims(x, num_dims):
  """Merge leading dimensions."""
  # Don't merge if there aren't dimensions to merge.
  if not ndim_at_least(x, num_dims):
    return x

  # TODO(tomhennigan) Pass dtype here to account for empty slices.
  new_shape = (np.prod(x.shape[:num_dims]),) + x.shape[num_dims:]
  return x.reshape(new_shape)


def split_leading_dim(x, to_dim):
  new_shape = to_dim + x.shape[1:]
  return x.reshape(new_shape)


class BatchApply:
  r"""Temporarily merges leading dimensions of input tensors.

  Merges the leading dimensions of a tensor into a single dimension, runs the
  given callable, then splits the leading dimension of the result to match the
  input.

  Input arrays whose rank is smaller than the number of dimensions to collapse
  are passed unmodified.

  This may be useful for applying a module to each timestep of e.g. a
  ``[Time, Batch, ...]`` array.

  For some ``f``\ s and platforms, this may be more efficient than
  :func:`jax.vmap`, especially when combined with other transformations like
  :func:`jax.grad`.
  """

  def __init__(self, f, num_dims=2):
    """Constructs a :class:`BatchApply` module.

    Args:
      f: The callable to be applied to the reshaped array.
      num_dims: The number of dimensions to merge.
    """
    self._f = f
    self.num_dims = num_dims

  def __call__(self, *args, **kwargs):
    example = arbitrary_mergeable_leaf(self.num_dims, args, kwargs)
    if example is None:
      raise ValueError(
          "BatchApply requires at least one input with ndim >= "
          f"{self.num_dims}.")

    merge = lambda x: merge_leading_dims(x, self.num_dims)
    split = lambda x: split_leading_dim(x, example.shape[:self.num_dims])
    args = jax.tree.map(merge, args)
    kwargs = jax.tree.map(merge, kwargs)
    outputs = self._f(*args, **kwargs)
    return jax.tree.map(split, outputs)


def expand_apply(f, axis=0):
  """Wraps f to temporarily add a size-1 axis to its inputs.

  Syntactic sugar for::

      ins = jax.tree.map(lambda t: np.expand_dims(t, axis=axis), ins)
      out = f(ins)
      out = jax.tree.map(lambda t: np.squeeze(t, axis=axis), out)

  This may be useful for applying a function built for ``[Time, Batch, ...]``
  arrays to a single timestep.

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
    args = jax.tree.map(expand, args)
    kwargs = jax.tree.map(expand, kwargs)
    outputs = f(*args, **kwargs)
    return jax.tree.map(lambda t: jnp.squeeze(t, axis=axis), outputs)

  return wrapper


def dropout(
    rng: PRNGKey, rate: float, x: jax.Array, broadcast_dims: Sequence[int] = ()
) -> jax.Array:
  """Randomly drop units in the input at a given rate.

  See: http://www.cs.toronto.edu/~hinton/absps/dropout.pdf

  Args:
    rng: A JAX random key.
    rate: Probability that each element of ``x`` is discarded. Must be a scalar
      in the range ``[0, 1)``.
    x: The value to be dropped out.
    broadcast_dims: specifies dimensions that will share the same dropout mask.

  Returns:
    x, but dropped out and scaled by ``1 / (1 - rate)``.

  Note:
    This involves generating `x.size` pseudo-random samples from U([0, 1))
    computed with the full precision required to compare them with `rate`. When
    `rate` is a Python float, this is typically 32 bits, which is often more
    than what applications require. A work-around is to pass `rate` with a lower
    precision, e.g. using `np.float16(rate)`.
  """
  return dropout_impl(rng, rate, x, broadcast_dims=broadcast_dims)


# Separated out to support monkey patching.
def dropout_impl(
    rng: PRNGKey, rate: float, x: jax.Array, broadcast_dims: Sequence[int] = ()
) -> jax.Array:
  """See dropout."""
  try:
    if rate < 0 or rate >= 1:
      raise ValueError("rate must be in [0, 1).")

    if rate == 0.0:
      return x
  except jax.errors.ConcretizationTypeError:
    pass

  broadcast_shape = list(x.shape)
  for dim in broadcast_dims:
    if dim > len(broadcast_shape):
      raise ValueError("Broadcast dimension does not exist. Got dimension "
                       f"{dim} for shape {broadcast_shape}.")
    broadcast_shape[dim] = 1

  keep_rate = 1.0 - rate
  keep = jax.random.bernoulli(rng, keep_rate, shape=broadcast_shape)
  keep = jnp.broadcast_to(keep, x.shape)
  return keep * x / keep_rate


# TODO(tomhennigan): Fix internal tests and replace with `hk.SupportsCall`.
class CallableModule(hk.Module):

  def __call__(self, *args, **kwargs) -> Any:
    raise NotImplementedError


def to_module(f: Callable[..., Any]) -> type[CallableModule]:
  """Converts a function into a callable module class.

  Sample usage:

  >>> def bias_fn(x):
  ...   b = hk.get_parameter("b", [], init=hk.initializers.RandomNormal())
  ...   return x + b
  >>> Bias = hk.to_module(bias_fn)
  >>> def net(x, y):
  ...   b = Bias(name="my_bias")
  ...   # Bias x and y by the same amount.
  ...   return b(x) * b(y)

  Args:
    f: The function to convert.

  Returns:
    A module class which runs ``f`` when called.
  """

  class ToModuleWrapper(CallableModule):
    """Module produced by `hk.to_module`."""

    def __init__(self, name=None):
      if name is None:
        name = f.__name__
      elif not isinstance(name, str):
        raise TypeError("Expected a string name as the first argument to the "
                        f"module constructor, got: {name}. Note that "
                        "`hk.to_module` returns a class not an object, so to "
                        "use your module you need to instantiate it first: "
                        "`cls = hk.to_module(fn); mod = cls(); out = mod(x)`.")

      super().__init__(name=name)

    def __call__(self, *a, **k):
      return f(*a, **k)

  if hasattr(f, "__doc__") and f.__doc__:
    ToModuleWrapper.__doc__ = f.__doc__
  functools.update_wrapper(ToModuleWrapper.__call__, f)

  return ToModuleWrapper
