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
"""Reshaping Haiku modules."""

import types
from typing import Optional, Sequence

from haiku._src import module
import jax.numpy as jnp
import numpy as np

# If you are forking replace this block with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.Module = module.Module
del module


def _infer_shape(output_shape, dimensions):
  """Replaces the -1 wildcard in the output shape vector.

  This function infers the correct output shape given the input dimensions.

  Args:
    output_shape: Output shape.
    dimensions: List of input non-batch dimensions.

  Returns:
    Tuple of non-batch output dimensions.
  """
  # Size of input.
  n = np.prod(dimensions)
  # Size of output where defined.
  v = np.array(output_shape)
  m = abs(np.prod(v))
  # Replace wildcard.
  v[v == -1] = n // m
  return tuple(v)


class Reshape(hk.Module):
  """Reshapes input Tensor, preserving the batch dimension.

  For example, given an input tensor with shape ``[B, H, W, C, D]``::

      >>> B, H, W, C, D = range(1, 6)
      >>> x = jnp.ones([B, H, W, C, D])

  The default behavior when ``output_shape`` is ``(-1, D)`` is to flatten
  all dimensions between ``B`` and ``D``::

      >>> mod = hk.Reshape(output_shape=(-1, D))
      >>> assert mod(x).shape == (B, H*W*C, D)

  You can change the number of preserved leading dimensions via
  ``preserve_dims``::

      >>> mod = hk.Reshape(output_shape=(-1, D), preserve_dims=2)
      >>> assert mod(x).shape == (B, H, W*C, D)

      >>> mod = hk.Reshape(output_shape=(-1, D), preserve_dims=3)
      >>> assert mod(x).shape == (B, H, W, C, D)

      >>> mod = hk.Reshape(output_shape=(-1, D), preserve_dims=4)
      >>> assert mod(x).shape == (B, H, W, C, 1, D)

  Alternatively, a negative value of ``preserve_dims`` specifies
  the number of trailing dimensions to replace with ``output_shape``::

      >>> mod = hk.Reshape(output_shape=(-1, D), preserve_dims=-3)
      >>> assert mod(x).shape == (B, H, W*C, D)

  This is useful in the case of applying the same module to batched
  and unbatched outputs::

      >>> mod = hk.Reshape(output_shape=(-1, D), preserve_dims=-3)
      >>> assert mod(x[0]).shape == (H, W*C, D)
  """

  def __init__(
      self,
      output_shape: Sequence[int],
      preserve_dims: int = 1,
      name: Optional[str] = None,
  ):
    """Constructs a :class:`Reshape` module.

    Args:
      output_shape: Shape to reshape the input tensor to while preserving its
        first ``preserve_dims`` dimensions. When the special value ``-1``
        appears in ``output_shape`` the corresponding size is automatically
        inferred. Note that ``-1`` can only appear once in ``output_shape``.
        To flatten all non-batch dimensions use :class:`Flatten`.
      preserve_dims: Number of leading dimensions that will not be reshaped.
        If negative, this is interpreted instead as the number of trailing
        dimensions to replace with the new shape.
      name: Name of the module.

    Raises:
      ValueError: If ``preserve_dims`` is zero.
    """
    super().__init__(name=name)
    if preserve_dims == 0:
      raise ValueError("Argument preserve_dims should be non-zero.")
    if output_shape.count(-1) > 1:
      raise ValueError("-1 can only occur once in `output_shape`.")

    self.output_shape = tuple(output_shape)
    self.preserve_dims = preserve_dims

  def __call__(self, inputs):
    if inputs.ndim <= self.preserve_dims:
      return inputs

    if -1 in self.output_shape:
      reshaped_shape = _infer_shape(self.output_shape,
                                    inputs.shape[self.preserve_dims:])
    else:
      reshaped_shape = self.output_shape
    shape = inputs.shape[:self.preserve_dims] + reshaped_shape
    return jnp.reshape(inputs, shape)


class Flatten(Reshape):
  """Flattens the input, preserving the batch dimension(s).

  By default, :class:`Flatten` combines all dimensions except the first.
  Additional leading dimensions can be preserved by setting ``preserve_dims``.

  >>> x = jnp.ones([3, 2, 4])
  >>> flat = hk.Flatten()
  >>> flat(x).shape
  (3, 8)

  When the input to flatten has fewer than ``preserve_dims`` dimensions it is
  returned unchanged:

  >>> x = jnp.ones([3])
  >>> flat(x).shape
  (3,)

  Alternatively, a negative value of `preserve_dims` specifies the number of
  trailing dimensions flattened:

  >>> x = jnp.ones([3, 2, 4])
  >>> negative_flat = hk.Flatten(preserve_dims=-2)
  >>> negative_flat(x).shape
  (3, 8)

  This allows the same module to be seamlessly applied to a single element or a
  batch of elements with the same element shape:

  >> negative_flat(x[0]).shape
  (8,)
  """

  def __init__(
      self,
      preserve_dims: int = 1,
      name: Optional[str] = None,
  ):
    super().__init__(
        output_shape=(-1,),
        preserve_dims=preserve_dims,
        name=name)
