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
"""Bias module."""

from typing import Optional, Sequence, Text, Union

from haiku._src import base
from haiku._src import module
from haiku._src import utils
import jax.numpy as jnp

FloatLike = Union[float, jnp.ndarray]
ShapeLike = Sequence[Optional[int]]


class Bias(module.Module):
  """Bias module.

  Example Usage:

      >>> N, H, W, C = 1, 2, 3, 4
      >>> x = jnp.ones([N, H, W, C])

      >>> scalar_bias = hk.Bias(bias_dims=[])
      >>> scalar_bias_output = scalar_bias(x)
      >>> assert scalar_bias.bias_shape == ()

  Create a bias over all non-minibatch dimensions:

      >>> all_bias = hk.Bias()
      >>> all_bias_output = all_bias(x)
      >>> assert all_bias.bias_shape == (H, W, C)

  Create a bias over the last non-minibatch dimension:

      >>> last_bias = hk.Bias(bias_dims=[-1])
      >>> last_bias_output = last_bias(x)
      >>> assert last_bias.bias_shape == (C,)

  Create a bias over the first non-minibatch dimension:

      >>> first_bias = hk.Bias(bias_dims=[1])
      >>> first_bias_output = first_bias(x)
      >>> assert first_bias.bias_shape == (H, 1, 1)

  Subtract and later add the same learned bias:

      >>> bias = hk.Bias()
      >>> h1 = bias(x, multiplier=-1)
      >>> h2 = bias(x)
      >>> h3 = bias(x, multiplier=-1)
      >>> reconstructed_x = bias(h3)
      >>> assert jnp.sum(x == reconstructed_x)
  """

  def __init__(self,
               output_size: Optional[Union[int, ShapeLike]] = None,
               bias_dims: Optional[Sequence[int]] = None,
               b_init: Optional[base.Initializer] = None,
               name: Optional[Text] = None):
    """Constructs a `Bias` module that supports broadcasting.

    Args:
      output_size: Output size (output shape without batch dimension). If
        `output_size` is left as `None`, the size will be directly inferred by
        the input.
      bias_dims: Sequence of which dimensions to retain from the input shape
        when constructing the bias. The remaining dimensions will be broadcast
        over (given size of 1), and leading dimensions will be removed
        completely. See class doc for examples.
      b_init: Optional initializer for the bias. Default to zeros.
      name: Name of the module.
    """
    super(Bias, self).__init__(name=name)
    self.output_size = output_size
    self.bias_dims = bias_dims
    self.b_init = b_init or jnp.zeros

  def __call__(self, inputs: jnp.ndarray, multiplier: FloatLike = None):
    """Adds bias to `inputs` and optionally multiplies by `multiplier`.

    Args:
      inputs: A Tensor of size `[batch_size, input_size1, ...]`.
      multiplier: A scalar or Tensor which the bias term is multiplied by before
        adding it to `inputs`. Anything which works in the expression `bias *
        multiplier` is acceptable here. This may be useful if you want to add a
        bias in one place and subtract the same bias in another place via
        `multiplier=-1`.

    Returns:
      A Tensor of size `[batch_size, input_size1, ...]`.
    """
    utils.assert_minimum_rank(inputs, 2)

    input_shape = inputs.shape
    self.bias_shape = calculate_bias_shape(input_shape, self.bias_dims)

    input_size = input_shape[1:]
    if self.output_size is not None and self.output_size != input_size:
      raise ValueError("Input shape must be {} not {}".format(
          (-1,) + self.output_size, input_shape))

    self.input_size = input_size
    b = base.get_parameter("b", self.bias_shape, inputs.dtype, init=self.b_init)
    b = jnp.broadcast_to(b, inputs.shape)

    if multiplier is not None:
      return inputs + (b * multiplier)
    else:
      return inputs + b


def calculate_bias_shape(input_shape: ShapeLike,
                         bias_dims: Sequence[int]):
  """Calculate `bias_shape` based on the `input_shape` and `bias_dims`.

  Args:
    input_shape: Shape of the input being passed into the module. The leading
      dimension is the mini-batch size.
    bias_dims: The dimensions that bias should be applied over. The remaining
      dimensions will be broadcast over.

  Returns:
    bias_shape: Tuple corresponding to the shape of bias Variable to create.

  Raises:
    ValueError: If the user attempts to add bias over the mini-batch dimension,
        e.g. `bias_dims=[0]`.
  """
  input_rank = len(input_shape)
  if bias_dims is None:
    # If None, default is to use all dimensions.
    return input_shape[1:]

  elif not bias_dims:
    # If empty list, use a scalar bias.
    return ()

  else:
    # Otherwise, calculate bias_shape from bias_dims.
    bias_shape = [1] * input_rank
    # Populate bias dimensions.
    for dim in bias_dims:
      if dim < 0:
        dim %= input_rank

      if dim == 0:
        raise ValueError("Cannot apply bias across the minibatch dimension.")
      elif dim >= input_rank:
        raise ValueError(
            "Dimension %d (bias_dims=%r) out of range for input of rank %r." %
            (dim, tuple(bias_dims), input_rank))

      bias_shape[dim] = input_shape[dim]
    # Strip leading unit dimensions.
    start = input_rank
    for dim in range(1, input_rank):
      if bias_shape[dim] != 1:
        start = dim
        break
    return tuple(bias_shape[start:])  # Do not apply across minibatch dimension.
