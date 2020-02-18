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
"""A minimal interface mlp module."""

from typing import Callable, Iterable, Optional, Text

from haiku._src import base
from haiku._src import basic
from haiku._src import module
import jax
import jax.numpy as jnp


class MLP(module.Module):
  """A multi-layer perceptron module."""

  def __init__(self,
               output_sizes: Iterable[int],
               w_init: Optional[base.Initializer] = None,
               b_init: Optional[base.Initializer] = None,
               with_bias: bool = True,
               activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
               activate_final: bool = False,
               name: Optional[Text] = None):
    """Constructs an MLP.

    Args:
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for Linear weights.
      b_init: Initializer for Linear bias. Must be `None` if `with_bias` is
        `False`.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between linear layers. Defaults
        to ReLU.
      activate_final: Whether or not to activate the final layer of the MLP.
      name: Optional name for this module.

    Raises:
      ValueError: If with_bias is False and b_init is not None.
    """
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")

    super(MLP, self).__init__(name=name)
    self._with_bias = with_bias
    self._w_init = w_init
    self._b_init = b_init
    self._activation = activation
    self._activate_final = activate_final
    self._layers = []
    for index, output_size in enumerate(output_sizes):
      self._layers.append(
          basic.Linear(
              output_size=output_size,
              w_init=w_init,
              b_init=b_init,
              with_bias=with_bias,
              name="linear_%d" % index))

  @property
  def layers(self):
    return tuple(self._layers)

  def __call__(
      self,
      inputs: jnp.ndarray,
      dropout_rate: Optional[float] = None,
      rng=None,
  ) -> jnp.ndarray:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of shape `[batch_size, input_size]`.
      dropout_rate: Optional dropout rate.
      rng: Optional RNG key. Require when using dropout.

    Returns:
      output: The output of the model of size `[batch_size, output_size]`.
    """
    if dropout_rate is not None and rng is None:
      raise ValueError("When using dropout an rng key must be passed.")
    elif dropout_rate is None and rng is not None:
      raise ValueError("RNG should only be passed when using dropout.")

    rng = base.PRNGSequence(rng) if rng is not None else None
    num_layers = len(self._layers)

    for i, layer in enumerate(self._layers):
      inputs = layer(inputs)
      if i < (num_layers - 1) or self._activate_final:
        # Only perform dropout if we are activating the output.
        if dropout_rate is not None:
          inputs = basic.dropout(next(rng), dropout_rate, inputs)
        inputs = self._activation(inputs)

    return inputs

  def reverse(self,
              activate_final: Optional[bool] = None,
              name: Optional[Text] = None) -> "MLP":
    """Returns a new MLP which is the layer-wise reverse of this MLP.

    NOTE: Since computing the reverse of an MLP requires knowing the input size
    of each linear layer this method will fail if the module has not been called
    at least once.

    The contract of reverse is that the reversed module will accept the output
    of the parent module as input and produce an output which is the input size
    of the parent.

    >>> mlp = hk.nets.MLP([1, 2, 3])
    >>> y = mlp(jnp.ones([1, 2]))
    >>> rev = mlp.reverse()
    >>> rev(y)
    DeviceArray(...)

    Args:
      activate_final: Whether the final layer of the MLP should be activated.
      name: Optional name for the new module. The default name will be the name
        of the current module prefixed with ``"reversed_"``.

    Returns:
      An MLP instance which is the reverse of the current instance. Note these
      instances do not share weights and, apart from being symmetric to each
      other, are not coupled in any way.
    """

    if activate_final is None:
      activate_final = self._activate_final
    if name is None:
      name = self.name + "_reversed"

    return MLP(
        output_sizes=(layer.input_size for layer in reversed(self._layers)),
        w_init=self._w_init,
        b_init=self._b_init,
        with_bias=self._with_bias,
        activation=self._activation,
        activate_final=activate_final,
        name=name)
