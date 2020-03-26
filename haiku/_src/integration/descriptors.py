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
"""Module descriptors programatically describe how to use modules."""

from typing import Any, Callable, NamedTuple

import haiku as hk
from haiku._src.typing import Shape, DType  # pylint: disable=g-multiple-import
import jax
import jax.numpy as jnp
import numpy as np

ModuleFn = Callable[[], Callable[[jnp.ndarray], jnp.ndarray]]


class Wrapped(hk.Module):

  def __init__(self, wrapped):
    super(Wrapped, self).__init__()
    self.wrapped = wrapped


class Training(Wrapped):

  def __call__(self, x: jnp.ndarray):
    return self.wrapped(x, is_training=True)


class Recurrent(Wrapped):
  """Unrolls a recurrent module."""

  def __init__(self, module: hk.RNNCore, unroller=None):
    super().__init__(module)
    self.unroller = unroller

  def __call__(self, x: jnp.ndarray):
    initial_state = jax.tree_map(
        lambda v: v.astype(x.dtype),
        self.wrapped.initial_state(batch_size=x.shape[0]))
    x = jnp.expand_dims(x, axis=0)
    return self.unroller(self.wrapped, x, initial_state)


def unwrap(module):
  while isinstance(module, Wrapped):
    module = module.wrapped
  return module


class ModuleDescriptor(NamedTuple):
  name: Any
  create: ModuleFn
  shape: Shape
  dtype: DType = jnp.float32


BATCH_SIZE = 8

# pylint: disable=unnecessary-lambda
# Modules that have equivalent behaviour with or without a batch dimension.
OPTIONAL_BATCH_MODULES = (
    ModuleDescriptor(
        name="Embed",
        create=lambda: hk.Embed(vocab_size=6, embed_dim=12),
        shape=(BATCH_SIZE,),
        dtype=jnp.int32),
    ModuleDescriptor(
        name="Linear",
        create=lambda: hk.Linear(10),
        shape=(BATCH_SIZE, 1)),
    ModuleDescriptor(
        name="Sequential",
        create=lambda: hk.Sequential([lambda x: x]),
        shape=(BATCH_SIZE, 2, 2)),
    ModuleDescriptor(
        name="nets.MLP",
        create=lambda: hk.nets.MLP([3, 4, 5]),
        shape=(BATCH_SIZE, 3)),
)

# Modules that require input to have a batch dimension.
BATCH_MODULES = (
    ModuleDescriptor(
        name="BatchNorm",
        create=lambda: Training(hk.BatchNorm(True, True, 0.9)),
        shape=(BATCH_SIZE, 2, 2, 3)),
    ModuleDescriptor(
        name="Bias",
        create=lambda: hk.Bias(),
        shape=(BATCH_SIZE, 3, 3, 3)),
    ModuleDescriptor(
        name="Flatten",
        create=lambda: hk.Flatten(),
        shape=(BATCH_SIZE, 3, 3, 3)),
    ModuleDescriptor(
        name="InstanceNorm",
        create=lambda: hk.InstanceNorm(True, True),
        shape=(BATCH_SIZE, 3, 2)),
    ModuleDescriptor(
        name="LayerNorm",
        create=lambda: hk.LayerNorm(1, True, True),
        shape=(BATCH_SIZE, 3, 2)),
    ModuleDescriptor(
        name="SpectralNorm",
        create=lambda: hk.SpectralNorm(),
        shape=(BATCH_SIZE, 3, 2)),
    ModuleDescriptor(
        name="nets.ResNet",
        create=lambda: Training(hk.nets.ResNet((3, 4, 6, 3), 1000)),
        shape=(BATCH_SIZE, 3, 3, 2)),
    # pylint: disable=g-long-lambda
    ModuleDescriptor(
        name="nets.MobileNetV1",
        create=lambda: Training(hk.nets.MobileNetV1(num_classes=1000,
                                                    strides=(1, 1, 1),
                                                    channels=(16, 32, 64))),
        shape=(BATCH_SIZE, 64, 64, 2)),
    # pylint: enable=g-long-lambda

    # TODO(tomhennigan) Make these modules support unbatched input.
    ModuleDescriptor(
        name="Conv1D",
        create=lambda: hk.Conv1D(3, 3),
        shape=(BATCH_SIZE, 2, 2)),
    ModuleDescriptor(
        name="Conv1DTranspose",
        create=lambda: hk.Conv1DTranspose(3, 3),
        shape=(BATCH_SIZE, 2, 2)),
    ModuleDescriptor(
        name="Conv2D",
        create=lambda: hk.Conv2D(3, 3),
        shape=(BATCH_SIZE, 2, 2, 2)),
    ModuleDescriptor(
        name="Conv2DTranspose",
        create=lambda: hk.Conv2DTranspose(3, 3),
        shape=(BATCH_SIZE, 2, 2, 2)),
    ModuleDescriptor(
        name="Conv3D",
        create=lambda: hk.Conv3D(3, 3),
        shape=(BATCH_SIZE, 2, 2, 2, 2)),
    ModuleDescriptor(
        name="Conv3DTranspose",
        create=lambda: hk.Conv3DTranspose(3, 3),
        shape=(BATCH_SIZE, 2, 2, 2, 2)),
    ModuleDescriptor(
        name="DepthwiseConv2D",
        create=lambda: hk.DepthwiseConv2D(1, 3),
        shape=(BATCH_SIZE, 2, 2, 2)),
)


class IdentityCore(hk.RNNCore):

  def initial_state(self, batch_size):
    return jnp.ones([batch_size, 128, 1])

  def __call__(self, inputs, state):
    return inputs, state


class ResetCoreAdapter(Wrapped, hk.RNNCore):

  def initial_state(self, batch_size):
    return self.wrapped.initial_state(batch_size)

  def __call__(self, inputs, state):
    t, b = inputs.shape
    resets = np.broadcast_to(True, (t, b))
    return self.wrapped((inputs, resets), state)


# RNN cores. For shape, use the shape of a single example.
RNN_CORES = (
    ModuleDescriptor(
        name="ResetCore",
        create=lambda: ResetCoreAdapter(hk.ResetCore(IdentityCore())),
        shape=(BATCH_SIZE, 128)),
    ModuleDescriptor(
        name="GRU",
        create=lambda: hk.GRU(1),
        shape=(BATCH_SIZE, 128)),
    ModuleDescriptor(
        name="LSTM",
        create=lambda: hk.LSTM(1),
        shape=(BATCH_SIZE, 128)),
    ModuleDescriptor(
        name="Conv1DLSTM",
        create=lambda: hk.Conv1DLSTM([2], 3, 3),
        shape=(BATCH_SIZE, 2, 2)),
    ModuleDescriptor(
        name="Conv2DLSTM",
        create=lambda: hk.Conv2DLSTM([2, 2], 3, 3),
        shape=(BATCH_SIZE, 2, 2, 2)),
    ModuleDescriptor(
        name="Conv3DLSTM",
        create=lambda: hk.Conv3DLSTM([2, 2, 2], 3, 3),
        shape=(BATCH_SIZE, 2, 2, 2, 2)),
    ModuleDescriptor(
        name="VanillaRNN",
        create=lambda: hk.VanillaRNN(8),
        shape=(BATCH_SIZE, 128)),
)


def recurrent_factory(
    create_core: Callable[[], hk.RNNCore],
    unroller,
) -> Callable[[], Recurrent]:
  return lambda: Recurrent(create_core(), unroller)


def unroll_descriptors(descriptors, unroller):
  """Returns `Recurrent` wrapped descriptors with the given unroller applied."""
  out = []
  for name, create, shape, dtype in descriptors:
    name = "Recurrent({}, {})".format(name, unroller.__name__)
    out.append(
        ModuleDescriptor(name=name,
                         create=recurrent_factory(create, unroller),
                         shape=shape,
                         dtype=dtype))
  return tuple(out)


# Modules that require time then batch input.
RECURRENT_MODULES = (
    unroll_descriptors(RNN_CORES, hk.dynamic_unroll) +
    unroll_descriptors(RNN_CORES, hk.static_unroll))

ALL_MODULES = OPTIONAL_BATCH_MODULES + BATCH_MODULES + RECURRENT_MODULES

IGNORED_MODULES = {
    # Stateless or abstract.
    hk.BatchApply,
    hk.Module,
    hk.Reshape,
    hk.AvgPool,
    hk.MaxPool,
    hk.experimental.lift,

    # Non-standard.
    hk.EMAParamsTree,
    hk.SNParamsTree,

    # Metrics.
    hk.ExponentialMovingAverage,

    # Recurrent.
    hk.DeepRNN,
    hk.RNNCore,

    # Tested transitively.
    hk.nets.ResNet50,
    hk.nets.ResNet101,
    hk.nets.ResNet152,
    hk.nets.ResNet200,
}
