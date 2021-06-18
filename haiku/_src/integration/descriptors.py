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

from typing import Any, Callable, NamedTuple, Type, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

ModuleFn = Callable[[], Callable[[jnp.ndarray], jnp.ndarray]]


class Wrapped(hk.Module):

  def __init__(self, wrapped):
    super().__init__()
    self.wrapped = wrapped


class Training(Wrapped):

  def __call__(self, x: jnp.ndarray):
    return self.wrapped(x, is_training=True)


class MultiInput(Wrapped):

  def __init__(self, wrapped, num_inputs):
    super().__init__(wrapped)
    self.num_inputs = num_inputs

  def __call__(self, x: jnp.ndarray):
    inputs = [x for _ in range(self.num_inputs)]
    return self.wrapped(*inputs)


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
  shape: Sequence[int]
  dtype: Any = jnp.float32


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
        create=lambda: hk.Sequential([]),
        shape=(BATCH_SIZE, 2, 2)),
    ModuleDescriptor(
        name="nets.MLP",
        create=lambda: hk.nets.MLP([3, 4, 5]),
        shape=(BATCH_SIZE, 3)),
    ModuleDescriptor(
        name="ConvND",
        create=lambda: hk.ConvND(1, 3, 3),
        shape=(BATCH_SIZE, 2, 2)),
    ModuleDescriptor(
        name="ConvNDTranspose",
        create=lambda: hk.ConvNDTranspose(1, 3, 3),
        shape=(BATCH_SIZE, 2, 2)),
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
        name="GroupNorm",
        create=lambda: hk.GroupNorm(5),
        shape=(BATCH_SIZE, 4, 4, 10)),
    ModuleDescriptor(
        name="LayerNorm",
        create=lambda: hk.LayerNorm(1, True, True),
        shape=(BATCH_SIZE, 3, 2)),
    ModuleDescriptor(
        name="MultiHeadAttention",
        create=lambda: MultiInput(  # pylint: disable=g-long-lambda
            hk.MultiHeadAttention(num_heads=8, key_size=64, w_init_scale=1.0),
            num_inputs=3),
        shape=(BATCH_SIZE, 3, 2)),
    ModuleDescriptor(
        name="RMSNorm",
        create=lambda: hk.RMSNorm(1),
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
    ModuleDescriptor(
        name="nets.VectorQuantizer",
        create=lambda: Training(hk.nets.VectorQuantizer(64, 512, 0.25)),
        shape=(BATCH_SIZE, 64)),
    ModuleDescriptor(
        name="nets.VectorQuantizerEMA",
        create=lambda: Training(hk.nets.VectorQuantizerEMA(64, 512, 0.25, 0.9)),
        shape=(BATCH_SIZE, 64)),

    # TODO(tomhennigan) Make these modules support unbatched input.
    ModuleDescriptor(
        name="DepthwiseConv2D",
        create=lambda: hk.DepthwiseConv2D(1, 3),
        shape=(BATCH_SIZE, 2, 2, 2)),
    ModuleDescriptor(
        name="SeparableDepthwiseConv2D",
        create=lambda: hk.SeparableDepthwiseConv2D(1, 3),
        shape=(BATCH_SIZE, 2, 2, 2)),
)


class DummyCore(hk.RNNCore):

  def initial_state(self, batch_size):
    if batch_size is not None:
      return jnp.ones([batch_size, 128, 1])
    else:
      return jnp.ones([128, 1])

  def __call__(self, inputs, state):
    return inputs, state


class ResetCoreAdapter(Wrapped, hk.RNNCore):

  def initial_state(self, batch_size):
    return self.wrapped.initial_state(batch_size)

  def __call__(self, inputs, state):
    batch_size = inputs.shape[0]
    resets = np.broadcast_to(True, (batch_size,))
    return self.wrapped((inputs, resets), state)


# RNN cores. For shape, use the shape of a single example.
RNN_CORES = (
    ModuleDescriptor(
        name="ResetCore",
        create=lambda: ResetCoreAdapter(hk.ResetCore(DummyCore())),
        shape=(BATCH_SIZE, 128)),
    ModuleDescriptor(
        name="GRU",
        create=lambda: hk.GRU(1),
        shape=(BATCH_SIZE, 128)),
    ModuleDescriptor(
        name="IdentityCore",
        create=lambda: hk.IdentityCore(),
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


def module_type(module_fn: ModuleFn) -> Type[hk.Module]:
  f = hk.transform_with_state(lambda: type(unwrap(module_fn())))
  return f.apply(*f.init(jax.random.PRNGKey(42)), None)[0]


def with_name(descriptors: Sequence[ModuleDescriptor]):
  return [[n, n, c, s, d] for  n, c, s, d in descriptors]


def to_file_name(descriptor: ModuleDescriptor):
  n = descriptor.name
  return n.replace(" ", "-").replace("(", "-").replace(")", "").replace(",", "")


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
    hk.nets.ResNet18,
    hk.nets.ResNet34,
    hk.nets.ResNet50,
    hk.nets.ResNet101,
    hk.nets.ResNet152,
    hk.nets.ResNet200,
}
