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
"""Haiku is a neural network library for JAX."""

from haiku import data_structures
from haiku import experimental
from haiku import initializers
from haiku import nets
from haiku import pad
from haiku import testing
from haiku._src.base import get_parameter
from haiku._src.base import get_state
from haiku._src.base import maybe_next_rng_key
from haiku._src.base import next_rng_key
from haiku._src.base import PRNGSequence
from haiku._src.base import set_state
from haiku._src.base import with_rng
from haiku._src.basic import BatchApply
from haiku._src.basic import dropout
from haiku._src.basic import expand_apply
from haiku._src.basic import Linear
from haiku._src.basic import multinomial
from haiku._src.basic import one_hot
from haiku._src.basic import Sequential
from haiku._src.basic import to_module
from haiku._src.batch_norm import BatchNorm
from haiku._src.bias import Bias
from haiku._src.conv import Conv1D
from haiku._src.conv import Conv1DTranspose
from haiku._src.conv import Conv2D
from haiku._src.conv import Conv2DTranspose
from haiku._src.conv import Conv3D
from haiku._src.conv import Conv3DTranspose
from haiku._src.depthwise_conv import DepthwiseConv2D
from haiku._src.embed import Embed
from haiku._src.embed import EmbedLookupStyle
from haiku._src.layer_norm import InstanceNorm
from haiku._src.layer_norm import LayerNorm
from haiku._src.module import Module
from haiku._src.module import transparent
from haiku._src.moving_averages import EMAParamsTree
from haiku._src.moving_averages import ExponentialMovingAverage
from haiku._src.pool import avg_pool
from haiku._src.pool import AvgPool
from haiku._src.pool import max_pool
from haiku._src.pool import MaxPool
from haiku._src.recurrent import Conv1DLSTM
from haiku._src.recurrent import Conv2DLSTM
from haiku._src.recurrent import Conv3DLSTM
from haiku._src.recurrent import deep_rnn_with_skip_connections
from haiku._src.recurrent import DeepRNN
from haiku._src.recurrent import dynamic_unroll
from haiku._src.recurrent import GRU
from haiku._src.recurrent import LSTM
from haiku._src.recurrent import ResetCore
from haiku._src.recurrent import RNNCore
from haiku._src.recurrent import static_unroll
from haiku._src.recurrent import VanillaRNN
from haiku._src.reshape import Flatten
from haiku._src.reshape import Reshape
from haiku._src.spectral_norm import SNParamsTree
from haiku._src.spectral_norm import SpectralNorm
from haiku._src.stateful import cond
from haiku._src.stateful import grad
from haiku._src.stateful import jit
from haiku._src.stateful import remat
from haiku._src.stateful import value_and_grad
from haiku._src.transform import transform
from haiku._src.transform import transform_with_state
from haiku._src.transform import Transformed
from haiku._src.transform import TransformedWithState
from haiku._src.transform import without_apply_rng
from haiku._src.transform import without_state
from haiku._src.typing import Params
from haiku._src.typing import State

__version__ = "0.0.1b0"

__all__ = (
    "AvgPool",
    "BatchApply",
    "BatchNorm",
    "Bias",
    "Conv1D",
    "Conv1DLSTM",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DLSTM",
    "Conv2DTranspose",
    "Conv3D",
    "Conv3DLSTM",
    "Conv3DTranspose",
    "DeepRNN",
    "DepthwiseConv2D",
    "EMAParamsTree",
    "Embed",
    "EmbedLookupStyle",
    "ExponentialMovingAverage",
    "Flatten",
    "GRU",
    "InstanceNorm",
    "LSTM",
    "LayerNorm",
    "Linear",
    "MaxPool",
    "Module",
    "PRNGSequence",
    "Params",
    "RNNCore",
    "ResetCore",
    "Reshape",
    "SNParamsTree",
    "Sequential",
    "SpectralNorm",
    "State",
    "Transformed",
    "TransformedWithState",
    "VanillaRNN",
    "avg_pool",
    "cond",
    "data_structures",
    "deep_rnn_with_skip_connections",
    "dropout",
    "dynamic_unroll",
    "expand_apply",
    "get_parameter",
    "get_state",
    "grad",
    "initializers",
    "jit",
    "max_pool",
    "maybe_next_rng_key",
    "multinomial",
    "nets",
    "next_rng_key",
    "one_hot",
    "pad",
    "remat",
    "set_state",
    "static_unroll",
    "testing",
    "to_module",
    "transform",
    "transform_with_state",
    "transparent",
    "value_and_grad",
    "with_rng",
    "without_apply_rng",
    "without_state",
)

#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Haiku public API.   /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
try:
  del _src  # pylint: disable=undefined-variable
except NameError:
  pass
