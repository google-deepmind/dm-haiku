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
from haiku import mixed_precision
from haiku import nets
from haiku import pad
from haiku import testing
from haiku._src.attention import MultiHeadAttention
from haiku._src.base import custom_creator
from haiku._src.base import custom_getter
from haiku._src.base import get_parameter
from haiku._src.base import get_state
from haiku._src.base import GetterContext
from haiku._src.base import maybe_next_rng_key
from haiku._src.base import next_rng_key
from haiku._src.base import next_rng_keys
from haiku._src.base import PRNGSequence
from haiku._src.base import reserve_rng_keys
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
from haiku._src.conv import ConvND
from haiku._src.conv import ConvNDTranspose
from haiku._src.depthwise_conv import DepthwiseConv2D
from haiku._src.depthwise_conv import SeparableDepthwiseConv2D
from haiku._src.embed import Embed
from haiku._src.embed import EmbedLookupStyle
from haiku._src.group_norm import GroupNorm
from haiku._src.layer_norm import InstanceNorm
from haiku._src.layer_norm import LayerNorm
from haiku._src.lift import lift
from haiku._src.module import intercept_methods
from haiku._src.module import MethodContext
from haiku._src.module import Module
from haiku._src.module import transparent
from haiku._src.moving_averages import EMAParamsTree
from haiku._src.moving_averages import ExponentialMovingAverage
from haiku._src.multi_transform import multi_transform
from haiku._src.multi_transform import multi_transform_with_state
from haiku._src.multi_transform import MultiTransformed
from haiku._src.multi_transform import MultiTransformedWithState
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
from haiku._src.recurrent import IdentityCore
from haiku._src.recurrent import LSTM
from haiku._src.recurrent import LSTMState
from haiku._src.recurrent import ResetCore
from haiku._src.recurrent import RNNCore
from haiku._src.recurrent import static_unroll
from haiku._src.recurrent import VanillaRNN
from haiku._src.reshape import Flatten
from haiku._src.reshape import Reshape
from haiku._src.rms_norm import RMSNorm
from haiku._src.spectral_norm import SNParamsTree
from haiku._src.spectral_norm import SpectralNorm
from haiku._src.stateful import cond
from haiku._src.stateful import eval_shape
from haiku._src.stateful import fori_loop
from haiku._src.stateful import grad
from haiku._src.stateful import jit
from haiku._src.stateful import remat
from haiku._src.stateful import scan
from haiku._src.stateful import switch
from haiku._src.stateful import value_and_grad
from haiku._src.stateful import vmap
from haiku._src.stateful import while_loop
from haiku._src.transform import running_init
from haiku._src.transform import transform
from haiku._src.transform import transform_with_state
from haiku._src.transform import Transformed
from haiku._src.transform import TransformedWithState
from haiku._src.transform import with_empty_state
from haiku._src.transform import without_apply_rng
from haiku._src.transform import without_state
from haiku._src.typing import Params
from haiku._src.typing import State

__version__ = "0.0.5"

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
    "ConvND",
    "ConvNDTranspose",
    "DeepRNN",
    "DepthwiseConv2D",
    "EMAParamsTree",
    "Embed",
    "EmbedLookupStyle",
    "ExponentialMovingAverage",
    "Flatten",
    "GetterContext",
    "GRU",
    "GroupNorm",
    "IdentityCore",
    "InstanceNorm",
    "LSTM",
    "LSTMState",
    "LayerNorm",
    "Linear",
    "MaxPool",
    "MethodContext",
    "Module",
    "MultiHeadAttention",
    "MultiTransformed",
    "MultiTransformedWithState",
    "PRNGSequence",
    "Params",
    "RNNCore",
    "ResetCore",
    "Reshape",
    "RMSNorm",
    "SNParamsTree",
    "Sequential",
    "SpectralNorm",
    "State",
    "Transformed",
    "TransformedWithState",
    "VanillaRNN",
    "avg_pool",
    "cond",
    "eval_shape",
    "custom_creator",
    "custom_getter",
    "data_structures",
    "deep_rnn_with_skip_connections",
    "dropout",
    "dynamic_unroll",
    "expand_apply",
    "fori_loop",
    "get_parameter",
    "get_state",
    "grad",
    "initializers",
    "intercept_methods",
    "jit",
    "lift",
    "max_pool",
    "maybe_next_rng_key",
    "mixed_precision",
    "multi_transform",
    "multi_transform_with_state",
    "multinomial",
    "nets",
    "next_rng_key",
    "next_rng_keys",
    "one_hot",
    "pad",
    "remat",
    "reserve_rng_keys",
    "running_init",
    "scan",
    "set_state",
    "static_unroll",
    "switch",
    "testing",
    "to_module",
    "transform",
    "transform_with_state",
    "transparent",
    "value_and_grad",
    "vmap",
    "while_loop",
    "with_empty_state",
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
