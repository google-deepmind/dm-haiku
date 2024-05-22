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
# pylint: disable=g-importing-member
"""Experimental features developed by the Haiku core team.

Features may be removed or modified at any time.
"""

from haiku._src.base import current_name
from haiku._src.base import custom_creator
from haiku._src.base import custom_getter
from haiku._src.base import DO_NOT_STORE
from haiku._src.base import get_current_state
from haiku._src.base import get_initial_state
from haiku._src.base import get_params
from haiku._src.base import GetterContext
from haiku._src.base import maybe_get_rng_sequence_state
from haiku._src.base import replace_rng_sequence_state
from haiku._src.config import check_jax_usage
from haiku._src.config import module_auto_repr
from haiku._src.config import rng_reserve_size
from haiku._src.dot import abstract_to_dot
from haiku._src.dot import to_dot
from haiku._src.eval_shape import fast_eval_shape
from haiku._src.layer_stack import layer_stack
from haiku._src.layer_stack import LayerStackTransparencyMapping
from haiku._src.lift import lift
from haiku._src.lift import lift_with_state
from haiku._src.lift import LiftWithStateUpdater
from haiku._src.lift import transparent_lift
from haiku._src.lift import transparent_lift_with_state
from haiku._src.module import force_name
from haiku._src.module import intercept_methods
from haiku._src.module import MethodContext
from haiku._src.module import name_like
from haiku._src.module import name_scope
from haiku._src.random import optimize_rng_use
from haiku._src.summarise import ArraySpec
from haiku._src.summarise import eval_summary
from haiku._src.summarise import MethodInvocation
from haiku._src.summarise import ModuleDetails
from haiku._src.summarise import tabulate
from haiku.experimental import jaxpr_info

# pylint: disable=g-import-not-at-top,g-bad-import-order
try:
  from haiku.experimental import flax
except ImportError:
  # We inject a shim module to provide helpful errors when flax isn't installed.
  from haiku._src import no_flax
  no_flax.inject_shim_module()
  del no_flax

from haiku.experimental import flax
# pylint: enable=g-import-not-at-top,g-bad-import-order

# TODO(tomhennigan): Remove deprecated alias.
ParamContext = GetterContext


__all__ = (
    "abstract_to_dot",
    "ArraySpec",
    "eval_summary",
    "check_jax_usage",
    "current_name",
    "custom_creator",
    "custom_getter",
    "DO_NOT_STORE",
    "force_name",
    "intercept_methods",
    "flax",
    "get_current_state",
    "get_initial_state",
    "get_params",
    "jaxpr_info",
    "layer_stack",
    "LayerStackTransparencyMapping",
    "lift",
    "lift_with_state",
    "LiftWithStateUpdater",
    "maybe_get_rng_sequence_state",
    "module_auto_repr",
    "MethodContext",
    "MethodInvocation",
    "ModuleDetails",
    "name_like",
    "name_scope",
    "optimize_rng_use",
    "GetterContext",
    "ParamContext",
    "replace_rng_sequence_state",
    "rng_reserve_size",
    "tabulate",
    "to_dot",
    "transparent_lift",
    "transparent_lift_with_state",
)
