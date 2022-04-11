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
"""Experimental features developed by the Haiku core team.

Features may be removed or modified at any time.
"""
import types

from haiku._src.base import custom_creator
from haiku._src.base import custom_getter
from haiku._src.base import GetterContext
from haiku._src.config import check_jax_usage
from haiku._src.config import module_auto_repr
from haiku._src.dot import abstract_to_dot
from haiku._src.dot import to_dot
import haiku._src.jaxpr_info as private_jaxpr_info
from haiku._src.layer_stack import layer_stack
from haiku._src.lift import lift
from haiku._src.lift import lift_with_state
from haiku._src.lift import LiftWithStateUpdater
from haiku._src.module import intercept_methods
from haiku._src.module import MethodContext
from haiku._src.module import name_like
from haiku._src.module import name_scope
from haiku._src.module import profiler_name_scopes
from haiku._src.random import optimize_rng_use
from haiku._src.stateful import named_call
from haiku._src.summarise import ArraySpec
from haiku._src.summarise import eval_summary
from haiku._src.summarise import MethodInvocation
from haiku._src.summarise import ModuleDetails
from haiku._src.summarise import tabulate

# TODO(tomhennigan): Remove deprecated alias.
ParamContext = GetterContext

jaxpr_info = types.ModuleType(
    "haiku.experimental.jaxpr_info", doc=private_jaxpr_info.__doc__)

# Ensure private definitions in the module are not accessible.
_JAXPR_SYMBOLS = [
    "make_model_info", "Expression", "Module", "format_module", "as_html_page",
    "as_html", "css", "js"
]
for name in _JAXPR_SYMBOLS:
  setattr(jaxpr_info, name, getattr(private_jaxpr_info, name))
jaxpr_info.__all__ = _JAXPR_SYMBOLS
del _JAXPR_SYMBOLS
del private_jaxpr_info

__all__ = (
    "abstract_to_dot",
    "ArraySpec",
    "eval_summary",
    "check_jax_usage",
    "custom_creator",
    "custom_getter",
    "intercept_methods",
    "jaxpr_info",
    "layer_stack",
    "lift",
    "lift_with_state",
    "LiftWithStateUpdater",
    "module_auto_repr",
    "MethodContext",
    "MethodInvocation",
    "ModuleDetails",
    "name_like",
    "name_scope",
    "named_call",
    "optimize_rng_use",
    "GetterContext",
    "ParamContext",
    "profiler_name_scopes",
    "tabulate",
    "to_dot",
)
