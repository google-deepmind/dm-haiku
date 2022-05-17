# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tools for understanding JAX + Haiku programs."""

from haiku._src.jaxpr_info import as_html
from haiku._src.jaxpr_info import as_html_page
from haiku._src.jaxpr_info import css
from haiku._src.jaxpr_info import Expression
from haiku._src.jaxpr_info import format_module
from haiku._src.jaxpr_info import js
from haiku._src.jaxpr_info import make_model_info
from haiku._src.jaxpr_info import Module

__all__ = (
    "as_html",
    "as_html_page",
    "css",
    "Expression",
    "format_module",
    "js",
    "make_model_info",
    "Module",
)
