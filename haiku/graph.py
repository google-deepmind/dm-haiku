# Lint as: python3
# Copyright 2019 The Haiku Authors. All Rights Reserved.
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
"""Graph Neural Network."""

from haiku._src.graph import batch
from haiku._src.graph import get_edge_padding_mask
from haiku._src.graph import get_graph_padding_mask
from haiku._src.graph import get_node_padding_mask
from haiku._src.graph import get_number_of_padding_edges
from haiku._src.graph import get_number_of_padding_graphs
from haiku._src.graph import get_number_of_padding_nodes
from haiku._src.graph import GraphNetwork
from haiku._src.graph import GraphsTuple
from haiku._src.graph import pad
from haiku._src.graph import unbatch
from haiku._src.graph import unpad
from haiku._src.graph import UpdateEdgeFn
from haiku._src.graph import UpdateGlobalsFn
from haiku._src.graph import UpdateNodeFn

__all__ = (
    "UpdateNodeFn",
    "UpdateEdgeFn",
    "UpdateGlobalsFn",
    "GraphsTuple",
    "GraphNetwork",
    "batch",
    "unbatch",
    "pad",
    "get_number_of_padding_graphs",
    "get_number_of_padding_nodes",
    "get_number_of_padding_edges",
    "unpad",
    "get_node_padding_mask",
    "get_edge_padding_mask",
    "get_graph_padding_mask",
)
