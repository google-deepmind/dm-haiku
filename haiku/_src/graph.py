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
"""GraphNetwork Haiku modules.

A (batched) graph is defined by:
 - n_node: a rank-1 array with shape [#graphs],
 - n_edge: a rank-1 array with shape [#graphs],
 - nodes: a rank-2 array with shape [#nodes, C_n],
 - edges: a rank-2 array with shape [#edges, C_e],
 - globals: a rank-1 array with shape [#graphs, C_g],
 - senders: a rank-1 array with shape [#edges],
 - receivers: a rank-1 array with shape [#edges],
where, #nodes = sum(n_node) and #edges = sum(n_edge).

Each edge is directed and connects a sender node to a receiver node.
"""

import collections
from typing import Callable, List, Optional, Text, Sequence

from haiku._src import module
import jax
import jax.numpy as jnp

UpdateNodeFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                        jnp.ndarray]
UpdateEdgeFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                        jnp.ndarray]
UpdateGlobalsFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]


# A tuple defining a graph.
GraphsTuple = collections.namedtuple(
    'GraphsTuple', 'n_node, n_edge, nodes, edges, globals, senders, receivers')


def _repeat(array, repeats, output_length):
  """Returns result of np.repeat with non-scalar repeats on axis 0.

  JAX's repeat function cannot be jit-compiled, because the size of the output
  tensor is data-dependent.
  This function works around this issue by explictly requiring the length of
  the output.

  Args:
    array: array whose elements are repeated.
    repeats: array containing the number of repeats per element.
    output_length: static int containing the length of the repeated array.
  """
  cumsum = jnp.cumsum(repeats)
  range_ = jnp.arange(output_length)
  indices = jnp.sum(range_[:, None] >= cumsum, axis=1)
  return jnp.take(array, indices, axis=0)


class GraphNetwork(module.Module):
  """Graph neural network layer.

  This implementation follows Algorithm 1 in this paper.
  https://arxiv.org/pdf/1806.01261.pdf

  There are two notable differences:
  - This class currently only supports sum as an aggregation function.
    Sum is the most commonly used aggregation function and Jax supports the
    necessary operation out-of-the-box.
  - For the nodes update the class aggregates over the sender edges and receiver
    edges separately. This is a bit more general than the algorithm described in
    the paper.
    The original behaviour can be recovered by using only the receiver edge
    aggregations for the update.
  """

  def __init__(self,
               update_edge_fn: UpdateNodeFn,
               update_node_fn: UpdateEdgeFn,
               update_globals_fn: UpdateGlobalsFn,
               name: Optional[Text] = None):
    """Initializes a GraphNetwork module.

    update_edge_fn must have the following signature:
      (edge features, sender node features, receiver node features, globals) ->
        updated edge features

    update_node_fn must have the following signature:
      (node features, sum of outgoing edge features, sum of incoming edge
       features, globals) -> updated node features

    update_globals_gn must have the following signature:
      (sum of all node features, sum of all edge features, globals) ->
         updated globals

    Args:
      update_edge_fn: function used to update the edges.
      update_node_fn: function used to update the nodes.
      update_globals_fn: function used to update the globals.
      name: The name of the module.
    """
    super(GraphNetwork, self).__init__(name=name)
    # Vectorizes the update functions.
    self._update_edges_fn = jax.vmap(update_edge_fn)
    self._update_nodes_fn = jax.vmap(update_node_fn)
    self._update_globals_fn = jax.vmap(update_globals_fn)

  def __call__(self, graph: GraphsTuple) -> GraphsTuple:
    """Connects GraphNetwork.

    Args:
      graph: a GraphsTuple containing the graph.

    Returns:
      updated GraphsTuple.
    """
    n_node, n_edge, nodes, edges, globals_, senders, receivers = graph

    edges = self._update_edges_fn(
        edges,
        jnp.take(nodes, senders, axis=0),
        jnp.take(nodes, receivers, axis=0),
        _repeat(globals_, n_edge, output_length=edges.shape[0]))

    nodes = self._update_nodes_fn(
        nodes,
        # Hard-coded sum aggregation for the nodes update.
        jax.ops.segment_sum(edges, senders, num_segments=nodes.shape[0]),
        jax.ops.segment_sum(edges, receivers, num_segments=nodes.shape[0]),
        _repeat(globals_, n_node, output_length=nodes.shape[0]))

    n_graph = n_node.shape[0]
    graph_index = jnp.arange(n_graph)
    node_graph_indices = _repeat(graph_index, n_node,
                                 output_length=nodes.shape[0])
    edge_graph_indices = _repeat(graph_index, n_edge,
                                 output_length=edges.shape[0])
    globals_ = self._update_globals_fn(
        # Hard-coded sum aggregation for the globals update.
        jax.ops.segment_sum(nodes, node_graph_indices, num_segments=n_graph),
        jax.ops.segment_sum(edges, edge_graph_indices, num_segments=n_graph),
        globals_)
    return GraphsTuple(
        n_node, n_edge, nodes, edges, globals_, senders, receivers)


def batch(graphs: Sequence[GraphsTuple]) -> GraphsTuple:
  """Returns batched graph given a list of graphs.

  Args:
    graphs: sequence of GraphsTuple which will be batched into a single graph.
  """
  # Calculates offsets for sender and receiver arrays, caused by concatenating
  # the nodes arrays.
  offsets = jnp.cumsum(
      jnp.array([0] + [jnp.sum(g.n_node) for g in graphs[:-1]]))

  return GraphsTuple(
      n_node=jnp.concatenate([g.n_node for g in graphs]),
      n_edge=jnp.concatenate([g.n_edge for g in graphs]),
      nodes=jnp.concatenate([g.nodes for g in graphs]),
      edges=jnp.concatenate([g.edges for g in graphs if g.edges.shape[-1]]),
      globals=jnp.concatenate([g.globals for g in graphs]),
      senders=jnp.concatenate([g.senders + o for g, o in zip(graphs, offsets)]),
      receivers=jnp.concatenate(
          [g.receivers + o for g, o in zip(graphs, offsets)]))


def unbatch(graph: GraphsTuple) -> List[GraphsTuple]:
  """Returns a list of graphs given a batched graph.

  This function does not support jax.jit, because the shape of the output
  is data-dependent!

  Args:
    graph: the batched graph, which will be unbatched into a list of graphs.
  """
  all_n_node = graph.n_node[:, None]
  all_n_edge = graph.n_edge[:, None]
  node_offsets = jnp.cumsum(graph.n_node[:-1])
  all_nodes = jnp.split(graph.nodes, node_offsets)
  edge_offsets = jnp.cumsum(graph.n_edge[:-1])
  all_edges = jnp.split(graph.edges, edge_offsets)
  all_globals = graph.globals[:, None]
  all_senders = jnp.split(graph.senders, edge_offsets)
  all_receivers = jnp.split(graph.receivers, edge_offsets)

  # Corrects offset in the sender and receiver arrays, caused by splitting the
  # nodes array.
  n_graphs = graph.n_node.shape[0]
  for graph_index in jnp.arange(n_graphs)[1:]:
    all_senders[graph_index] -= node_offsets[graph_index - 1]
    all_receivers[graph_index] -= node_offsets[graph_index - 1]

  return [GraphsTuple._make(elements)
          for elements in zip(all_n_node, all_n_edge, all_nodes, all_edges,
                              all_globals, all_senders, all_receivers)]


def pad(graph, n_node, n_edge, n_graph=2):
  """Pads the given graph to given sizes.

  The graph is padded by first adding a dummy graph which contains the padding
  nodes and edges and finally empty graphs without nodes or edges,
  The empty graphs and the dummy graph won't interfer with the graphnet
  calculations.

  The padding graph requires at least one node and one graph.

  This function does not support jax.jit, because the shape of the output
  is data-dependent!

  Args:
    graph: GraphsTuple padded with dummy graph and empty graphs.
    n_node: the number of nodes in the padded graph.
    n_edge: the number of edges in the padded graph.
    n_graph: the number of graphs in the padded graph. Default is 2, which is
      the lowest possible value, because we always have at least one graph in
      the original GraphsTuple and we need one dummy graph for the padding.

  Raises:
    RuntimeError: if the given graph is too large for the given padding.

  Returns:
    The padded graph.
  """
  pad_n_node = int(n_node - jnp.sum(graph.n_node))
  pad_n_edge = int(n_edge - jnp.sum(graph.n_edge))
  pad_n_graph = int(n_graph - graph.n_node.shape[0])
  if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
    raise RuntimeError(
        'Given graph is too large for the given padding.'
        'difference: n_node {}, n_edge {}, n_graph {}'.format(
            pad_n_node, pad_n_edge, pad_n_graph))

  pad_n_empty_graph = pad_n_graph - 1

  padding_graph = GraphsTuple(
      n_node=jnp.concatenate([jnp.array([pad_n_node]),
                              jnp.zeros(pad_n_empty_graph, dtype=jnp.int32)]),
      n_edge=jnp.concatenate([jnp.array([pad_n_edge]),
                              jnp.zeros(pad_n_empty_graph, dtype=jnp.int32)]),
      nodes=jnp.zeros((pad_n_node,) + graph.nodes.shape[1:]),
      edges=jnp.zeros((pad_n_edge,) + graph.edges.shape[1:]),
      globals=jnp.zeros((pad_n_graph,) + graph.globals.shape[1:]),
      senders=jnp.zeros(pad_n_edge, dtype=jnp.int32),
      receivers=jnp.zeros(pad_n_edge, dtype=jnp.int32),
  )
  return batch([graph, padding_graph])


def get_number_of_padding_graphs(padded_graph):
  """Returns number of padding graphs in given padded_graph."""
  n_empty_graph = jnp.argmin(padded_graph.n_node[::-1] == 0)
  return n_empty_graph + 1


def get_number_of_padding_nodes(padded_graph):
  """Returns number of padding nodes in given padded_graph."""
  return padded_graph.n_node[-get_number_of_padding_graphs(padded_graph)]


def get_number_of_padding_edges(padded_graph):
  """Returns number of padding edges in given padded_graph."""
  return padded_graph.n_edge[-get_number_of_padding_graphs(padded_graph)]


def unpad(padded_graph):
  """Unpads the given graph by removing the dummy graph and empty graphs.

  This function assumes that the given graph was padded with the `pad` function.

  This function does not support jax.jit, because the shape of the output
  is data-dependent!

  Args:
    padded_graph: GraphsTuple padded with a dummy graph and empty graphs.

  Returns:
    The unpadded graph.
  """
  n_padding_graph = get_number_of_padding_graphs(padded_graph)
  n_padding_node = get_number_of_padding_nodes(padded_graph)
  n_padding_edge = get_number_of_padding_edges(padded_graph)

  unpadded_graph = GraphsTuple(
      n_node=padded_graph.n_node[:-n_padding_graph],
      n_edge=padded_graph.n_edge[:-n_padding_graph],
      nodes=padded_graph.nodes[:-n_padding_node],
      edges=padded_graph.edges[:-n_padding_edge],
      globals=padded_graph.globals[:-n_padding_graph],
      senders=padded_graph.senders[:-n_padding_edge],
      receivers=padded_graph.receivers[:-n_padding_edge],
  )
  return unpadded_graph


def get_node_padding_mask(padded_graph):
  """Returns a mask for the nodes of a padded graph.

  The mask contains 1 for a real node, and 0 for a padding nodes.

  Args:
    padded_graph: GraphsTuple padded using `pad`.
  """
  n_padding_node = get_number_of_padding_nodes(padded_graph)
  n_valid_node = padded_graph.nodes.shape[0] - n_padding_node
  return jnp.arange(padded_graph.nodes.shape[0], dtype=jnp.int32) < n_valid_node


def get_edge_padding_mask(padded_graph):
  """Returns a mask for the edges of a padded graph.

  The mask contains 1 for a real edge, and 0 for a padding edge.

  Args:
    padded_graph: GraphsTuple padded using `pad`.
  """
  n_padding_edge = get_number_of_padding_edges(padded_graph)
  n_valid_edge = padded_graph.edges.shape[0] - n_padding_edge
  return jnp.arange(padded_graph.edges.shape[0], dtype=jnp.int32) < n_valid_edge


def get_graph_padding_mask(padded_graph):
  """Returns a mask for the graphs of a padded graph.

  The mask contains 1 for a real graph, and 0 for a padding graph.

  Args:
    padded_graph: GraphsTuple padded using `pad`.
  """
  n_padding_graph = get_number_of_padding_graphs(padded_graph)
  n_valid_graph = padded_graph.globals.shape[0] - n_padding_graph
  return (jnp.arange(padded_graph.globals.shape[0], dtype=jnp.int32)
          < n_valid_graph)
