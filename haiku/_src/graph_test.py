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
"""Tests for haiku._src.graph."""

from absl.testing import absltest
from haiku._src import base
from haiku._src import graph
import jax
from jax import random
from jax import test_util
import jax.numpy as jnp
import numpy as np


def _get_random_graph(max_n_graph=10):
  n_graph = np.random.randint(1, max_n_graph + 1)
  n_node = np.random.randint(0, 10, n_graph)
  n_edge = np.random.randint(0, 20, n_graph)
  # We cannot have any edges if there are no nodes.
  n_edge[n_node == 0] = 0

  senders = []
  receivers = []
  offset = 0
  for n_node_in_graph, n_edge_in_graph in zip(n_node, n_edge):
    if n_edge_in_graph != 0:
      senders += list(
          np.random.randint(0, n_node_in_graph, n_edge_in_graph) + offset)
      receivers += list(
          np.random.randint(0, n_node_in_graph, n_edge_in_graph) + offset)
    offset += n_node_in_graph

  return graph.GraphsTuple(
      n_node=jnp.asarray(n_node),
      n_edge=jnp.asarray(n_edge),
      nodes=jnp.asarray(np.random.random(size=(np.sum(n_node), 4))),
      edges=jnp.asarray(np.random.random(size=(np.sum(n_edge), 3))),
      globals=jnp.asarray(np.random.random(size=(n_graph, 5))),
      senders=jnp.asarray(senders),
      receivers=jnp.asarray(receivers))


def _get_graph_network(graphs_tuple):
  # Our test update functions are just identity functions.
  update_node_fn = lambda n, se, re, g: n
  update_edge_fn = lambda e, sn, rn, g: e
  update_globals_fn = lambda gn, ge, g: g
  net = graph.GraphNetwork(update_edge_fn,
                           update_node_fn,
                           update_globals_fn)
  return net(graphs_tuple)


class GraphTest(test_util.JaxTestCase):

  def test_connect_graphnetwork(self):
    _, batched_graphs_tuple = self._get_list_and_batched_graph()
    init_fn, apply_fn = base.transform(_get_graph_network)
    params = init_fn(random.PRNGKey(428), batched_graphs_tuple)
    with self.subTest('nojit'):
      out = apply_fn(params, batched_graphs_tuple)
      self.assertAllClose(out, batched_graphs_tuple, check_dtypes=True)
    with self.subTest('jit'):
      out = jax.jit(apply_fn)(params, batched_graphs_tuple)
      self.assertAllClose(out, batched_graphs_tuple, check_dtypes=True)

  def _get_list_and_batched_graph(self):
    """Returns a list of individual graphs and a batched version.

    This test-case includes the following corner-cases:
      - single node,
      - multiple nodes,
      - no edges,
      - single edge,
      - and multiple edges.
    """
    batched_graph = graph.GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0, 2, 0, 0]),
        n_edge=jnp.array([2, 5, 0, 0, 1, 0, 0]),
        nodes=jnp.arange(14).reshape(7, 2),
        edges=jnp.arange(24).reshape(8, 3),
        globals=jnp.arange(14).reshape(7, 2),
        senders=jnp.array([0, 0, 1, 1, 2, 3, 3, 6]),
        receivers=jnp.array([0, 0, 2, 1, 3, 2, 1, 5])
    )

    list_graphs = [
        graph.GraphsTuple(n_node=jnp.array([1]),
                          n_edge=jnp.array([2]),
                          nodes=jnp.array([[0, 1]]),
                          edges=jnp.array([[0, 1, 2], [3, 4, 5]]),
                          globals=jnp.array([[0, 1]]),
                          senders=jnp.array([0, 0]),
                          receivers=jnp.array([0, 0])),
        graph.GraphsTuple(n_node=jnp.array([3]),
                          n_edge=jnp.array([5]),
                          nodes=jnp.array([[2, 3], [4, 5], [6, 7]]),
                          edges=jnp.array([[6, 7, 8], [9, 10, 11], [12, 13, 14],
                                           [15, 16, 17], [18, 19, 20]]),
                          globals=jnp.array([[2, 3]]),
                          senders=jnp.array([0, 0, 1, 2, 2]),
                          receivers=jnp.array([1, 0, 2, 1, 0])),
        graph.GraphsTuple(n_node=jnp.array([1]),
                          n_edge=jnp.array([0]),
                          nodes=jnp.array([[8, 9]]),
                          edges=jnp.zeros((0, 3)),
                          globals=jnp.array([[4, 5]]),
                          senders=jnp.array([]),
                          receivers=jnp.array([])),
        graph.GraphsTuple(n_node=jnp.array([0]),
                          n_edge=jnp.array([0]),
                          nodes=jnp.zeros((0, 2)),
                          edges=jnp.zeros((0, 3)),
                          globals=jnp.array([[6, 7]]),
                          senders=jnp.array([]),
                          receivers=jnp.array([])),
        graph.GraphsTuple(n_node=jnp.array([2]),
                          n_edge=jnp.array([1]),
                          nodes=jnp.array([[10, 11], [12, 13]]),
                          edges=jnp.array([[21, 22, 23]]),
                          globals=jnp.array([[8, 9]]),
                          senders=jnp.array([1]),
                          receivers=jnp.array([0])),
        graph.GraphsTuple(n_node=jnp.array([0]),
                          n_edge=jnp.array([0]),
                          nodes=jnp.zeros((0, 2)),
                          edges=jnp.zeros((0, 3)),
                          globals=jnp.array([[10, 11]]),
                          senders=jnp.array([]),
                          receivers=jnp.array([])),
        graph.GraphsTuple(n_node=jnp.array([0]),
                          n_edge=jnp.array([0]),
                          nodes=jnp.zeros((0, 2)),
                          edges=jnp.zeros((0, 3)),
                          globals=jnp.array([[12, 13]]),
                          senders=jnp.array([]),
                          receivers=jnp.array([]))]

    return list_graphs, batched_graph

  def test_batch(self):
    """Tests batching of graph."""
    list_graphs_tuple, batched_graphs_tuple = self._get_list_and_batched_graph()
    graphs_tuple = graph.batch(list_graphs_tuple)
    self.assertAllClose(graphs_tuple, batched_graphs_tuple, check_dtypes=True)

  def test_unbatch(self):
    """Tests unbatching of graph."""
    list_graphs_tuple, batched_graphs_tuple = self._get_list_and_batched_graph()
    graphs_tuples = graph.unbatch(batched_graphs_tuple)
    self.assertAllClose(graphs_tuples, list_graphs_tuple, check_dtypes=True)

  def test_batch_unbatch_with_random_graphs(self):
    """Tests batch(unbatch) is identity with random graphs."""
    np.random.seed(42)
    for _ in range(100):
      g = _get_random_graph()
      self.assertAllClose(
          graph.batch(graph.unbatch(g)), g, check_dtypes=True)

    for _ in range(10):
      graphs1 = [_get_random_graph(1) for _ in range(np.random.randint(1, 10))]
      graphs2 = graph.unbatch(graph.batch(graphs1))
      for g1, g2 in zip(graphs1, graphs2):
        self.assertAllClose(g1, g2, check_dtypes=True)

  def test_pad(self):
    """Tests padding of graph."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    padded_graphs_tuple = graph.pad(graphs_tuple, 10, 12, 9)
    expected_padded_graph = graph.GraphsTuple(
        n_node=jnp.concatenate([graphs_tuple.n_node, jnp.array([3, 0])]),
        n_edge=jnp.concatenate([graphs_tuple.n_edge, jnp.array([4, 0])]),
        nodes=jnp.concatenate([graphs_tuple.nodes, jnp.zeros((3, 2))]),
        edges=jnp.concatenate([graphs_tuple.edges, jnp.zeros((4, 3))]),
        globals=jnp.concatenate([graphs_tuple.globals, jnp.zeros((2, 2))]),
        senders=jnp.concatenate([graphs_tuple.senders,
                                 jnp.array([7, 7, 7, 7])]),
        receivers=jnp.concatenate([graphs_tuple.receivers,
                                   jnp.array([7, 7, 7, 7])]),
    )
    self.assertAllClose(
        padded_graphs_tuple, expected_padded_graph, check_dtypes=True)

  def test_unpad(self):
    """Tests unpadding of graph."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    unpadded_graphs_tuple = graph.unpad(graphs_tuple)
    expected_unpadded_graph = graph.GraphsTuple(
        n_node=jnp.array([1, 3, 1, 0]),
        n_edge=jnp.array([2, 5, 0, 0]),
        nodes=jnp.arange(10).reshape(5, 2),
        edges=jnp.arange(21).reshape(7, 3),
        globals=jnp.arange(8).reshape(4, 2),
        senders=jnp.array([0, 0, 1, 1, 2, 3, 3]),
        receivers=jnp.array([0, 0, 2, 1, 3, 2, 1])
    )
    self.assertAllClose(
        unpadded_graphs_tuple, expected_unpadded_graph, check_dtypes=True)

  def test_pad_unpad_with_random_graphs(self):
    """Tests unpad(pad) is identity with random graphs."""
    np.random.seed(42)
    for _ in range(100):
      g = _get_random_graph()
      self.assertAllClose(
          graph.unpad(graph.pad(g, 101, 200, 11)), g, check_dtypes=True)

  def test_get_number_of_padding_graphs(self):
    """Tests the number of padding graphs calculation."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected = 3
    with self.subTest('nojit'):
      self.assertAllClose(graph.get_number_of_padding_graphs(graphs_tuple),
                          expected, check_dtypes=True)
    with self.subTest('jit'):
      self.assertAllClose(
          jax.jit(graph.get_number_of_padding_graphs)(graphs_tuple),
          expected, check_dtypes=True)

  def test_get_number_of_padding_nodes(self):
    """Tests the number of padding nodes calculation."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected = 2
    with self.subTest('nojit'):
      self.assertAllClose(graph.get_number_of_padding_nodes(graphs_tuple),
                          expected, check_dtypes=True)
    with self.subTest('jit'):
      self.assertAllClose(
          jax.jit(graph.get_number_of_padding_nodes)(graphs_tuple),
          expected, check_dtypes=True)

  def test_get_number_of_padding_edges(self):
    """Tests the number of padding edges calculation."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected = 1
    with self.subTest('nojit'):
      self.assertAllClose(graph.get_number_of_padding_edges(graphs_tuple),
                          expected, check_dtypes=True)
    with self.subTest('jit'):
      self.assertAllClose(
          jax.jit(graph.get_number_of_padding_edges)(graphs_tuple),
          expected, check_dtypes=True)

  def test_get_node_padding_mask(self):
    """Tests construction of node padding mask."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected_mask = jnp.array([1, 1, 1, 1, 1, 0, 0], dtype=jnp.int32)
    with self.subTest('nojit'):
      mask = graph.get_node_padding_mask(graphs_tuple)
      self.assertAllClose(mask, expected_mask, check_dtypes=True)
    with self.subTest('jit'):
      mask = jax.jit(graph.get_node_padding_mask)(graphs_tuple)
      self.assertAllClose(mask, expected_mask, check_dtypes=True)

  def test_get_edge_padding_mask(self):
    """Tests construction of edge padding mask."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected_mask = jnp.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=jnp.int32)
    with self.subTest('nojit'):
      mask = graph.get_edge_padding_mask(graphs_tuple)
      self.assertAllClose(mask, expected_mask, check_dtypes=True)
    with self.subTest('jit'):
      mask = jax.jit(graph.get_edge_padding_mask)(graphs_tuple)
      self.assertAllClose(mask, expected_mask, check_dtypes=True)

  def test_get_graph_padding_mask(self):
    """Tests construction of graph padding mask."""
    _, graphs_tuple = self._get_list_and_batched_graph()
    expected_mask = jnp.array([1, 1, 1, 1, 0, 0, 0], dtype=jnp.int32)
    with self.subTest('nojit'):
      mask = graph.get_graph_padding_mask(graphs_tuple)
      self.assertAllClose(mask, expected_mask, check_dtypes=True)
    with self.subTest('jit'):
      mask = jax.jit(graph.get_graph_padding_mask)(graphs_tuple)
      self.assertAllClose(mask, expected_mask, check_dtypes=True)


if __name__ == '__main__':
  absltest.main()
