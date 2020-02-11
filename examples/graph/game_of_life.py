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
"""Implementation of Conway's game of life using hk.graph."""

import time

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def conway_mlp(x):
  """Implements a MLP representing Conway's game of life rules."""
  w = jnp.array([[0.0, -1.0], [0.0, 1.0], [0.0, 1.0],
                 [0, -1.0], [1.0, 1.0], [1.0, 1.0]])
  b = jnp.array([3.5, -3.5, -1.5, 1.5, -2.5, -3.5])
  h = jnp.maximum(jnp.dot(w, x) + b, 0.)
  w = jnp.array([[2.0, -4.0, 2.0, -4.0, 2.0, -4.0]])
  b = jnp.array([-4.0])
  y = jnp.maximum(jnp.dot(w, h) + b, 0.0)
  return y


def conway_graph(size) -> hk.graph.GraphsTuple:
  """Returns a graph representing the game field of conway's game of life."""
  # Creates nodes: each node represents a cell in the game.
  n_node = size**2
  nodes = np.zeros((n_node, 1))
  node_indices = jnp.arange(n_node)
  # Creates edges, senders and receivers:
  # the senders represent the connections to the 8 neighboring fields.
  n_edge = 8 * n_node
  edges = jnp.zeros((n_edge, 1))
  senders = jnp.vstack(
      [node_indices - size - 1, node_indices - size, node_indices - size + 1,
       node_indices - 1, node_indices + 1,
       node_indices + size - 1, node_indices + size, node_indices + size + 1])
  senders = senders.T.reshape(-1)
  senders = (senders + size**2) % size**2
  receivers = jnp.repeat(node_indices, 8)
  # Adds a glider to the game
  nodes[0, 0] = 1.0
  nodes[1, 0] = 1.0
  nodes[2, 0] = 1.0
  nodes[2 + size, 0] = 1.0
  nodes[1 + 2 * size, 0] = 1.0
  return hk.graph.GraphsTuple(n_node=jnp.array([n_node]),
                              n_edge=jnp.array([n_edge]),
                              nodes=jnp.asarray(nodes),
                              edges=edges,
                              globals=jnp.array([[1.0]]),
                              senders=senders,
                              receivers=receivers)


def display_graph(graph: hk.graph.GraphsTuple):
  """Prints the nodes of the graph representing Conway's game of life."""
  size = int(np.sqrt(np.sum(graph.n_node)))

  def _display_node(node):
    if node == 1.0:
      return 'x'
    else:
      return ' '

  nodes = graph.nodes.copy()
  output = '\n'.join(
      ''.join(_display_node(nodes[i * size + j][0])
              for j in range(size))
      for i in range(size))
  print('-' * size + '\n' + output)


def main(_):

  def net_fn(graph: hk.graph.GraphsTuple):
    unf = lambda n, e_s, e_r, g: conway_mlp(jnp.concatenate([n, e_r], axis=-1))
    net = hk.graph.GraphNetwork(
        update_edge_fn=lambda e, n_s, n_r, g: n_s,
        update_node_fn=unf,
        update_globals_fn=lambda n, e, g: g)
    return net(graph)

  net = hk.transform(net_fn)

  cg = conway_graph(size=20)
  params = net.init(jax.random.PRNGKey(42), cg)
  for _ in range(100):
    time.sleep(0.05)
    cg = jax.jit(net.apply)(params, cg)
    display_graph(cg)

if __name__ == '__main__':
  app.run(main)
