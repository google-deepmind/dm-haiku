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
r"""Zachary's karate club example.

Here we train a graph neural network to process Zachary's karate club.
https://en.wikipedia.org/wiki/Zachary%27s_karate_club

Zachary's karate club is used in the literature as an example of a social graph.
Here we we a graphnet to optimize the assignments of the students in the
karate club to two distinct karate instructors (Mr. Hi and John A).
"""

import logging
from absl import app
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp


def get_zacharys_karate_club() -> hk.GraphsTuple:
  """Returns GraphsTuple representing Zachary's karate club."""
  social_graph = [
      (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
      (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
      (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
      (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
      (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
      (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
      (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
      (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
      (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
      (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
      (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
      (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
      (33, 31), (33, 32)]
  # Add reverse edges.
  social_graph += [(edge[1], edge[0]) for edge in social_graph]
  n_club_members = 34

  return hk.GraphsTuple(
      n_node=jnp.asarray([n_club_members]),
      n_edge=jnp.asarray([len(social_graph)]),
      # One-hot encoding for nodes.
      nodes=jnp.eye(n_club_members),
      edges=jnp.zeros((len(social_graph), n_club_members)),
      globals=jnp.zeros((1, 0)),
      senders=jnp.asarray([edge[0] for edge in social_graph]),
      receivers=jnp.asarray([edge[1] for edge in social_graph]))


def get_ground_truth_assignments_for_zacharys_karate_club() -> jnp.DeviceArray:
  """Returns ground truth assignments for Zachary's karate club."""
  return jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                    0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def network_definition(graph: hk.GraphsTuple) -> jnp.DeviceArray:
  """Defines a graph neural network.

  Args:
    graph: GraphsTuple the network processes.

  Returns:
    processed nodes.
  """
  gn = hk.GraphNetwork(
      update_edge_fn=lambda e, s, r, g: s,
      update_node_fn=lambda n, o, i, g: jax.nn.relu(hk.Linear(5)(i)),
      update_globals_fn=lambda n, e, g: g)
  graph = gn(graph)

  gn = hk.GraphNetwork(
      update_edge_fn=lambda e, s, r, g: s,
      update_node_fn=lambda n, o, i, g: hk.Linear(2)(i),
      update_globals_fn=lambda n, e, g: g)
  graph = gn(graph)
  return graph.nodes


def optimize_club(num_steps: int):
  """Solves the karte club problem by optimizing the assignments of students."""
  network = hk.transform(network_definition)
  zacharys_karate_club = get_zacharys_karate_club()
  labels = get_ground_truth_assignments_for_zacharys_karate_club()
  params = network.init(jax.random.PRNGKey(42), zacharys_karate_club)

  @jax.jit
  def prediction_loss(params):
    decoded_nodes = network.apply(params, zacharys_karate_club)
    # We interpret the decoded nodes as a pair of logits for each node.
    log_prob = jax.nn.log_softmax(decoded_nodes)
    # The only two assignments we know a-priori are those of Mr. Hi (Node 0)
    # and John A (Node 33).
    return -(log_prob[0, 0] + log_prob[33, 1])

  opt_init, opt_update = optix.adam(1e-2)
  opt_state = opt_init(params)

  @jax.jit
  def update(params, opt_state):
    g = jax.grad(prediction_loss)(params)
    updates, opt_state = opt_update(g, opt_state)
    return optix.apply_updates(params, updates), opt_state

  @jax.jit
  def accuracy(params):
    decoded_nodes = network.apply(params, zacharys_karate_club)
    return jnp.mean(jnp.argmax(decoded_nodes, axis=1) == labels)

  for step in range(num_steps):
    logging.info("step %r accuracy %r", step, accuracy(params).item())
    params, opt_state = update(params, opt_state)


def main(_):
  optimize_club(num_steps=30)


if __name__ == "__main__":
  app.run(main)
