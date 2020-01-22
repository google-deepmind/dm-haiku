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
r"""2-SAT solver example.

Here we train a graph neural network to solve 2-sat problems.
https://en.wikipedia.org/wiki/2-satisfiability

For instance a 2 sat problem with 3 literals would look like this:
   (a or b)  and  (not a or c)  and (not b or not c)

We represent this problem in form of a bipartite-graph, with edges
connecting the literal-nodes (a, b, c) with the constraint-nodes (O).
The corresponding graph looks like this:
     O    O   O
     |\  /\  /|
     | \/  \/ |
     | /\  /\ |
     |/  \/  \|
     a    b   c

The nodes are one-hot encoded with literal nodes as (1, 0) and constraint nodes
as (0, 1). The edges are one-hot encoded with (1, 0) if the literal should be
true and (0, 1) if the literal should be false.

The graph neural network encodes the nodes and the edges and runs multiple
message passing steps by calculting message for each edge and aggregating
all the messages of the nodes.

The training dataset consists of randomly generated 2-sat problems with 2 to 15
literals.
The test dataset consists of randomly generated 2-sat problems with 16 to 20
literals.
"""

import collections
import logging
import random

from absl import app
import haiku as hk
import jax
from jax.experimental import optix
import jax.numpy as jnp


Problem = collections.namedtuple("Problem", ("graph", "labels", "mask"))


def get_2sat_problem(min_n_literals: int, max_n_literals: int) -> Problem:
  """Creates bipartite-graph representing a randomly generated 2-sat problem.

  Args:
    min_n_literals: minimum number of literals in the 2-sat problem.
    max_n_literals: maximum number of literals in the 2-sat problem.

  Returns:
    bipartite-graph, node labels and node mask.
  """
  n_literals = random.randint(min_n_literals, max_n_literals)
  n_literals_true = random.randint(1, n_literals - 1)
  n_constraints = n_literals * (n_literals + 1) // 2

  n_node = n_literals +  n_constraints
  # 0 indicates a literal node
  # 1 indicates a constraint node.
  nodes = [0 if i < n_literals else 1 for i in range(n_node)]
  edges = []
  senders = []
  for literal_node1 in range(n_literals):
    for literal_node2 in range(literal_node1, n_literals):
      senders.append(literal_node1)
      senders.append(literal_node2)
      # 1 indicates that the literal must be true for this constraint.
      # 0 indicates that the literal must be false for this constraint.
      # I.e. with literals a and b, we have the following possible constraints:
      # 0, 0 -> a or b
      # 1, 0 -> not a or b
      # 0, 1 -> a or not b
      # 1, 1 -> not a or not b
      edges.append(1 if literal_node1 < n_literals_true else 0)
      edges.append(1 if literal_node2 < n_literals_true else 0)

  graph = hk.GraphsTuple(
      n_node=jnp.asarray([n_node]),
      n_edge=jnp.asarray([2 * n_constraints]),
      # One-hot encoding for nodes and edges.
      nodes=jnp.eye(2)[nodes],
      edges=jnp.eye(2)[edges],
      globals=jnp.zeros((1, 0)),
      senders=jnp.asarray(senders),
      receivers=jnp.repeat(jnp.arange(n_constraints) + n_literals, 2))

  # In order to jit compile our code, we have to pad the nodes and edges of
  # the GraphsTuple to a static shape.
  max_n_constraints = max_n_literals * (max_n_literals + 1) // 2
  max_nodes = max_n_literals + max_n_constraints  + 1
  max_edges = 2 * max_n_constraints
  graph = hk.graph.pad(graph, max_nodes, max_edges)

  # The ground truth solution for the 2-sat problem.
  labels = (jnp.arange(max_nodes) < n_literals_true).astype(jnp.int32)
  labels = jnp.eye(2)[labels]

  # For the loss calculation we create a mask for the nodes, which masks the
  # the constraint nodes and the padding nodes.
  mask = (jnp.arange(max_nodes) < n_literals).astype(jnp.int32)
  return Problem(graph=graph, labels=labels, mask=mask)


def network_definition(
    graph: hk.GraphsTuple,
    num_message_passing_steps: int = 5) -> jnp.DeviceArray:
  """Defines a graph neural network.

  Args:
    graph: Graphstuple the network processes.
    num_message_passing_steps: number of message passing steps.

  Returns:
    Decoded nodes.
  """
  node_encoder = hk.Linear(output_size=16)
  edge_encoder = hk.Linear(output_size=16)

  def update_fn(*args):
    net = hk.Sequential([
        hk.Linear(10), jax.nn.relu,
        hk.Linear(10), jax.nn.relu,
        hk.Linear(10), jax.nn.relu])
    return net(jnp.concatenate(args, axis=-1))

  graph = graph._replace(
      nodes=node_encoder(graph.nodes),
      edges=edge_encoder(graph.edges))

  for _ in range(num_message_passing_steps):
    gn = hk.GraphNetwork(
        update_edge_fn=update_fn,
        update_node_fn=update_fn,
        update_globals_fn=lambda n, e, g: g)
    graph = gn(graph)

  return hk.Linear(2)(graph.nodes)


def train(num_steps: int):
  """Trains a graph neural network on a 2-sat problem."""
  train_dataset = (2, 15)
  test_dataset = (16, 20)
  random.seed(42)

  network = hk.transform(network_definition)
  problem = get_2sat_problem(*train_dataset)
  params = network.init(jax.random.PRNGKey(42), problem.graph)

  @jax.jit
  def prediction_loss(params, problem):
    decoded_nodes = network.apply(params, problem.graph)
    # We interpret the decoded nodes as a pair of logits for each node.
    log_prob = jax.nn.log_softmax(decoded_nodes) * problem.labels
    return -jnp.sum(log_prob * problem.mask[:, None]) / jnp.sum(problem.mask)

  opt_init, opt_update = optix.adam(2e-4)
  opt_state = opt_init(params)

  @jax.jit
  def update(params, opt_state, problem):
    g = jax.grad(prediction_loss)(params, problem)
    updates, opt_state = opt_update(g, opt_state)
    return optix.apply_updates(params, updates), opt_state

  for step in range(num_steps):
    problem = get_2sat_problem(*train_dataset)
    params, opt_state = update(params, opt_state, problem)
    if step % 1000 == 0:
      train_loss = jnp.mean(
          jnp.asarray([
              prediction_loss(params, get_2sat_problem(*train_dataset))
              for _ in range(100)
          ])).item()
      test_loss = jnp.mean(
          jnp.asarray([
              prediction_loss(params, get_2sat_problem(*test_dataset))
              for _ in range(100)
          ])).item()
      logging.info("step %r loss train %r test %r", step, train_loss, test_loss)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  train(num_steps=10000)


if __name__ == "__main__":
  app.run(main)
