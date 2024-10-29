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
"""Tests for haiku._src.dot."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import dot
from haiku._src import module
from haiku._src import test_utils
import jax
import jax.numpy as jnp


class DotTest(parameterized.TestCase):

  def test_empty(self):
    self.skipTest("TODO: re-enable once new JAX version lands")
    graph, args, out = dot.to_graph(lambda: None)()
    self.assertEmpty(args)
    self.assertIsNone(out)
    self.assertEmpty(graph.nodes)
    self.assertEmpty(graph.edges)
    self.assertEmpty(graph.subgraphs)

  @test_utils.transform_and_run
  def test_add_module(self):
    self.skipTest("TODO: re-enable once new JAX version lands")
    mod = AddModule()
    a = b = jnp.ones([])
    graph, args, c = dot.to_graph(mod)(a, b)
    self.assertEqual(args, (a, b))
    self.assertEqual(c, a + b)
    self.assertEmpty(graph.edges)
    add_graph, = graph.subgraphs
    self.assertEqual(add_graph.title, "add_module")
    self.assertEmpty(add_graph.subgraphs)
    add_edge_a, add_edge_b = add_graph.edges
    self.assertEqual(add_edge_a, (a, c))
    self.assertEqual(add_edge_b, (b, c))
    add_node, = add_graph.nodes
    self.assertEqual(add_node.title, "add")
    add_out, = add_node.outputs
    self.assertEqual(add_out, c)

  @test_utils.transform_and_run
  def test_inline_jit_add_module(self):
    self.skipTest("TODO: re-enable once new JAX version lands")
    mod = InlineJitAddModule()
    a = b = jnp.ones([])
    graph, args, c = dot.to_graph(mod)(a, b)
    self.assertEqual(args, (a, b))
    self.assertEqual(c, a + b)
    self.assertEmpty(graph.edges)
    add_graph, = graph.subgraphs
    self.assertEqual(add_graph.title, "inline_jit_add_module")
    self.assertEmpty(add_graph.subgraphs)
    add_edge_a, add_edge_b = add_graph.edges
    self.assertEqual(add_edge_a, (a, c))
    self.assertEqual(add_edge_b, (b, c))
    add_node, = add_graph.nodes
    self.assertEqual(add_node.title, "add")
    add_out, = add_node.outputs
    self.assertEqual(add_out, c)

  def test_call(self):
    self.skipTest("TODO: re-enable once new JAX version lands")
    def my_function(x):
      return x

    graph, _, _ = dot.to_graph(jax.jit(my_function))(jnp.ones([]))
    self.assertEmpty(graph.nodes)
    self.assertEmpty(graph.edges)
    jit, = graph.subgraphs
    self.assertIn(jit.title, ("pjit (my_function)",))

  def test_pmap(self):
    self.skipTest("TODO: re-enable once new JAX version lands")
    def my_function(x):
      return x

    n = jax.local_device_count()
    graph, _, _ = dot.to_graph(jax.pmap(my_function))(jnp.ones([n]))
    self.assertEmpty(graph.nodes)
    self.assertEmpty(graph.edges)
    jit, = graph.subgraphs
    self.assertEqual(jit.title, "xla_pmap (my_function)")


class AddModule(module.Module):

  def __call__(self, a, b):
    return a + b


class InlineJitAddModule(module.Module):

  def __call__(self, a, b):
    return jax.jit(lambda x, y: x + y, inline=True)(a, b)

if __name__ == "__main__":
  absltest.main()
