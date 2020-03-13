# Lint as: python3
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
"""Converts Haiku functions to dot."""

import collections
import contextlib
import functools
from typing import NamedTuple, List, Optional

from haiku._src import data_structures
from haiku._src import module
import jax
import tree


graph_stack = data_structures.ThreadLocalStack()
Node = collections.namedtuple('Node', 'id,title,outputs')
Edge = collections.namedtuple('Edge', 'a,b')


class Graph(NamedTuple):
  """Represents a graphviz digraph/subgraph.."""

  title: str
  nodes: List[Node]
  edges: List[Edge]
  subgraphs: List['Graph']

  @classmethod
  def create(cls, title: str = None):
    return Graph(title=title, nodes=[], edges=[], subgraphs=[])

  def evolve(self, **kwargs) -> 'Graph':
    return Graph(**{**self._asdict(), **kwargs})


def to_dot(fun):
  """Converts a function using Haiku modules to a dot graph."""
  graph_fun = to_graph(fun)
  @functools.wraps(fun)
  def wrapped_fun(*args):
    return _graph_to_dot(*graph_fun(*args))
  return wrapped_fun


def to_graph(fun):
  """Converts a Haiku function into an graph IR (extracted for testing)."""
  @functools.wraps(fun)
  def wrapped_fun(*args):
    """See `fun`."""
    f = jax.linear_util.wrap_init(fun)
    args_flat, in_tree = jax.tree_flatten((args, {}))
    flat_fun, _ = jax.api_util.flatten_fun(f, in_tree)
    graph = Graph.create(title=getattr(fun, '__name__', str(fun)))

    @contextlib.contextmanager
    def method_hook(mod: module.Module, method_name: str):
      subg = Graph.create()
      with graph_stack(subg):
        yield
      title = mod.module_name
      if method_name != '__call__':
        title += f' ({method_name})'
      graph_stack.peek().subgraphs.append(subg.evolve(title=title))

    with graph_stack(graph), \
         module.hook_methods(method_hook), \
         jax.core.new_master(DotTrace) as master:
      out = _interpret_subtrace(flat_fun, master).call_wrapped(*args_flat)

    return graph, args, out

  return wrapped_fun


@jax.linear_util.transformation
def _interpret_subtrace(master, *in_vals):
  trace = DotTrace(master, jax.core.cur_sublevel())
  in_tracers = [DotTracer(trace, val) for val in in_vals]
  outs = yield in_tracers, {}
  out_tracers = map(trace.full_raise, outs)
  out_vals = [t.val for t in out_tracers]
  yield out_vals


class DotTracer(jax.core.Tracer):
  """JAX tracer used in DotTrace."""

  def __init__(self, trace, val):
    super().__init__(trace)
    self.val = val

  @property
  def aval(self):
    if isinstance(self.val, jax.core.Tracer):
      return self.val.aval
    return jax.abstract_arrays.ShapedArray(self.val.shape, self.val.dtype)

  def full_lower(self):
    return self


class DotTrace(jax.core.Trace):
  """Traces a JAX function to dot."""

  def pure(self, val):
    return DotTracer(self, val)

  def lift(self, val):
    return DotTracer(self, val)

  def sublift(self, val):
    return DotTracer(self, val.val)

  def process_primitive(self, primitive, tracers, params):
    val_out = primitive.bind(*[t.val for t in tracers], **params)

    inputs = [t.val for t in tracers]
    outputs = list(jax.tree_leaves(val_out))

    graph = graph_stack.peek()
    node = Node(id=outputs[0], title=str(primitive), outputs=outputs)
    graph.nodes.append(node)
    graph.edges.extend([(i, outputs[0]) for i in inputs])

    return jax.tree_map(lambda v: DotTracer(self, v), val_out)

  def process_call(self, call_primitive, f, tracers, params):
    assert call_primitive.multiple_results
    graph = Graph.create(title=str(call_primitive))
    graph_stack.peek().subgraphs.append(graph)
    with graph_stack(graph):
      f = _interpret_subtrace(f, self.master)
      vals_out = f.call_wrapped(*[t.val for t in tracers])
      return [DotTracer(self, v) for v in vals_out]


def _format_val(val):
  shape = ','.join(map(str, val.shape))
  dtype = val.dtype.name
  dtype = dtype.replace('complex', 'c')
  dtype = dtype.replace('float', 'f')
  dtype = dtype.replace('uint', 'u')
  dtype = dtype.replace('int', 's')
  return f'{dtype}[{shape}]'


def _graph_to_dot(graph: Graph, args, outputs):
  """Converts from an internal graph IR to 'dot' format."""

  lines = [
      'digraph G {',
      'rankdir = TD;',
      'compound = true;',
      f'label = <<b>{graph.title}</b>>;',
      'labelloc = t;',
      'stylesheet = <',
      '  data:text/css,',
      '  @import url(https://fonts.googleapis.com/css?family=Roboto:400,700);',
      '  svg text {',
      '    font-family: \'Roboto\';',
      '    font-size: 12px;',
      '  }',
      '>',
  ]

  def render_graph(g: Graph, parent: Optional[Graph] = None):
    """Renders a given graph by appending 'dot' format lines."""

    if parent:
      lines.extend([
          f'subgraph cluster_{id(g)} {{',
          '  style="rounded,filled";',
          '  fillcolor="#F0F5F5";',
          '  color="#14234B;";',
          f'  label = <{g.title}>;',
          '  labelloc = t;',
      ])

    for node in g.nodes:
      label = f'<b>{node.title}</b>'
      for o in node.outputs:
        label += '<br/>' + _format_val(o)

      node_id = id(node.id)
      lines.append(f'{node_id} [label=<{label}>, '
                   ' shape=rect,'
                   ' style="filled",'
                   ' tooltip=" ",'
                   ' fontcolor="black",'
                   ' color="#FFDB13",'
                   ' fillcolor="#FFF26E"];')

    for s in g.subgraphs:
      render_graph(s, parent=g)

    if parent:
      lines.append(f'}}  // subgraph cluster_{id(g)}')

    for a, b in g.edges:
      a, b = map(id, (a, b))
      lines.append(f'{a} -> {b};')

  render_graph(graph, parent=None)

  # Process inputs and label them in the graph.
  for path, value in tree.flatten_with_path(args):
    if value is None:
      continue

    node_id = id(value)
    label = f'<b>args[{path[0]}]'
    if len(path) > 1:
      label += ': ' + '/'.join(map(str, path[1:]))
    label += '</b>'
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
      label += f'<br/>{_format_val(value)}'
    lines.append(f'{node_id} [label=<{label}>'
                 ' shape=rect,'
                 ' style="filled",'
                 ' fontcolor="black",'
                 ' color="#FF8A4F",'
                 ' fillcolor="#FFDEAF"];')

  # Process outputs and label them in the graph.
  if outputs:
    # Inspired by the XLA graph viewer we create an output node and link all the
    # outputs to it.
    lines.append('output [label=<<b>Output</b>>,'
                 ' shape=circle,'
                 ' tooltip=" ",'
                 ' style="filled",'
                 ' fontcolor="black",'
                 ' color="#8c7b75",'
                 ' fillcolor="#bcaaa4"];')

    for output in outputs:
      lines.append(f'{id(output)} -> output;')

  lines.append('} // digraph G')
  return '\n'.join(lines) + '\n'
