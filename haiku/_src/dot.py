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
import html
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


def name_or_str(o):
  return getattr(o, '__name__', str(o))


def to_graph(fun):
  """Converts a Haiku function into an graph IR (extracted for testing)."""
  @functools.wraps(fun)
  def wrapped_fun(*args):
    """See `fun`."""
    f = jax.linear_util.wrap_init(fun)
    args_flat, in_tree = jax.tree_flatten((args, {}))
    flat_fun, out_tree = jax.api_util.flatten_fun(f, in_tree)
    graph = Graph.create(title=name_or_str(fun))

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
      out_flat = _interpret_subtrace(flat_fun, master).call_wrapped(*args_flat)
    out = jax.tree_unflatten(out_tree(), out_flat)

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
    return jax.core.get_aval(self.val)

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
    graph = Graph.create(title=f'{call_primitive} ({name_or_str(f.f)})')
    graph_stack.peek().subgraphs.append(graph)
    with graph_stack(graph):
      f = _interpret_subtrace(f, self.master)
      vals_out = f.call_wrapped(*[t.val for t in tracers])
      return [DotTracer(self, v) for v in vals_out]


def _format_val(val):
  if not hasattr(val, 'shape'):
    return repr(val)
  shape = ','.join(map(str, val.shape))
  dtype = val.dtype.name
  dtype = dtype.replace('complex', 'c')
  dtype = dtype.replace('float', 'f')
  dtype = dtype.replace('uint', 'u')
  dtype = dtype.replace('int', 's')
  return f'{dtype}[{shape}]'


def escape(value):
  return html.escape(str(value))


# Determine maximum nesting depth to appropriately scale subgraph labels.
def _max_depth(g: Graph) -> int:
  if g.subgraphs:
    return 1 + max(0, *[_max_depth(s) for s in g.subgraphs])
  else:
    return 1


def _scaled_font_size(depth: int) -> int:
  return int(1.4**depth * 14)


def _graph_to_dot(graph: Graph, args, outputs):
  """Converts from an internal graph IR to 'dot' format."""

  def format_path(path):
    if isinstance(outputs, tuple):
      out = f'output[{path[0]}]'
      if len(path) > 1:
        out += ': ' + '/'.join(map(str, path[1:]))
    else:
      out = 'output'
      if path:
        out += ': ' + '/'.join(map(str, path))
    return out

  lines = []
  used_argids = set()
  argid_usecount = collections.Counter()
  op_outids = set()
  captures = []
  argids = {id(v) for v in jax.tree_leaves(args)}
  outids = {id(v) for v in jax.tree_leaves(outputs)}
  outname = {id(v): format_path(p) for p, v in tree.flatten_with_path(outputs)}

  def render_graph(g: Graph, parent: Optional[Graph] = None, depth: int = 0):
    """Renders a given graph by appending 'dot' format lines."""

    if parent:
      lines.extend([
          f'subgraph cluster_{id(g)} {{',
          '  style="rounded,filled";',
          '  fillcolor="#F0F5F5";',
          '  color="#14234B;";',
          '  pad=0.1;',
          f'  fontsize={_scaled_font_size(depth)};',
          f'  label = <<b>{escape(g.title)}</b>>;',
          '  labelloc = t;',
      ])

    for node in g.nodes:
      label = f'<b>{escape(node.title)}</b>'
      for o in node.outputs:
        label += '<br/>' + _format_val(o)
        op_outids.add(id(o))

      node_id = id(node.id)
      if node_id in outids:
        label = f'<b>{escape(outname[node_id])}</b><br/>' + label
        color = '#0053D6'
        fillcolor = '#AABFFF'
        style = 'filled,bold'
      else:
        color = '#FFDB13'
        fillcolor = '#FFF26E'
        style = 'filled'

      lines.append(f'{node_id} [label=<{label}>, '
                   f' id="node{node_id}",'
                   ' shape=rect,'
                   f' style="{style}",'
                   ' tooltip=" ",'
                   ' fontcolor="black",'
                   f' color="{color}",'
                   f' fillcolor="{fillcolor}"];')

    for s in g.subgraphs:
      render_graph(s, parent=g, depth=depth - 1)

    if parent:
      lines.append(f'}}  // subgraph cluster_{id(g)}')

    for a, b in g.edges:
      if id(a) not in argids and id(a) not in op_outids:
        captures.append(a)

      a, b = map(id, (a, b))
      if a in argids:
        i = argid_usecount[a]
        argid_usecount[a] += 1
        lines.append(f'{a}{i} -> {b};')
      else:
        lines.append(f'{a} -> {b};')
      used_argids.add(a)

  graph_depth = _max_depth(graph)
  render_graph(graph, parent=None, depth=graph_depth)

  # Process inputs and label them in the graph.
  for path, value in tree.flatten_with_path(args):
    if value is None:
      continue

    node_id = id(value)
    if node_id not in used_argids:
      continue

    for i in range(argid_usecount[node_id]):
      label = f'<b>args[{escape(path[0])}]'
      if len(path) > 1:
        label += ': ' + '/'.join(map(str, path[1:]))
      label += '</b>'
      if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        label += f'<br/>{escape(_format_val(value))}'
      fillcolor = '#FFDEAF'
      fontcolor = 'black'

      if i > 0:
        label = '<b>(reuse)</b><br/>' + label
        fillcolor = '#FFEACC'
        fontcolor = '#565858'

      lines.append(f'{node_id}{i} [label=<{label}>'
                   f' id="node{node_id}{i}",'
                   ' shape=rect,'
                   ' style="filled",'
                   f' fontcolor="{fontcolor}",'
                   ' color="#FF8A4F",'
                   f' fillcolor="{fillcolor}"];')

  for value in captures:
    node_id = id(value)
    if hasattr(value, 'size') and value.size == 1:
      label = f'<b>{value.item()}</b>'
    else:
      label = f'<b>{escape(_format_val(value))}</b>'

    lines.append(f'{node_id} [label=<{label}>'
                 ' shape=rect,'
                 ' style="filled",'
                 ' fontcolor="black",'
                 ' color="#A261FF",'
                 ' fillcolor="#E6D6FF"];')

  head = [
      'digraph G {',
      'rankdir = TD;',
      'compound = true;',
      f'label = <<b>{escape(graph.title)}</b>>;',
      f'fontsize={_scaled_font_size(graph_depth)};',
      'labelloc = t;',
      'stylesheet = <',
      '  data:text/css,',
      '  @import url(https://fonts.googleapis.com/css?family=Roboto:400,700);',
      '  svg text {',
      '    font-family: \'Roboto\';',
      '  }',
      '  .node text {',
      '    font-size: 12px;',
      '  }',
  ]
  for node_id, use_count in argid_usecount.items():
    if use_count == 1:
      continue
    # Add hover animation for reused args.
    for a in range(use_count):
      for b in range(use_count):
        if a == b:
          head.append(f'%23node{node_id}{a}:hover '
                      '{ stroke-width: 0.2em; }')
        else:
          head.append(
              f'%23node{node_id}{a}:hover ~ %23node{node_id}{b} '
              '{ stroke-width: 0.2em; }')
  head.append('>')

  lines.append('} // digraph G')
  return '\n'.join(head + lines) + '\n'
