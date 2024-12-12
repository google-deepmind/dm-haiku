# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Utilities to visualize JAX expressions and Haiku modules.

The main entry point to this module is `make_model_info`, which captures JAX
expression and Haiku module usage information about the provided callable.

The resulting nested `Module` information can either be consumed directly, or
formatted in a textual representation using `format_module` or in an interactive
HTML tree visualization using `as_html` or `as_html_page`.

Example usage:

    mod = jaxpr_info.make_model_info(my_callable)(some, arguments)
    print(jaxpr_info.format_module(mod))

"""
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import itertools
import logging
import os
import sys
from typing import Any, NamedTuple, TypeAlias

from haiku._src import summarise
import jax
import jax.core
from jax.extend import core as jax_core

# TODO(tomhennigan): Update to use symbols from jax.extend.core when available.
Atom: TypeAlias = jax.core.Atom
DropVar: TypeAlias = jax.core.DropVar


@dataclasses.dataclass
class Module:
  """Information about a Haiku module."""
  # Name of the module, e.g. the Haiku module name.
  name: str

  # How many flops it takes to compute this module, including all operations
  # contained by sub-modules.
  # Only populated if `compute_flops` was passed to `make_model_info`.
  flops: int | None = None

  # Expressions that are directly part of this module, e.g. in the __call__
  # function.
  expressions: list['Expression'] = dataclasses.field(default_factory=list)

  # How many parameters are used by this module, including all sub-modules,
  # and shape information for each parameter. Inferred from haiku module
  # information.
  total_param_size: int = 0
  param_info: dict[str, str] = dataclasses.field(default_factory=dict)

  # Same as above, but for haiku state.
  total_state_size: int = 0
  state_info: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Expression:
  """Information about a single JAX expression."""
  # Type of the expression, e.g. 'add' or 'conv_general_dilated'.
  primitive: str

  # Space separated lists of input and output variables.
  invars: str
  outvars: str

  # Estimated number of flops required to compute this expression.
  # Only populated if `compute_flops` was passed to `make_model_info`.
  flops: int | None = None

  # Additional details, e.g. input/output shapes.
  details: str = ''

  # Some expressions take extra parameters, such as input/output dtypes for
  # casts.
  params: dict[str, str] = dataclasses.field(default_factory=dict)

  # Some expressions, such as named_call, contain a whole subtree of modules
  # and expressions.
  submodule: Module | None = None

  # For internal use only, the first variable in outvars.
  first_outvar: str = ''

  # For internal use, the name scope stack of this expression.
  name_stack: Sequence[str] = dataclasses.field(default_factory=list)


ComputeFlopsFn = Callable[[jax_core.JaxprEqn, Expression], int]


def make_model_info(
    f: Callable[..., Any],
    name: str | None = None,
    include_module_info: bool = True,
    compute_flops: ComputeFlopsFn | None = None,
    axis_env: Sequence[tuple[Any, int]] | None = None,
) -> Callable[..., Module]:
  """Creates a function that computes flop, param and state information.

  Args:
    f: The function for which to compute information. Haiku modules and
      jax.named_call expressions will be represented as nested Modules in the
      result.
    name: Optional, the name of the root expression.
    include_module_info: Whether to include parameter and state count
      information for haiku modules. Can be slow for very large computations.
    compute_flops: Optional, a function that returns an estimate of the number
      of flops required to execute an equation.
    axis_env: Sizes of pmapped axes.  See docs of jax.make_jaxpr for details.

  Returns:
    A wrapped version of `f` that when applied to example arguments returns a
    `Module` representation of `f` for those arguments.

  `Module` and `Expression` contain high level information about JAX operations
  (jaxprs) and can be visualized in concise and interactive formats; see
  `format_module`, `as_html_page` or `as_html`.
  """
  if not name:
    name = f.__name__
  make_jaxpr = jax.make_jaxpr(f, axis_env=axis_env)
  if include_module_info:
    # Wrap f in a lambda so eval_summary doesn't try to un-transform it.
    # TODO(tomhennigan): remove lambda trick
    make_module_info = summarise.eval_summary(lambda *a, **k: f(*a, **k))  # pylint: disable=unnecessary-lambda

  def make_module(*args, **kwargs):
    old_limit = sys.getrecursionlimit()

    try:
      # Increase recursion limit as graphs may be very deep
      sys.setrecursionlimit(int(10e3))

      jaxpr = make_jaxpr(*args, **kwargs).jaxpr

      # Compute flops for all expressions.
      module = Module(name=name)
      _process_jaxpr(
          jaxpr,
          compute_flops,
          scope=_ModuleScope(named_call_id='0'),
          seen=set(),
          module=module)

      _name_scopes_to_modules(module)

      if include_module_info:
        # Add haiku param and state counts for all haiku modules.
        module_infos = make_module_info(*args, **kwargs)
        by_name = {i.module_details.module.module_name: i for i in module_infos}
        by_name = {k.replace('/~/', '/'): v for k, v in by_name.items()}

        for expr in module.expressions:
          submodule = expr.submodule
          if submodule:
            _add_param_counts(submodule, submodule.name, by_name)
    finally:
      sys.setrecursionlimit(old_limit)
    return module

  return make_module


def _name_scopes_to_modules(module: Module):
  """Converts name scopes to nested Modules, as if a call jaxpr were present."""
  expressions = list(module.expressions)
  del module.expressions[:]

  if module.flops:
    # Flops are recomputed below.
    module.flops = 0

  # The nested set of Modules corresponding to the current name scope stack.
  module_stack = [module]

  for e in expressions:
    if e.submodule is not None:
      _name_scopes_to_modules(e.submodule)

    # Close open scopes.
    while len(module_stack) > len(e.name_stack) + 1:
      module_stack.pop()

    for i, n in enumerate(e.name_stack):
      i = i + 1  # Offset to account for the outermost module.
      if i < len(module_stack) and n != module_stack[i].name:
        # Entering a different name scope.
        while len(module_stack) > i:
          module_stack.pop()

      if i == len(module_stack):
        # Represent the scope as a dummy module.
        scope = Expression(primitive='name_scope', invars='', outvars='')
        scope.submodule = Module(name=n)
        module_stack[-1].expressions.append(scope)
        module_stack.append(scope.submodule)

    # Expressions belong to the corresponding innermost module.
    module_stack[-1].expressions.append(e)
    if e.flops:
      for m in module_stack:
        m.flops = (m.flops or 0) + e.flops


class _ModuleScope(NamedTuple):
  """Helper object to track the nesting of named_calls and module scopes."""
  # The concatenation of all outer named_call jaxprs. This is used to uniquely
  # identify which computations we've already seen.
  named_call_id: str

  def join(self, eqn: jax_core.JaxprEqn) -> '_ModuleScope':
    return _ModuleScope(
        named_call_id=self.named_call_id + '/' + str(id(eqn)))


def _format_shape(var):
  return var.aval.str_short().replace('float', 'f')


def _mark_seen(
    binder_idx: dict[jax_core.Var, int],
    seen: set[str],
    var: jax_core.Var,
    scope: _ModuleScope,
) -> bool:
  """Marks a variable as seen. Returns True if it was not previously seen."""
  key = scope.named_call_id + '/' + _var_to_str(binder_idx, var)
  if key in seen:
    return False
  seen.add(key)
  return True


def _var_sort_key(s: str):
  """Sorts variables in order of use by JAX, short names first and `_` last."""
  return (1000 if s == '_' else len(s), s)


def _var_to_str(
    binder_idx: dict[jax_core.Var, int], atom: Atom
) -> str:
  """Returns an atom name based on var binding order in its containing jaxpr."""
  if isinstance(atom, DropVar):
    return '_'
  if isinstance(atom, jax_core.Literal):
    return str(atom)
  assert isinstance(atom, jax_core.Var)
  n = binder_idx[atom]
  s = ''
  while not s or n:
    n, i = n // 26, n % 26
    s = chr(97 + i % 26) + s
  return s


def _process_eqn(
    eqn: jax_core.JaxprEqn,
    seen: set[str],
    eqns_by_output: Mapping[str, jax_core.JaxprEqn],
    compute_flops: ComputeFlopsFn | None,
    scope: _ModuleScope,
    module: Module,
    binder_idx: dict[jax_core.Var, int],
) -> int | None:
  """Recursive walks the JaxprEqn to compute the flops it takes."""
  for out_var in eqn.outvars:
    _mark_seen(binder_idx, seen, out_var, scope)

  outvars = sorted((_var_to_str(binder_idx, e) for e in eqn.outvars),
                   key=_var_sort_key)
  name_stack = str(eqn.source_info.name_stack)
  expression = Expression(
      primitive=eqn.primitive.name,
      invars=' '.join(_var_to_str(binder_idx, v) for v in eqn.invars)
      if len(eqn.invars) < 10 else f'{len(eqn.invars)} inputs',
      outvars=' '.join(outvars) if len(outvars) < 10 else
      f'{outvars[0]}, {len(outvars) - 1} more outputs',
      first_outvar=outvars[0],
      name_stack=name_stack.split('/') if name_stack else [])

  if eqn.primitive.name in [
      'named_call',
      'custom_jvp_call_jaxpr',
      'custom_vjp_call_jaxpr',
      'custom_jvp_call',
      'custom_vjp_call',
      'remat_call',
      'scan',
      'while',
      'xla_call',
      'xla_pmap',
      'pjit',
      'remat2',
      'shard_map',
  ]:
    flops_multiplier = 1
    if eqn.primitive.name in ['named_call', 'xla_call']:
      # Haiku module or named scope.
      name = eqn.params['name']
      jaxpr = eqn.params['call_jaxpr']
    elif eqn.primitive.name in ['remat_call', 'xla_pmap']:
      name = eqn.primitive.name + ' ' + eqn.params['name']
      jaxpr = eqn.params['call_jaxpr']
    elif eqn.primitive.name == 'remat2':
      name = 'remat'
      jaxpr = eqn.params['jaxpr']
    elif eqn.primitive.name == 'while':
      name = 'while'
      jaxpr = eqn.params['body_jaxpr'].jaxpr
      # The loop also has a `cond_jaxpr` that we don't display for now.
    elif eqn.primitive.name == 'scan':
      name = f'scan length={eqn.params["length"]} unroll={eqn.params["unroll"]}'
      jaxpr = eqn.params['jaxpr'].jaxpr
      flops_multiplier = eqn.params['length'] // eqn.params['unroll']
    elif eqn.primitive.name == 'pjit':
      name = eqn.params['name']
      jaxpr = eqn.params['jaxpr'].jaxpr
    elif eqn.primitive.name == 'shard_map':
      name = 'shard_map'
      jaxpr = eqn.params['jaxpr']
      flops_multiplier = eqn.params['mesh'].size
    elif eqn.primitive.name in ['custom_jvp_call', 'custom_vjp_call']:
      name = eqn.primitive.name.replace('_call', '')
      jaxpr = eqn.params['call_jaxpr']
    elif eqn.primitive.name in [
        'custom_jvp_call_jaxpr', 'custom_vjp_call_jaxpr'
    ]:
      # Custom gradient.
      name = eqn.primitive.name.replace('_call_jaxpr', '')
      jaxpr = eqn.params['fun_jaxpr'].jaxpr
    else:
      raise ValueError(f'unmatched eqn {eqn.primitive.name}')

    expression.submodule = Module(name=name)
    flops = _process_jaxpr(jaxpr, compute_flops, scope.join(eqn), seen,
                           expression.submodule)
    if compute_flops is not None:
      flops *= flops_multiplier

    expression.submodule.flops = flops
    expression.flops = flops
    if compute_flops is None or flops > 0:
      module.expressions.append(expression)
  else:
    details = []
    if eqn.invars:
      details.append('in ' + ', '.join(_format_shape(v) for v in eqn.invars))
    if eqn.outvars:
      details.append('out ' + ', '.join(_format_shape(v) for v in eqn.outvars))

    expression.details = ', '.join(details)
    expression.params.update({k: str(v) for k, v in eqn.params.items()})

    flops = None if compute_flops is None else compute_flops(eqn, expression)
    expression.flops = flops
    module.expressions.append(expression)

  for var in eqn.invars:
    if isinstance(var, jax_core.Literal):
      continue

    key = _var_to_str(binder_idx, var)
    if key == '*':
      continue

    if not _mark_seen(binder_idx, seen, var, scope):
      continue

    if key not in eqns_by_output:
      logging.warning('missing var %s = %s', key, type(var))
      continue
    f = _process_eqn(eqns_by_output[key], seen, eqns_by_output, compute_flops,
                     scope, module, binder_idx)
    if compute_flops is not None:
      flops += f

  return flops


def _process_jaxpr(
    jaxpr: jax_core.Jaxpr,
    compute_flops: ComputeFlopsFn | None,
    scope: _ModuleScope,
    seen: set[str],
    module: Module,
) -> int | None:
  """Computes the flops used for a JAX expression, tracking module scope."""
  if isinstance(jaxpr, jax_core.ClosedJaxpr):
    return _process_jaxpr(jaxpr.jaxpr, compute_flops, scope, seen, module)

  # Label variables by the order in which they're introduced.
  lam_binders = itertools.chain(jaxpr.constvars, jaxpr.invars)
  let_binders = itertools.chain.from_iterable(e.outvars for e in jaxpr.eqns)
  all_binders = itertools.chain(lam_binders, let_binders)
  binder_idx = dict(zip(all_binders, itertools.count()))

  # Build a map that for each output contains the equation to compute it.
  eqns_by_output = {}
  for eqn in jaxpr.eqns:
    for var in eqn.outvars:
      eqns_by_output[_var_to_str(binder_idx, var)] = eqn

  # Seed the set of variables that have been seen with the inputs and constants.
  for var in jaxpr.invars + jaxpr.constvars:
    _mark_seen(binder_idx, seen, var, scope)

  # Recursively walk the computation graph.
  flops = None if compute_flops is None else 0
  for var in jaxpr.outvars:
    if (isinstance(var, jax_core.Var) and
        _mark_seen(binder_idx, seen, var, scope)):
      f = _process_eqn(eqns_by_output[_var_to_str(binder_idx, var)], seen,
                       eqns_by_output, compute_flops, scope, module, binder_idx)
      if compute_flops is not None:
        flops += f

  module.expressions.sort(key=lambda e: _var_sort_key(e.first_outvar))
  module.flops = flops
  return flops


def _add_param_counts(model_info: Module, prefix: str,
                      invocations: Mapping[str, summarise.MethodInvocation]):
  """Adds parameter count to the module info, if present."""
  if prefix in invocations:
    invocation = invocations[prefix]
    model_info.total_param_size = sum(
        spec.size for spec in invocation.module_details.params.values())
    model_info.total_state_size = sum(
        spec.size for spec in invocation.module_details.state.values())
    for k, s in invocation.module_details.params.items():
      model_info.param_info[k.replace('/~/', '/').replace(prefix, '.')] = str(s)
    for k, s in invocation.module_details.state.items():
      model_info.state_info[k.replace('/~/', '/').replace(prefix, '.')] = str(s)

  for expr in model_info.expressions:
    submodule = expr.submodule
    if submodule:
      _add_param_counts(
          submodule, os.path.join(prefix, submodule.name), invocations
      )


def _decimal_prefix(n: int, unit: str, precision: int = 4):
  prefixes = [('E', 18), ('P', 15), ('T', 12), ('G', 9), ('M', 6), ('k', 3)]
  for prefix, exponent in prefixes:
    scale = 10**exponent
    if n > scale:
      return f'%.{precision}g {prefix}{unit}' % (n / scale)
  return f'{n} {unit}'


def format_module(module: Module, depth: int = 0) -> str:
  """Recursively formats module information as a human readable string."""
  s = '  ' * depth + module.name
  if module.flops is not None:
    s += ' ' + _decimal_prefix(module.flops, 'flops')
  if module.total_param_size > 0:
    s += ' ' + _decimal_prefix(module.total_param_size, 'params')
  if module.total_state_size > 0:
    s += ' ' + _decimal_prefix(module.total_state_size, 'state')
  s += '\n'
  for exp in module.expressions:
    s += format_expression(exp, depth + 1)
  return s


def format_expression(exp: Expression, depth: int) -> str:
  if exp.submodule:
    return format_module(exp.submodule, depth)
  s = '  ' * depth + exp.primitive
  if exp.flops is not None:
    s += ' ' + _decimal_prefix(exp.flops, 'flops')
  s += ' ' + exp.details
  return s + '\n'


def as_html_page(module: Module, min_flop: int = 1000) -> str:
  """Formats the output of `make_model_info` as an interactive HTML page."""
  return f"""<html>
    <head>
    <style>{css()}</style>
    <script>{js()}</script>
    </head>
    <body>{as_html(module, min_flop=min_flop)}</body>
    </html>"""


def css() -> str:
  """The CSS for HTML visualization of a `Module`."""
  return """.node {
  list-style-type:none;
}
.node ol {
  padding-left: 0;
  margin-left: 30px;
}

li.node {
  border-left: 1px solid black;
}
li.node:last-child {
  border-left: none;
}
/* Correctly align the elbow connector for the last child node. */
li.node:last-child > span:first-child {
  margin-left: -5px;
}
.expander {
  cursor: pointer;
  font-family: monospace;
}

.expression {
  padding-left: 25px;
}

.tooltip-container .tooltip {
  visibility: hidden;
  background-color: #e3eaff;
  padding: 8px;
  margin-top: 1.4em;
  border-radius: 6px;
  position: absolute;
  z-index: 1;
}

.tooltip-container:hover .tooltip {
  visibility: visible;
}

.model-info {
  font-family: monospace;
}

.flops {
  color: grey;
}

.primitive {
  color: #3f48b0;
}
"""


def js() -> str:
  """The JavaScript for HTML visualization of a `Module`."""
  return """
function expand(el) {
    console.log(el);
    const siblings = el.parentElement.children;
    siblings[0].onclick = () => collapse(el);
    // Expand list of children.
    siblings[siblings.length - 1].style.display = 'block';
    // Hide expander link.
    siblings[siblings.length - 2].style.display = 'none';
}
function collapse(el) {
    console.log(el);
    const siblings = el.parentElement.children;
    siblings[0].onclick = () => expand(el);
    // Hide list of children.
    siblings[siblings.length - 1].style.display = 'none';
    // Show expander link.
    siblings[siblings.length - 2].style.display = 'inline';
}
    """


def as_html(module: Module,
            min_flop: int = 1000,
            outvars: str = '',
            last: bool = False) -> str:
  """Formats a `Module` as a tree of interactive HTML elements.

  When embedding this in a page, the outputs of `css` and `js` must be embedded
  too for the visualization to work.
  To only visualize a single module directly, see `as_html_page`.

  Args:
    module: The module to visualize, as an interactive HTML tree.
    min_flop: Minimum number of flops for an operation to be shown.
    outvars: For internal use, the outputs of this module.
    last: For internal use, whether this module is the last of its siblings.

  Returns:
    HTML representation of `module`.
  """
  s = '<span>'

  if last:
    s += """ <span class = "expander"
                    onclick = "javascript:expand(this)"
                    style = "padding-right: 6px" >
                    &#9492;&#9472;
                </span>"""
  else:
    s += """<span class="expander"
                    onclick="javascript:expand(this)">
                    &#9472;&#9472;
                </span>"""

  s += f"""<span class="tooltip-container"> {outvars} = <b>{module.name}</b>"""
  if module.flops is not None:
    s += f"<span class='flops'>{_decimal_prefix(module.flops, 'flop')}</span>"

  if module.total_param_size > 0:
    s += f"<span class='flops'>{_decimal_prefix(module.total_param_size, 'param')}</span>"
  if module.total_state_size > 0:
    s += f"<span class='flops'>{_decimal_prefix(module.total_state_size, 'state')}</span>"

  # Tooltip with detailed param and state information.
  s += """<span class="tooltip">
      <table>
        <tr><th colspan=2>Param</th></tr>"""
  for key, val in module.param_info.items():
    s += f'<tr><td>{key}</td><td>{val}</td></tr>'
  s += '<tr><th colspan=2>State</th></tr>'
  for key, val in module.state_info.items():
    s += f'<tr><td>{key}</td><td>{val}</td></tr>'
  s += """
      </table>
    </span>
    </span>"""

  expressions = [
      e for e in module.expressions if e.flops is None or e.flops >= min_flop
  ]
  if expressions:
    s += """
        <span class="expander" onclick="javascript:expand(this)">
            [expand]
        </span>
        <ol style="display:none">
        """

    for i, exp in enumerate(expressions):
      last = i + 1 == len(expressions)
      s += "<li class='node'>"
      if exp.submodule:
        s += as_html(
            exp.submodule, min_flop=min_flop, outvars=exp.outvars, last=last)
      else:
        s += f"""<div class='expression tooltip-container'>
                            <span class='tooltip'>{exp.details}"""
        for key, val in exp.params.items():
          s += f'<div>{key}: {val}</div>'
        s += f"""</span>
                        {exp.outvars} = <span class='primitive'>{exp.primitive}</span>
                        {exp.invars}
                        <span class='flops'>{_decimal_prefix((exp.flops or 0), 'flops')}</span>
                    </div>"""
      s += '</li>'

    s += '</ol>'

  return s + '</span>'
