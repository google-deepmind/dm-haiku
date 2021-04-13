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
"""Summarises Haiku modules."""

import functools
import pprint
import types
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import dataclasses
from haiku._src import base
from haiku._src import data_structures
from haiku._src import module as module_lib
from haiku._src import transform
from haiku._src import utils
import jax
import jax.numpy as jnp
import numpy as np
import tabulate as tabulate_lib

# If you are forking replace this block with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.experimental = types.ModuleType("experimental")
hk.Module = module_lib.Module
hk.experimental.MethodContext = module_lib.MethodContext
hk.experimental.intercept_methods = module_lib.intercept_methods
hk.transform_with_state = transform.transform_with_state
hk.Transformed = transform.Transformed
hk.TransformedWithState = transform.TransformedWithState
del module_lib

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class ArraySpec:
  """Shaped and sized specification of an array.

  Attributes:
    shape: Shape of the array.
    dtype: DType of the array.
  """
  shape: Sequence[int]
  dtype: jnp.dtype

  # Used so pformat(..) formats spec as: f32[1,2,3]
  __repr__ = __str__ = utils.format_array

  @property
  def size(self):  # pylint: disable=g-missing-from-attributes
    return int(np.prod(self.shape))

  @classmethod
  def from_array(cls, array) -> "ArraySpec":
    return ArraySpec(array.shape, array.dtype)


@dataclasses.dataclass(frozen=True)
class ModuleDetails:
  """Module and method related information.

  Attributes:
    module: A :class:`~haiku.Module` instance.
    method_name: The method name that was invoked on the module.
    params: The modules params dict with arrays converted to :class:`ArraySpec`.
    state: The modules state dict with arrays converted to :class:`ArraySpec`.
  """

  module: hk.Module
  method_name: str
  params: Mapping[str, ArraySpec]
  state: Mapping[str, ArraySpec]

  @classmethod
  def of(cls, module: hk.Module, method_name: str) -> "ModuleDetails":
    params = jax.tree_map(ArraySpec.from_array, module.params_dict())
    state = jax.tree_map(ArraySpec.from_array, module.state_dict())
    return ModuleDetails(module=module, method_name=method_name, params=params,
                         state=state)


@dataclasses.dataclass(frozen=True)
class MethodInvocation:
  """Record of a method being invoked on a given module.

  Attributes:
    module_details: Details about which module and method were invoked.
    args_spec: Positional arguments to the method invocation with arrays
      replaced by :class:`ArraySpec`.
    kwargs_spec: Keyword arguments to the method invocation with arrays
      replaced by :class:`ArraySpec`.
    output_spec: Output of the method invocation with arrays replaced by
      :class:`ArraySpec`.
    context: Additional context information for the method call as provided
      by :func:`~haiku.experimental.intercept_methods`.
    call_stack: Stack of modules currently active while calling this module
      method. For example if ``A`` calls ``B`` which calls ``C`` then the call
      stack for ``C`` will be ``[B_DETAILS, A_DETAILS]``.
  """

  module_details: ModuleDetails
  args_spec: Tuple[Any, ...]
  kwargs_spec: Dict[str, Any]
  output_spec: Any  # Actual: PyTree[Union[Any, ArraySpec]]
  context: hk.experimental.MethodContext
  call_stack: Sequence[ModuleDetails]


def get_call_stack() -> Sequence[ModuleDetails]:
  frame = base.current_frame()
  return tuple(map(
      lambda s: ModuleDetails.of(s.module, s.method_name),
      list(frame.module_stack)))


def to_spec(tree):
  return jax.tree_map(
      lambda x: ArraySpec.from_array(x) if isinstance(x, jnp.ndarray) else x,
      tree)

IGNORED_METHODS = ("__init__", "params_dict", "state_dict")


def log_used_modules(
    used_modules: List[MethodInvocation],
    next_f: Callable[..., T],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    context: hk.experimental.MethodContext,
) -> T:
  """Method interceptor that logs used modules to the given list."""
  if context.method_name in IGNORED_METHODS:
    return next_f(*args, **kwargs)

  idx = len(used_modules)
  used_modules.append(None)  # pytype: disable=container-type-mismatch
  out = next_f(*args, **kwargs)
  used_modules[idx] = MethodInvocation(
      module_details=ModuleDetails.of(context.module, context.method_name),
      args_spec=to_spec(args),
      kwargs_spec=to_spec(kwargs),
      output_spec=to_spec(out),
      context=context,
      call_stack=get_call_stack())
  return out


def eval_summary(
    f: Union[Callable[..., Any], hk.Transformed, hk.TransformedWithState],
) -> Callable[..., Sequence[MethodInvocation]]:
  """Records module method calls performed by ``f``.

  >>> f = lambda x: hk.nets.MLP([300, 100, 10])(x)
  >>> x = jnp.ones([8, 28 * 28])
  >>> for i in hk.experimental.eval_summary(f)(x):
  ...   print("mod := {:14} | in := {} out := {}".format(
  ...       i.module_details.module.module_name, i.args_spec[0], i.output_spec))
  mod := mlp            | in := f32[8,784] out := f32[8,10]
  mod := mlp/~/linear_0 | in := f32[8,784] out := f32[8,300]
  mod := mlp/~/linear_1 | in := f32[8,300] out := f32[8,100]
  mod := mlp/~/linear_2 | in := f32[8,100] out := f32[8,10]

  Args:
    f: A function or transformed function to trace.

  Returns:
    A callable taking the same arguments as the provided function, but returning
    a sequence of :class:`MethodInvocation` instances revealing the methods
    called on each module when applying ``f``.

  See Also:
    :func:`tabulate`: Pretty prints a summary of the execution of a function.
  """
  sidechannel = data_structures.ThreadLocalStack()

  try:
    f = transform.get_original_fn(f)
  except AttributeError:
    pass

  def f_logged(*args, **kwargs):
    used_modules = sidechannel.peek()
    logging_interceptor = functools.partial(log_used_modules, used_modules)

    with hk.experimental.intercept_methods(logging_interceptor):
      f(*args, **kwargs)

  f_orig = hk.transform_with_state(f)
  f_logged = hk.transform_with_state(f_logged)

  def init_apply(*args, **kwargs):
    rng = jax.random.PRNGKey(42)
    params, state = f_orig.init(rng, *args, **kwargs)
    f_logged.apply(params, state, rng, *args, **kwargs)

  def wrapper(*args, **kwargs) -> Sequence[MethodInvocation]:
    used_modules = []
    with sidechannel(used_modules):
      jax.eval_shape(init_apply, *args, **kwargs)
    return used_modules

  return wrapper


# Support for writing out tables.


@dataclasses.dataclass(frozen=True)
class Filter:
  f: Callable[[MethodInvocation], bool]


@dataclasses.dataclass(frozen=True)
class Column:
  header: str
  f: Callable[[MethodInvocation], str]
  align: str = "left"


def owned_params(module: ModuleDetails) -> Mapping[str, jnp.ndarray]:
  out = {}
  for fq_name, param in module.params.items():
    module_name, param_name = fq_name.rsplit("/", 1)
    if module_name == module.module.module_name:
      out[param_name] = param
  return out


def format_owned_params(invocation: MethodInvocation) -> str:
  """Newline separated params sorted by size (desc) then name (asc)."""
  params = owned_params(invocation.module_details)
  size_desc_then_name = lambda i: (-np.prod(i[1].shape), i[0])
  return "\n".join(f"{k}: {utils.format_array(v)}"
                   for k, v in sorted(params.items(), key=size_desc_then_name))


def format_input(invocation: MethodInvocation) -> str:
  out = ", ".join(map(pprint.pformat, invocation.args_spec))
  kwargs = invocation.kwargs_spec
  if kwargs:
    out += "\n"
    out += ", ".join(f"{k}={pprint.pformat(v)}" for k, v in kwargs.items())
  return out


def format_output(invocation: MethodInvocation) -> str:
  return pprint.pformat(invocation.output_spec)


def format_call_stack(invocation: MethodInvocation) -> str:
  """Formats the name and call_stack of a given module as a string."""
  def format_entry(state: ModuleDetails) -> str:
    class_name = type(state.module).__name__
    module_name = state.module.module_name
    method_name = state.method_name
    if method_name == "__call__":
      return f"{module_name} ({class_name})"
    else:
      return f"{module_name} ({class_name}.{method_name})"

  return "\n └ ".join(map(format_entry, invocation.call_stack))


all_filters = {
    "has_output": Filter(lambda r: r.output_spec is not None),
    "has_params": Filter(lambda r: bool(r.module_details.params)),
}

all_columns = {
    "module": Column("Module", format_call_stack),
    "config": Column("Config", lambda r: repr(r.module_details.module)),
    "owned_params": Column("Module params", format_owned_params),
    "input": Column("Input", format_input),
    "output": Column("Output", format_output),
    "params_size": Column(
        "Param count",
        lambda r: "{:,}".format(utils.tree_size(r.module_details.params)),
        "right"),
    "params_bytes": Column(
        "Param bytes",
        lambda r: utils.format_bytes(utils.tree_bytes(r.module_details.params)),
        "right"),
}

DEFAULT_COLUMNS = ("module", "config", "owned_params", "input", "output",
                   "params_size", "params_bytes")
DEFAULT_FILTERS = ("has_output",)


def tabulate(
    f: Union[Callable[..., Any], hk.Transformed, hk.TransformedWithState],
    *,
    columns: Optional[Sequence[str]] = DEFAULT_COLUMNS,
    filters: Optional[Sequence[str]] = DEFAULT_FILTERS,
    tabulate_kwargs={"tablefmt": "grid"},
) -> Callable[..., str]:
  # pylint: disable=line-too-long
  """Produces a summarised view of the execution of ``f``.

  >>> def f(x):
  ...   return hk.nets.MLP([300, 100, 10])(x)
  >>> x = jnp.ones([8, 28 * 28])
  >>> f = hk.transform(f)
  >>> print(hk.experimental.tabulate(f)(x))
  +-------------------------+------------------------------------------+-----------------+------------+------------+---------------+---------------+
  | Module                  | Config                                   | Module params   | Input      | Output     |   Param count |   Param bytes |
  +=========================+==========================================+=================+============+============+===============+===============+
  | mlp (MLP)               | MLP(output_sizes=[300, 100, 10])         |                 | f32[8,784] | f32[8,10]  |       266,610 |       1.07 MB |
  +-------------------------+------------------------------------------+-----------------+------------+------------+---------------+---------------+
  | mlp/~/linear_0 (Linear) | Linear(output_size=300, name='linear_0') | w: f32[784,300] | f32[8,784] | f32[8,300] |       235,500 |     942.00 KB |
  |  └ mlp (MLP)            |                                          | b: f32[300]     |            |            |               |               |
  +-------------------------+------------------------------------------+-----------------+------------+------------+---------------+---------------+
  | mlp/~/linear_1 (Linear) | Linear(output_size=100, name='linear_1') | w: f32[300,100] | f32[8,300] | f32[8,100] |        30,100 |     120.40 KB |
  |  └ mlp (MLP)            |                                          | b: f32[100]     |            |            |               |               |
  +-------------------------+------------------------------------------+-----------------+------------+------------+---------------+---------------+
  | mlp/~/linear_2 (Linear) | Linear(output_size=10, name='linear_2')  | w: f32[100,10]  | f32[8,100] | f32[8,10]  |         1,010 |       4.04 KB |
  |  └ mlp (MLP)            |                                          | b: f32[10]      |            |            |               |               |
  +-------------------------+------------------------------------------+-----------------+------------+------------+---------------+---------------+

  Possible values for ``columns``:

  * ``module``: Displays module and method name.
  * ``config``: Displays the constructor arguments used for the module.
  * ``owned_params``: Displays parameters directly owned by this module.
  * ``input``: Displays module inputs.
  * ``output``: Displays module output.
  * ``params_size``: Displays the number of parameters
  * ``params_bytes``: Displays parameter size in bytes.

  Possible values for ``filters``:

  * ``has_output``: Only include methods returning a value other than ``None``.
  * ``has_params``: Removes methods from modules that do not have parameters.

  Args:
    f: A function to transform OR one of the init/apply functions from Haiku
      or the result of :func:`transform` or :func:`transform_with_state`.
    columns: A list of column names to enable.
    filters: A list of filters to apply to remove certain module methods.
    tabulate_kwargs: Keyword arguments to pass to ``tabulate.tabulate(..)``.

  Returns:
    A callable that takes the same arguments as ``f`` but returns a string
    summarising the modules used during the execution of ``f``.

  See Also:
    :func:`eval_summary`: Raw data used to generate this table.
  """
  # pylint: enable=line-too-long
  f = eval_summary(f)

  if columns is None:
    columns = DEFAULT_COLUMNS
  else:
    invalid = [c for c in columns if c not in all_columns]
    if invalid:
      raise ValueError(
          f"Invalid column(s) {invalid}, valid columns {list(all_columns)}")

  if filters is None:
    filters = DEFAULT_FILTERS
  else:
    invalid = [f for f in filters if f not in all_filters]
    if invalid:
      raise ValueError(
          f"Invalid filter(s) {invalid}, valid filters {list(all_filters)}")

  columns = [all_columns[c] for c in columns]
  filters = [all_filters[f] for f in filters]

  def generate_summary(*args, **kwargs) -> str:
    """Generates a string summary of the given Haiku function."""
    rows = []
    for method_invocation in f(*args, **kwargs):
      if not all(filter.f(method_invocation) for filter in filters):
        continue

      row = [column.f(method_invocation) for column in columns]
      rows.append(row)

    if rows:
      headers = [col.header for col in columns]
      colalign = [col.align for col in columns]
      return tabulate_lib.tabulate(
          rows, headers=headers, colalign=colalign, **tabulate_kwargs)
    else:
      return "No modules matching filters."

  return generate_summary
