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
"""Provides a way to add a named_call to a Jaxpr."""
import functools
from typing import Any, Callable, Sequence, Optional, Tuple

from haiku._src import base
from haiku._src import stateful

import jax
from jax import api
from jax import core
from jax.config import config
from jax.interpreters import ad
from jax.interpreters import xla
import jax.linear_util as lu

xc = jax.lib.xla_client
xe = xc._xla  # pylint: disable=protected-access

# Registering named call as a primitive
named_call_p = core.CallPrimitive('named_call')
# named_call is implemented as a plain core.call and only diverges
# under compilation (see named_call_translation_rule)
named_call_p.def_impl(core.call_impl)


def _named_call_translation_rule(
    comp_builder: xe.XlaBuilder,
    axis_env: xla.AxisEnv,
    in_nodes: Sequence[xe.XlaOp],
    name_stack: str,
    backend: Optional[Any],
    name: str,
    call_jaxpr: core.Jaxpr,
) -> xe.XlaOp:
  """Compile and add a custom name to the XLA metadata."""
  subcomp_builder = xla.xb.make_computation_builder(f'named_call_{name}')
  args = [xla.xb.parameter(subcomp_builder, i, comp_builder.GetShape(n))
          for i, n in enumerate(in_nodes)]
  out_nodes = xla.jaxpr_subcomp(subcomp_builder, call_jaxpr,
                                backend, axis_env, (),
                                jax.util.extend_name_stack(name_stack, name),
                                *args)
  subcomp = subcomp_builder.Build(xla.xops.Tuple(subcomp_builder, out_nodes))
  return xla.xops.Call(comp_builder, subcomp, list(in_nodes))

ad.primitive_transposes[named_call_p] = functools.partial(ad.call_transpose,
                                                          named_call_p)
xla.call_translations[named_call_p] = _named_call_translation_rule


def statefulify(
    fun: Callable[..., Any],
    state: stateful.InternalState,
) -> Callable[..., Any]:
  """Wraps the given function so it is evaluated with the given Haiku state."""
  @functools.wraps(fun)
  def stateful_fun(*args, **kwargs) -> Tuple[Any, stateful.InternalState]:
    """Explictly returns the changed Haiku state after fun has been executed."""
    with stateful.temporary_internal_state(state):
      out = fun(*args, **kwargs)
      return out, stateful.difference(state, stateful.internal_state())
  return stateful_fun


def stateful_named_call(
    fun: Callable[..., Any],
    *,
    name: Optional[str] = None,
) -> Callable[..., Any]:
  """Wraps a function in a name_scope and maintains Haiku state."""
  @functools.wraps(fun)
  def wrapper(*args, **kwargs):
    if base.inside_transform():
      # fun might be stateful, in which case we need to explicitly thread
      # state in and out of fun to preserve fun as functionally pure.
      state = stateful.internal_state()
      named_f = _named_call(statefulify(fun, state), name=name)
      out, state = named_f(*args, **kwargs)
      stateful.update_internal_state(state)
    else:
      out = _named_call(fun, name=name)(*args, **kwargs)
    return out
  return wrapper


def _named_call(
    fun: Callable[..., Any],
    *,
    name: Optional[str] = None,
) -> Callable[..., Any]:
  """Wraps a function in a name_scope with the provided name."""
  if name is None:
    name = fun.__name__

  def named_fun(*args, **kwargs):
    # Wrap and flatten f for JAX internals.
    f = lu.wrap_init(fun)
    flat_args, in_tree = jax.tree_flatten((args, kwargs))
    flat_f, out_tree = api.flatten_fun(f, in_tree)

    if config.omnistaging_enabled:
      # Avoid abstracting inputs by calling as a thunk
      f_thunk = lu.wrap_init(lambda: flat_f.call_wrapped(*flat_args),)
      out_flat = named_call_p.bind(f_thunk, name=name)
    else:
      # Hide any args that are not a valid JaxType by partially applying flat_f
      dyn_argnums = [i for (i, x) in enumerate(flat_args)
                     if jax.api._valid_jaxtype(x)]  # pylint: disable=protected-access
      part_f, dyn_args = jax.argnums_partial(flat_f, dyn_argnums, flat_args)

      # Call with a custom XLA subcomputation via named_call & unflatten result.
      out_flat = named_call_p.bind(part_f, *dyn_args, name=name)

    return jax.tree_unflatten(out_tree(), out_flat)
  return named_fun
