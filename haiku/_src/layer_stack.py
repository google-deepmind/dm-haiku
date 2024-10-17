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
"""Function to stack repeats of a layer function without shared parameters."""

import collections
from collections.abc import Callable
import functools
import inspect
from typing import Any, Protocol, Union

from haiku._src import base
from haiku._src import lift
from haiku._src import module
from haiku._src import transform
import jax
import jax.numpy as jnp


LayerStackCarry = collections.namedtuple("LayerStackCarry", ["x"])
LayerStackScanned = collections.namedtuple(
    "LayerStackScanned", ["params", "rng", "state", "args_ys"])

# WrappedFn should take in arbitrarily nested `jax.Array`, and return the
# exact same type. We cannot express this with `typing`. So we just use it
# to inform the user. In reality, the typing below will accept anything.
NestedArray = Any
WrappedFn = Callable[..., Union[NestedArray, tuple[NestedArray]]]


def _check_no_varargs(f):
  if list(inspect.signature(
      f).parameters.values())[0].kind == inspect.Parameter.VAR_POSITIONAL:
    raise ValueError(
        "The function `f` should not have any `varargs` (that is *args) "
        "argument. Instead, it should only use explicit positional"
        "arguments")


def _get_rng_stack(count: int) -> jax.Array | None:
  rng = base.maybe_next_rng_key()
  if rng is None:
    return None
  return jax.random.split(rng, count)


class LayerStackTransparencyMapping(Protocol):
  """Module name mapping for transparent layer_stack."""

  def stacked_to_flat(self, stacked_module_name: str, scan_idx: int) -> str:
    """Creates flat module name from stacked name and index during scan."""
    ...

  def flat_to_stacked(
      self, unstacked_module_name: str
  ) -> tuple[str, int] | None:
    """Creates stacked module name and scan index from flat name.

    Returns None when the module is not a part of layer_stack.  This happens
    when the caller module transparently calling layer_stack has its own
    parameters.  This function is basically inverse of `stacked_to_flat`,

    Args:
      unstacked_module_name: Name of the module to be converted to stacked.

    Returns:
      Name and layer index of the module when stacked.  None if the module is
      not part of the stack.
    """
    ...


def _split_params(
    stacked_params: base.Params,
    num_layers: int,
    name_map: LayerStackTransparencyMapping,
) -> base.Params:
  """Splits the stacked parameters."""
  params = {}
  for mod_name, mod_params in stacked_params.items():
    for i in range(num_layers):
      new_mod_name = name_map.stacked_to_flat(mod_name, i)
      if new_mod_name in params:
        raise ValueError(
            f"Found conflicting unstacked module name for {mod_name} at"
            f" {new_mod_name}."
        )
      params[new_mod_name] = jax.tree.map(lambda x: x[i], mod_params)  # pylint:disable=cell-var-from-loop
  return params


def _stack_params(
    split_params: base.Params,
    num_layers: int,
    name_map: LayerStackTransparencyMapping,
) -> base.Params:
  """Stacks the split parameters."""
  # Construct a separate tree for each loop iteration, which we will then
  # multimap over in a call to jnp.stack. This formulation preserves custom
  # pytree node types.
  param_trees = [{} for _ in range(num_layers)]
  for mod_name, mod_params in split_params.items():
    stacked_name_idx = name_map.flat_to_stacked(mod_name)
    # If the transparency map returns None, this param is not part of the stack.
    if stacked_name_idx is None:
      continue
    stacked_mod_name, idx = stacked_name_idx
    if stacked_mod_name not in param_trees[idx]:
      param_trees[idx][stacked_mod_name] = {}
    for k, v in mod_params.items():
      if k in param_trees[idx][stacked_mod_name]:
        raise ValueError(
            f"Found conflicting values for param {stacked_mod_name}/{k} at"
            f" index {idx}."
        )
      param_trees[idx][stacked_mod_name][k] = v

  return jax.tree.map(lambda *args: jnp.stack(args, axis=0), *param_trees)


class _LayerStack:
  """Module to compose parameterized functions, implemented as a scan."""

  def __init__(
      self,
      count: int,
      unroll: int,
      pass_reverse_to_layer_fn: bool = False,
      transparency_map: LayerStackTransparencyMapping | None = None,
      name: str = "",
  ):
    """Iterate f count times, with non-shared parameters."""
    self._name = name
    self._count = count
    self._unroll = unroll
    self._pass_reverse_to_layer_fn = pass_reverse_to_layer_fn
    self._transparency_map = transparency_map

  def __call__(self, x, *args_ys, reverse=False):
    count = self._count
    init_fn, apply_fn = transform.transform_with_state(self._call_wrapped)

    def per_layer_init_fn(c, a):
      c, rng = c
      if rng is not None:
        rng, next_rng, apply_rng = jax.random.split(rng, 3)
      else:
        rng, next_rng, apply_rng = None, None, None
      params, state = init_fn(rng, c, *a)
      (c, _), state = apply_fn(params, state, apply_rng, c, *a)
      return (c, next_rng), (params, state)

    def scanned_init_fn(x, rng):
      _, (params, state) = jax.lax.scan(per_layer_init_fn, (x, rng), args_ys,
                                        length=self._count)
      if self._transparency_map is not None:
        return (_split_params(params, self._count, self._transparency_map),
                _split_params(state, self._count, self._transparency_map))
      return params, state

    rng = base.maybe_next_rng_key()

    if self._transparency_map is not None:
      params_and_state_fn, updater = lift.transparent_lift_with_state(
          scanned_init_fn, allow_reuse=True
      )
    else:
      params_and_state_fn, updater = lift.lift_with_state(
          scanned_init_fn, allow_reuse=True, name=self._name
      )
    params, state = params_and_state_fn(x, rng)

    # Use scan during apply, threading through random seed so that it's
    # unique for each layer.
    def layer(
        carry: LayerStackCarry, scanned: LayerStackScanned
    ) -> tuple[LayerStackCarry, Any]:
      rng = scanned.rng
      params = scanned.params
      state = scanned.state

      kwargs = {}
      if self._pass_reverse_to_layer_fn:
        kwargs["reverse"] = reverse
      (out_x, z), state = apply_fn(
          params, state, rng, carry.x, *scanned.args_ys, **kwargs)
      return LayerStackCarry(x=out_x), (z, state)

    rng = _get_rng_stack(count)

    if self._transparency_map is not None:
      params = _stack_params(params, self._count, self._transparency_map)
      state = _stack_params(state, self._count, self._transparency_map)

    carry = LayerStackCarry(x=x)
    scanned = LayerStackScanned(params=params,
                                state=state,
                                rng=rng,
                                args_ys=args_ys)

    carry, (zs, states) = jax.lax.scan(
        layer, carry, scanned, length=count, unroll=self._unroll,
        reverse=reverse)
    updater.update(states)
    return carry.x, zs

  def _call_wrapped(
      self,
      x: jax.Array,
      *args,
  ) -> tuple[jax.Array, jax.Array | None]:
    raise NotImplementedError()


class _LayerStackNoPerLayer(_LayerStack):
  """_LayerStack impl with no per-layer inputs provided to the function."""

  def __init__(
      self,
      f: WrappedFn,
      count: int,
      unroll: int,
      pass_reverse_to_layer_fn: bool = False,
      transparency_map: LayerStackTransparencyMapping | None = None,
      name: str = "",
  ):
    super().__init__(
        count=count,
        unroll=unroll,
        pass_reverse_to_layer_fn=pass_reverse_to_layer_fn,
        transparency_map=transparency_map,
        name=name,
    )
    _check_no_varargs(f)
    self._f = f

  @module.transparent
  def _call_wrapped(self, x, **kwargs):
    ret = self._f(*x, **kwargs)
    if len(x) == 1:
      # If the function takes a single argument, the wrapped function receives
      # a tuple of length 1, and therefore it must return a tuple of length 1.
      ret = (ret,)
    return ret, None


class _LayerStackWithPerLayer(_LayerStack):
  """_LayerStack impl with per-layer inputs provided to the function."""

  def __init__(
      self,
      f: WrappedFn,
      count: int,
      unroll: int,
      pass_reverse_to_layer_fn: bool = False,
      transparency_map: LayerStackTransparencyMapping | None = None,
      name: str = "",
  ):
    super().__init__(
        count=count,
        unroll=unroll,
        pass_reverse_to_layer_fn=pass_reverse_to_layer_fn,
        transparency_map=transparency_map,
        name=name,
    )
    self._f = f

  @module.transparent
  def _call_wrapped(self, x, *args, **kwargs):
    return self._f(x, *args, **kwargs)


def layer_stack(
    num_layers: int,
    with_per_layer_inputs=False,
    unroll: int = 1,
    pass_reverse_to_layer_fn: bool = False,
    transparent: bool = False,
    transparency_map: LayerStackTransparencyMapping | None = None,
    name: str | None = None,
):
  """Utility to wrap a Haiku function and recursively apply it to an input.

  This can be used to improve model compile times.

  A function is valid if it uses only explicit position parameters, and
  its return type matches its input type. The position parameters can be
  arbitrarily nested structures with ``jax.Array`` at the leaf nodes. Note
  that kwargs are not supported, neither are functions with variable number
  of parameters (specified by ``*args``).

  If ``with_per_layer_inputs=False`` then the new, wrapped function can be
  understood as performing the following:

  >>> f = lambda x: x+1
  >>> num_layers = 4
  >>> x = 0
  >>> for i in range(num_layers):
  ...   x = f(x)
  >>> x
  4

  And if ``with_per_layer_inputs=True``, assuming ``f`` takes two arguments on
  top of ``x``:

  >>> f = lambda x, y0, y1: (x+1, y0+y1)
  >>> num_layers = 4
  >>> x = 0
  >>> ys_0 = [1, 2, 3, 4]
  >>> ys_1 = [5, 6, 7, 8]
  >>> zs = []
  >>> for i in range(num_layers):
  ...   x, z = f(x, ys_0[i], ys_1[i])
  ...   zs.append(z)
  >>> x, zs
  (4, [6, 8, 10, 12])

  The code using ``layer_stack`` for the above function would be:

  >>> f = lambda x, y0, y1: (x+1, y0+y1)
  >>> num_layers = 4
  >>> x = 0
  >>> ys_0 = jnp.array([1, 2, 3, 4])
  >>> ys_1 = jnp.array([5, 6, 7, 8])
  >>> stack = hk.layer_stack(num_layers, with_per_layer_inputs=True)
  >>> x, zs = stack(f)(x, ys_0, ys_1)
  >>> print(x, zs)
  4 [ 6  8 10 12]

  Check the tests in ``layer_stack_test.py`` for further examples.

  Crucially, any parameters created inside ``f`` will not be shared across
  iterations.

  Args:
    num_layers: The number of times to iterate the wrapped function.
    with_per_layer_inputs: Whether or not to pass per-layer inputs to the
      wrapped function.
    unroll: the unroll used by ``scan``.
    pass_reverse_to_layer_fn: Whether or not to pass the ``reverse`` keyword to
      the function ``f``, so that it is aware if the layer stack is being run
      forward or in reverse (and the underlying ``scan``). To run the layer
      stack in reverse you need to pass in ``reverse=True`` to the call to the
      layer stack.
    transparent: Whether to apply layer_stack transparently.  When this is True,
      and a correct transparency_map is provided, the parameters are generated
      in such a way that layer_stack can be replaced by a regular for loop
      without changing the parameter tree.
    transparency_map: How to map stacked module names to flat names and reverse.
      See ``LayerStackTransparencyMapping`` and ``layer_stack_test.py`` for an
      example.
    name: name of the Haiku context.

  Returns:
    Callable that will produce a layer stack when called with a valid function.
  """
  if transparent and transparency_map is None:
    raise ValueError("transparency_map must be provided with transparent=True.")

  if not name:
    if with_per_layer_inputs:
      name = "__layer_stack_with_per_layer"
    else:
      name = "__layer_stack_no_per_layer"

  def iterate(f):
    if with_per_layer_inputs:
      @functools.wraps(f)
      def wrapped(x, *args, **kwargs):
        for ys in jax.tree.leaves(args):
          assert ys.shape[0] == num_layers, f"{ys.shape[0]} != {num_layers}"
        mod = _LayerStackWithPerLayer(
            f,
            num_layers,
            unroll=unroll,
            pass_reverse_to_layer_fn=pass_reverse_to_layer_fn,
            transparency_map=transparency_map,
            name=name,
        )
        return mod(x, *args, **kwargs)
    else:
      _check_no_varargs(f)
      @functools.wraps(f)
      def wrapped(*args, **kwargs):
        mod = _LayerStackNoPerLayer(
            f,
            num_layers,
            unroll=unroll,
            pass_reverse_to_layer_fn=pass_reverse_to_layer_fn,
            transparency_map=transparency_map,
            name=name,
        )
        ret = mod(x=args, **kwargs)[0]
        if len(args) == 1:
          # If the function takes a single argument, we must also return a
          # single value, and not a tuple of length 1.
          ret = ret[0]
        return ret

    return wrapped
  return iterate
