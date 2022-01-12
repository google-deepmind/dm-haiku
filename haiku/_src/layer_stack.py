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
import contextlib
import functools
import inspect
from typing import Any, Callable, Optional, Tuple, Union

from haiku._src import base
from haiku._src import lift
from haiku._src import module
from haiku._src import transform
import jax
import jax.numpy as jnp


class LayerStackStateError(Exception):
  """Raise if trying to use layer_stack with Haiku state."""

LayerStackCarry = collections.namedtuple("LayerStackCarry", ["x"])
LayerStackScanned = collections.namedtuple("LayerStackScanned",
                                           ["params", "rng", "args_ys"])

# WrappedFn should take in arbitrarily nested `jnp.ndarray`, and return the
# exact same type. We cannot express this with `typing`. So we just use it
# to inform the user. In reality, the typing below will accept anything.
NestedArray = Any
WrappedFn = Callable[..., Union[NestedArray, Tuple[NestedArray]]]


def _check_no_varargs(f):
  if list(inspect.signature(
      f).parameters.values())[0].kind == inspect.Parameter.VAR_POSITIONAL:
    raise ValueError(
        "The function `f` should not have any `varargs` (that is *args) "
        "argument. Instead, it should only use explicit positional"
        "arguments")


@contextlib.contextmanager
def nullcontext():
  yield


def maybe_with_rng(key):
  if key is not None:
    return base.with_rng(key)
  else:
    return nullcontext()


def maybe_fold_in(key, data):
  if key is not None:
    return jax.random.fold_in(key, data)
  else:
    return None


def _get_rng_stack(count: int) -> jnp.ndarray:
  rng = base.maybe_next_rng_key()
  if rng is not None:
    rng = jax.random.split(rng, count)
  else:
    rng = jnp.zeros([count, 2], dtype=jnp.uint32)
  return rng


class _LayerStack(module.Module):
  """Module to compose parameterized functions, implemented as a scan."""

  def __init__(self,
               count: int,
               unroll: int,
               name: Optional[str] = None):
    """Iterate f count times, with non-shared parameters."""
    super().__init__(name=name)
    self._count = count
    self._unroll = unroll

  def __call__(self, x, *args_ys):
    count = self._count
    try:
      init_fn, apply_fn = transform.transform(self._call_wrapped)
    except ValueError as e:
      raise LayerStackStateError("LayerStack can only be used in Haiku "
                                 "functions which do not make use of Haiku "
                                 "state.") from e

    def per_layer_init_fn(c, a):
      c, rng = c
      if rng is not None:
        rng, next_rng, apply_rng = jax.random.split(rng, 3)
      else:
        rng, next_rng, apply_rng = None, None, None
      params = init_fn(rng, c, *a)
      c, _ = apply_fn(params, apply_rng, c, *a)
      return (c, next_rng), params

    def scanned_init_fn(x, rng):
      _, params = jax.lax.scan(per_layer_init_fn, (x, rng), args_ys,
                               length=self._count)
      return params

    rng = base.maybe_next_rng_key()
    lifted_init_fn = lift.transparent_lift(scanned_init_fn)
    params = lifted_init_fn(x, rng)

    # Use scan during apply, threading through random seed so that it's
    # unique for each layer.
    def layer(carry: LayerStackCarry,
              scanned: LayerStackScanned) -> Tuple[LayerStackCarry, Any]:
      rng = scanned.rng
      params = scanned.params

      out_x, z = apply_fn(params, rng, carry.x, *scanned.args_ys)
      return LayerStackCarry(x=out_x), z

    rng = _get_rng_stack(count)

    carry = LayerStackCarry(x=x)
    scanned = LayerStackScanned(params=params,
                                rng=rng,
                                args_ys=args_ys)

    carry, zs = jax.lax.scan(
        layer, carry, scanned, length=count, unroll=self._unroll)
    return carry.x, zs

  def _call_wrapped(self,
                    x: jnp.ndarray,
                    *args,
                    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    raise NotImplementedError()


class _LayerStackNoPerLayer(_LayerStack):
  """_LayerStack impl with no per-layer inputs provided to the function."""

  def __init__(self,
               f: WrappedFn,
               count: int,
               unroll: int,
               name: Optional[str] = None):
    super().__init__(count=count, unroll=unroll, name=name)
    _check_no_varargs(f)
    self._f = f

  @module.transparent
  def _call_wrapped(self, args, y):
    del y
    ret = self._f(*args)
    if len(args) == 1:
      # If the function takes a single argument, the wrapped function receives
      # a tuple of length 1, and therefore it must return a tuple of length 1.
      ret = (ret,)
    return ret, None


class _LayerStackWithPerLayer(_LayerStack):
  """_LayerStack impl with per-layer inputs provided to the function."""

  def __init__(self,
               f: WrappedFn,
               count: int,
               unroll: int,
               name: Optional[str] = None):
    super().__init__(count=count, unroll=unroll, name=name)
    self._f = f

  @module.transparent
  def _call_wrapped(self, x, *args):
    return self._f(x, *args)


def layer_stack(num_layers: int,
                with_per_layer_inputs=False,
                unroll: int = 1,
                name: Optional[str] = None):
  """Utility to wrap a Haiku function and recursively apply it to an input.

  This can be used to improve model compile times.

  A function is valid if it uses only explicit position parameters, and
  its return type matches its input type. The position parameters can be
  arbitrarily nested structures with ``jnp.ndarray`` at the leaf nodes. Note
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
  >>> stack = hk.experimental.layer_stack(num_layers,
  ...                                     with_per_layer_inputs=True)
  >>> x, zs = stack(f)(x, ys_0, ys_1)
  >>> x, zs
  (DeviceArray(4, dtype=int32, weak_type=True),
      DeviceArray([ 6,  8, 10, 12], dtype=int32))

  Check the tests in ``layer_stack_test.py`` for further examples.

  Crucially, any parameters created inside ``f`` will not be shared across
  iterations.

  Args:
    num_layers: The number of times to iterate the wrapped function.
    with_per_layer_inputs: Whether or not to pass per-layer inputs to the
      wrapped function.
    unroll: the unroll used by ``scan``.
    name: name of the Haiku context.

  Returns:
    Callable that will produce a layer stack when called with a valid function.
  """
  def iterate(f):
    if with_per_layer_inputs:
      @functools.wraps(f)
      def wrapped(x, *args):
        for ys in args:
          assert ys.shape[0] == num_layers
        return _LayerStackWithPerLayer(
            f, num_layers, unroll=unroll, name=name)(x, *args)
    else:
      _check_no_varargs(f)
      @functools.wraps(f)
      def wrapped(*args):
        ret = _LayerStackNoPerLayer(
            f, num_layers, unroll=unroll, name=name)(args, None)[0]
        if len(args) == 1:
          # If the function takes a single argument, we must also return a
          # single value, and not a tuple of length 1.
          ret = ret[0]
        return ret

    return wrapped
  return iterate
