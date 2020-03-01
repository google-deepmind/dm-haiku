# Lint as: python3
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Wrappers for JAX transformations that respect Haiku internal state."""

import collections
import functools
from typing import Any, Mapping, MutableMapping, Text

from haiku._src import base
import jax

InternalState = collections.namedtuple("InternalState", "params,state,rng")
Bundle = Mapping[Text, Mapping[Text, Any]]


def copy_structure(bundle: Bundle) -> Bundle:
  return jax.tree_map(lambda x: x, bundle)


def internal_state() -> InternalState:
  frame = base.current_frame()
  rng = frame.rng_stack.peek()
  if rng is not None:
    rng = rng.peek()
  return InternalState(params=copy_structure(frame.params),
                       state=copy_structure(frame.state),
                       rng=rng)


def update_recursive(dst: MutableMapping[Any, Any], src: Mapping[Any, Any]):
  for k, v in src.items():
    if isinstance(v, collections.Mapping):
      dst.setdefault(k, {})
      update_recursive(dst[k], v)
    else:
      dst[k] = v


def update_internal_state(state: InternalState):
  frame = base.current_frame()
  if not frame.params_frozen:
    update_recursive(frame.params, state.params)
  update_recursive(frame.state, state.state)
  rng = state.rng
  if rng is not None:
    frame.rng_stack.peek().replace(rng)


def temporary_internal_state(state: InternalState):
  rng = state.rng
  if rng is not None:
    rng = base.PRNGSequence(rng)
  frame = base.current_frame()
  frame = frame.evolve(params=state.params, state=state.state, rng=rng)
  return base.frame_stack(frame)


def grad(fun, argnums=0, has_aux=False, holomorphic=False):
  """Creates a function which evaluates the gradient of `fun`.

  NOTE: You only need this in a very specific case that you want to take a
  gradient **inside** a `hk.transform`ed function and the function you are
  differentiating uses `hk.set_state`. For example:

  >>> class MyModule(hk.Module):
  ...   def __call__(self, x):
  ...     hk.set_state("last", x ** 2)
  ...     return x ** 2

  >>> def f(x):
  ...   m = MyModule()
  ...   g = hk.grad(m)(x)
  ...   return g

  >>> f = hk.transform_with_state(f)
  >>> x = jnp.array(2.)
  >>> params, state = jax.jit(f.init)(None, x)
  >>> state["my_module"]["last"]
  DeviceArray(4., dtype=float32)

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      `argnums` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape `()` but not
      arrays with shape `(1,)` etc.)
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether `fun` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether `fun` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as `fun`, that evaluates the gradient of
    `fun`. If `argnums` is an integer then the gradient has the same shape and
    type as the positional argument indicated by that integer. If argnums is a
    tuple of integers, the gradient is a tuple of values with the same shapes
    and types as the corresponding arguments. If `has_aux` is True then a pair
    of (gradient, auxiliary_data) is returned.

  For example:

  >>> grad_tanh = jax.grad(jax.numpy.tanh)
  >>> print(grad_tanh(0.2))
  0.96...
  """
  value_and_grad_fun = value_and_grad(fun, argnums=argnums, has_aux=has_aux,
                                      holomorphic=holomorphic)

  def grad_fn(*args, **kwargs):
    value, grads = value_and_grad_fun(*args, **kwargs)
    if has_aux:
      value, aux = value
      return grads, aux
    else:
      return grads

  return grad_fn


def value_and_grad(fun, argnums=0, has_aux=False, holomorphic=False):
  """Creates a function which evaluates both `fun` and the gradient of `fun`.

  NOTE: You only need this in a very specific case that you want to take a
  gradient **inside** a `hk.transform`ed function and the function you are
  differentiating uses `hk.set_state`. For example:

  >>> class MyModule(hk.Module):
  ...   def __call__(self, x):
  ...     hk.set_state("last", jnp.sum(x))
  ...     return x ** 2

  >>> def f(x):
  ...   m = MyModule()
  ...   y, g = hk.value_and_grad(m)(x)
  ...   return y, g

  >>> f = hk.transform_with_state(f)
  >>> x = jnp.array(2.)
  >>> _ = jax.jit(f.init)(None, x)

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      `argnums` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape `()` but not
      arrays with shape `(1,)` etc.)
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether `fun` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether `fun` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as `fun` that evaluates both `fun` and
    the gradient of `fun` and returns them as a pair (a two-element tuple). If
    `argnums` is an integer then the gradient has the same shape and type as the
    positional argument indicated by that integer. If argnums is a tuple of
    integers, the gradient is a tuple of values with the same shapes and types
    as the corresponding arguments.
  """
  if not base.inside_transform():
    raise ValueError("hk.grad() should not be used outside of hk.transform(). "
                     "Use jax.grad() instead.")

  @functools.wraps(fun)
  def stateful_fun(*args, **kwargs):
    with temporary_internal_state(kwargs.pop("hk_state")):
      out = fun(*args, **kwargs)
      out, aux = (out if has_aux else (out, None))
      return out, (aux, internal_state())

  grad_fun = jax.value_and_grad(stateful_fun, argnums=argnums,
                                has_aux=True, holomorphic=holomorphic)

  @functools.wraps(grad_fun)
  def wrapper(*args, **kwargs):
    kwargs["hk_state"] = internal_state()
    (value, (aux, hk_state)), grads = grad_fun(*args, **kwargs)
    update_internal_state(hk_state)
    if has_aux:
      return (value, aux), grads
    else:
      return value, grads

  return wrapper


def thread_hk_state_in_kwargs(dec_fun):
  """Equivalent to jax.{} but passing Haiku state.""".format(dec_fun.__name__)

  def wrapped_dec_fun(fun, *dec_args, **dec_kwargs):
    """Decorates a modified version of `fun` that passes haiku state."""

    if not base.inside_transform():
      raise ValueError(
          "hk.{0}() should not be used outside of hk.transform. "
          "Use jax.{0}() instead.".format(dec_fun.__name__))

    @functools.wraps(fun)
    def stateful_fun(*args, **kwargs):
      with temporary_internal_state(kwargs.pop("hk_state")):
        out = fun(*args, **kwargs)
        return out, internal_state()

    dec_stateful_fun = dec_fun(stateful_fun, *dec_args, **dec_kwargs)

    @functools.wraps(dec_stateful_fun)
    def wrapper(*args, **kwargs):
      kwargs["hk_state"] = internal_state()
      out, state = dec_stateful_fun(*args, **kwargs)
      update_internal_state(state)
      return out

    return wrapper

  return wrapped_dec_fun


jit = thread_hk_state_in_kwargs(jax.jit)
remat = thread_hk_state_in_kwargs(jax.remat)


def stateful_branch(branch_fun):
  @functools.wraps(branch_fun)
  def new_branch_fun(operand):
    state, operand = operand
    with temporary_internal_state(state):
      out = branch_fun(operand)
      return out, internal_state()
  return new_branch_fun


def cond(pred, true_operand, true_fun, false_operand, false_fun):
  """Equivalent to `jax.lax.cond` but with Haiku state threaded in and out."""
  if not base.inside_transform():
    raise ValueError("hk.cond() should not be used outside of hk.transform(). "
                     "Use jax.cond() instead.")
  state = internal_state()
  out, state = jax.lax.cond(pred,
                            true_operand=(state, true_operand),
                            true_fun=stateful_branch(true_fun),
                            false_operand=(state, false_operand),
                            false_fun=stateful_branch(false_fun))
  update_internal_state(state)
  return out
