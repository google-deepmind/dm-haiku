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
from typing import Any, Mapping, MutableMapping, Optional, Tuple, TypeVar

from haiku._src import base
import jax

InternalState = collections.namedtuple("InternalState", "params,state,rng")
Bundle = Mapping[str, Mapping[str, Any]]
T = TypeVar("T")


def copy_structure(bundle: T) -> T:
  return jax.tree_map(lambda x: x, bundle)


def internal_state() -> InternalState:
  frame = base.current_frame()
  rng = frame.rng_stack.peek()
  if rng is not None:
    rng = rng.internal_state
  return InternalState(params=copy_structure(frame.params),
                       state=copy_structure(frame.state),
                       rng=copy_structure(rng))


def update_recursive(dst: MutableMapping[Any, Any], src: Mapping[Any, Any]):
  for k, v in src.items():
    if isinstance(v, collections.Mapping):
      dst.setdefault(k, {})
      update_recursive(dst[k], v)
    else:
      if v is not None:
        # NOTE: We only expect `None` values thanks to `difference`.
        dst[k] = v


def update_internal_state(state: InternalState):
  frame = base.current_frame()
  if not frame.params_frozen:
    update_recursive(frame.params, state.params)
  update_recursive(frame.state, state.state)
  rng = state.rng
  if rng is not None:
    frame.rng_stack.peek().replace_internal_state(rng)


def temporary_internal_state(state: InternalState):
  state = copy_structure(state)
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
    state_in = kwargs.pop("hk_state")
    with temporary_internal_state(state_in):
      out = fun(*args, **kwargs)
      out, aux = (out if has_aux else (out, None))
      state_out = difference(state_in, internal_state())
      return out, (aux, state_out)

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


class Box:
  """A pytree leaf that acts as a box."""

  def __init__(self, value):
    self.value = value

TwoLevelMapping = Mapping[Any, Mapping[Any, Any]]
TwoLevelMappingToBox = Mapping[Any, Mapping[Any, Box]]


def box_and_fill_missing(
    a: TwoLevelMapping,
    b: TwoLevelMapping,
) -> Tuple[TwoLevelMappingToBox, TwoLevelMappingToBox]:
  """Returns boxed two level mappings with the same structure.

  It is assumed that ``a`` is a subset of ``b``.

  Args:
    a: A two level mapping (e.g. Haiku parameters or state).
    b: A two level mapping (e.g. Haiku parameters or state).

  Returns:
    A pair of two level mappings with ``Box`` wrapped leaves (suitable for use
    with ``jax.tree_*``). The mappings have the contents of ``a`` and ``b``
    respectively. Both mappings have the structure from ``b``. Any missing
    elements are set to ``Box(None)``.
  """
  out_a = {k: {} for k in b}
  out_b = {k: {} for k in b}
  for k1, v1 in b.items():
    for k2 in v1:
      out_b[k1][k2] = Box(b[k1][k2])
      if k1 in a and k2 in a[k1]:
        out_a[k1][k2] = Box(a[k1][k2])
      else:
        out_a[k1][k2] = Box(None)
  return out_a, out_b


def difference(before: InternalState, after: InternalState) -> InternalState:
  """Returns an InternalState object with unchanged items set to ``None``.

  Note that to determine what values have changed we compare them by identity
  not by value. This is only reasonable to do if `difference` is used to compare
  state *inside* a JAX transform (e.g. comparing the arguments passed into JIT
  with the values that you are about to return from it).

  This function never produces false negatives (e.g. we will never incorrectly
  say that a piece of state is unchanged when it has), however it may produce
  false positives. One well known case is if a value is traced by an inner JAX
  transform but unchanged, the identity of the Python object will differ from
  the value passed into the outer function, but the value will not have changed.
  In this case `difference` will say that the value has changed. For example if
  the following change happened inside a function whose state was being diffed
  we would defensively say that ``u`` had changed value even though it had only
  changed Python identity:

  >>> u = hk.get_state("u", [], init=jnp.ones)
  >>> u, _ = jax.jit(lambda a: a, a ** 2)(u)
  >>> hk.set_state("u", u)

  Args:
    before: state before.
    after: state after.

  Returns:
    The difference between before and after, with any values that have the same
    identity before and after set to `None`.
  """

  def if_changed(is_new, box_a, box_b):
    if box_a.value is None or is_new(box_a.value, box_b.value):
      return box_b.value
    else:
      return None

  # params
  is_new_param = lambda a, b: a is not b
  params_before, params_after = box_and_fill_missing(before.params,
                                                     after.params)
  params_after = jax.tree_multimap(functools.partial(if_changed, is_new_param),
                                   params_before, params_after)

  # state
  def is_new_state(a: base.StatePair, b: base.StatePair):
    return a.initial is not b.initial or a.current is not b.current

  state_before, state_after = box_and_fill_missing(before.state, after.state)
  state_after = jax.tree_multimap(functools.partial(if_changed, is_new_state),
                                  state_before, state_after)

  # rng
  def is_new_rng(a: Optional[base.PRNGSequenceState],
                 b: Optional[base.PRNGSequenceState]):
    if a is None:
      return True
    assert len(a) == 2 and len(b) == 2
    return a[0] is not b[0] or a[1] is not b[1]

  rng = after.rng if is_new_rng(before.rng, after.rng) else None

  return InternalState(params_after, state_after, rng)


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
      state_in = kwargs.pop("hk_state")
      with temporary_internal_state(state_in):
        out = fun(*args, **kwargs)
        return out, difference(state_in, internal_state())

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
      # TODO(tomhennigan) Return difference of state in/out here.
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
