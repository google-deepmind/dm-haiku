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
import inspect
from typing import Any, Callable, Mapping, MutableMapping, Optional, Tuple, TypeVar

from haiku._src import base
import jax
import jax.numpy as jnp

InternalState = collections.namedtuple("InternalState", "params,state,rng")
Bundle = Mapping[str, Mapping[str, Any]]
T = TypeVar("T")


def copy_structure(bundle: T) -> T:
  return jax.tree_map(lambda x: x, bundle)


def internal_state(*, params=True) -> InternalState:
  frame = base.current_frame()
  rng = frame.rng_stack.peek()
  if rng is not None:
    rng = rng.internal_state
  return InternalState(
      params=(copy_structure(frame.params) if params else None),
      state=copy_structure(frame.state),
      rng=copy_structure(rng))


def update_recursive(dst: MutableMapping[Any, Any], src: Mapping[Any, Any]):
  for k, v in src.items():
    if isinstance(v, collections.abc.Mapping):
      dst.setdefault(k, {})
      update_recursive(dst[k], v)
    else:
      if v is not None:
        # NOTE: We only expect `None` values thanks to `difference`.
        dst[k] = v


def update_internal_state(state: InternalState):
  frame = base.current_frame()
  if not frame.params_frozen and state.params is not None:
    update_recursive(frame.params, state.params)
  update_recursive(frame.state, state.state)
  rng = state.rng
  if rng is not None:
    frame.rng_stack.peek().replace_internal_state(rng)


def temporary_internal_state(state: InternalState, *, share_python_state=False):
  """Pushes a temporary copy of the internal state."""
  state = copy_structure(state)
  rng = state.rng
  if rng is not None:
    rng = base.PRNGSequence(rng)
  current_state = internal_state()
  params = state.params
  if params is None:
    params = current_state.params
  state = state.state
  if state is None:
    state = current_state.state
  frame = base.current_frame()
  frame = frame.evolve(params=params, state=state, rng=rng,
                       decoupled=(not share_python_state))
  return base.frame_stack(frame)


def reserve_up_to_full_rng_block():
  """If RNG is active in the current frame, reserve up to the default block."""
  # TODO(lenamartens): Fix needing full block reservation in stateful
  # control-flow by keeping track of current key with index and keeping a full
  # block in PRNGSequence at all time.
  rng_seq = base.current_frame().rng_stack.peek()
  if rng_seq:
    rng_seq.reserve_up_to_full()


def grad(fun, argnums=0, has_aux=False, holomorphic=False):
  r"""Creates a function which evaluates the gradient of ``fun``.

  NOTE: You only need this in a very specific case that you want to take a
  gradient **inside** a :func:`transform`\ ed function and the function you are
  differentiating uses :func:`set_state`. For example:

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
  DeviceArray(4., dtype=float32, weak_type=True)

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape ``()`` but not
      arrays with shape ``(1,)`` etc.)
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the gradient
    of ``fun``. If `argnums` is an integer then the gradient has the same shape
    and type as the positional argument indicated by that integer. If argnums is
    a tuple of integers, the gradient is a tuple of values with the same shapes
    and types as the corresponding arguments. If ``has_aux`` is True then a pair
    of ``gradient, auxiliary_data`` is returned.

  For example:

  >>> grad_tanh = jax.grad(jax.numpy.tanh)
  >>> print(grad_tanh(0.2))
  0.96...
  """
  value_and_grad_fun = value_and_grad(fun, argnums=argnums, has_aux=has_aux,
                                      holomorphic=holomorphic)

  @functools.wraps(fun)
  def grad_fn(*args, **kwargs):
    value, grads = value_and_grad_fun(*args, **kwargs)
    if has_aux:
      value, aux = value
      return grads, aux
    else:
      return grads

  return grad_fn


def value_and_grad(fun, argnums=0, has_aux=False, holomorphic=False):
  r"""Creates a function which evaluates both ``fun`` and the grad of ``fun``.

  NOTE: You only need this in a very specific case that you want to take a
  gradient **inside** a :func:`transform`\ ed function and the function you are
  differentiating uses :func:`set_state`. For example:

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
      ``argnums`` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape ``()`` but not
      arrays with shape ``(1,)`` etc.)
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun`` that evaluates both ``fun``
    and the gradient of ``fun`` and returns them as a pair (a two-element
    tuple). If ``argnums`` is an integer then the gradient has the same shape
    and type as the positional argument indicated by that integer. If argnums is
    a tuple of integers, the gradient is a tuple of values with the same shapes
    and types as the corresponding arguments.
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

  @functools.wraps(dec_fun)
  def wrapped_dec_fun(fun, *dec_args, **dec_kwargs):
    """Decorates a modified version of ``fun`` that passes Haiku state."""

    if not base.inside_transform():
      raise ValueError(
          "hk.{0}() should not be used outside of hk.transform. "
          "Use jax.{0}() instead.".format(dec_fun.__name__))

    @functools.wraps(fun)
    def stateful_fun(*args, **kwargs):
      state_in = kwargs.pop("hk_state")
      with temporary_internal_state(state_in, share_python_state=True):
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

  wrapped_dec_fun.__doc__ = f"Equivalent to jax.{dec_fun.__name__} but passing Haiku state."
  return wrapped_dec_fun


jit = thread_hk_state_in_kwargs(jax.jit)
remat = thread_hk_state_in_kwargs(jax.remat)


def stateful_branch(branch_fun):
  @functools.wraps(branch_fun)
  def new_branch_fun(operand):
    state, operand = operand
    with temporary_internal_state(state):
      out = branch_fun(operand)
      reserve_up_to_full_rng_block()
      # TODO(tomhennigan) Return difference of state in/out here.
      return out, internal_state()
  return new_branch_fun


def _new_cond(pred, true_fun, false_fun, operand):
  del pred, true_fun, false_fun, operand


def _old_cond(pred, true_operand, true_fun, false_operand, false_fun):
  del pred, true_operand, true_fun, false_operand, false_fun


def _memoize_by_id(f):
  """Memoizes the result of a higher order function on input function id."""
  cache = {}
  @functools.wraps(f)
  def wrapper(g):
    i = id(g)
    try:
      res = cache[i]
    except KeyError:
      res = cache[i] = f(g)
    return res
  return wrapper


def cond(*args, **kwargs):
  """Equivalent to :func:`jax.lax.cond` but with Haiku state passed in/out."""
  if not base.inside_transform():
    raise ValueError("hk.cond() should not be used outside of hk.transform(). "
                     "Use jax.cond() instead.")

  try:
    bound_args = inspect.signature(_old_cond).bind(*args, **kwargs)
  except TypeError:
    bound_args = inspect.signature(_new_cond).bind(*args, **kwargs)
    pred, true_fun, false_fun, operand = bound_args.args
  else:
    pred, true_operand, true_fun, false_operand, false_fun = bound_args.args
    true_fun = lambda op, f=true_fun: f(op[0])
    false_fun = lambda op, f=false_fun: f(op[1])
    operand = (true_operand, false_operand)

  reserve_up_to_full_rng_block()
  stateful_branch_mem = _memoize_by_id(stateful_branch)
  state = internal_state()
  out, state = jax.lax.cond(pred,
                            true_fun=stateful_branch_mem(true_fun),
                            false_fun=stateful_branch_mem(false_fun),
                            operand=(state, operand))
  update_internal_state(state)
  return out


def switch(index, branches, operand):
  """Equivalent to :func:`jax.lax.switch` but with Haiku state passed in/out."""
  if not base.inside_transform():
    raise ValueError(
        "hk.switch() should not be used outside of hk.transform(). "
        "Use jax.switch() instead.")

  reserve_up_to_full_rng_block()
  stateful_branch_mem = _memoize_by_id(stateful_branch)
  state = internal_state()
  out, state = jax.lax.switch(
      index, tuple(map(stateful_branch_mem, branches)), (state, operand))
  update_internal_state(state)
  return out


def scan(f, init, xs, length=None, reverse=False, unroll=1):
  """Equivalent to :func:`jax.lax.scan` but with Haiku state passed in/out."""
  if not base.inside_transform():
    raise ValueError("hk.scan() should not be used outside of hk.transform(). "
                     "Use jax.scan() instead.")

  if length is None:
    length = jax.tree_leaves(xs)[0].shape[0]

  running_init_fn = not base.params_frozen()

  if running_init_fn:
    # During `init` we need to unroll one step of the scan, this is because our
    # carry contains the Haiku state and during `init` this may change structure
    # (e.g. as state is created).
    if not length:
      x0 = jax.tree_map(lambda x: jnp.zeros(x.shape[1:], x.dtype), xs)
      _, y0 = f(init, x0)
      y0 = jax.tree_map(lambda y: jnp.zeros((0,) + y.shape, y.dtype), y0)
      return init, y0

    if reverse:
      x0 = jax.tree_map(lambda x: x[-1], xs)
      xs = jax.tree_map(lambda x: x[:-1], xs)
    else:
      x0 = jax.tree_map(lambda x: x[0], xs)
      xs = jax.tree_map(lambda x: x[1:], xs)
    init, y0 = f(init, x0)
    y0 = jax.tree_map(lambda y: jnp.expand_dims(y, 0), y0)
    length -= 1
    if not length:
      return init, y0

  @functools.wraps(f)
  def stateful_fun(carry, x):
    carry, state = carry
    with temporary_internal_state(state):
      with base.assert_no_new_parameters():
        carry, out = f(carry, x)
      reserve_up_to_full_rng_block()
      carry = (carry, internal_state(params=False))
      return carry, out

  # Before pulling out the  internal state,  reserve a full block  of RNG keys.
  # This is to make sure we're always passing in the same amount of subkeys in
  # and out of the scan carry (scan requires equal length lists).
  # After every scan iteration we reserve back up to the full block.
  reserve_up_to_full_rng_block()

  # We know that we don't need to thread params in and out, since for init we
  # have already created them (given that above we unroll one step of the scan)
  # and for apply we know they are immutable. As such we only need to thread the
  # state and rng in and out.

  init = (init, internal_state(params=False))
  (carry, state), ys = jax.lax.scan(
      stateful_fun, init, xs, length, reverse, unroll=unroll)
  update_internal_state(state)

  if running_init_fn:
    if reverse:
      ys = jax.tree_multimap(lambda y0, ys: jnp.concatenate([ys, y0]), y0, ys)
    else:
      ys = jax.tree_multimap(lambda y0, ys: jnp.concatenate([y0, ys]), y0, ys)

  return carry, ys


def fori_loop(lower, upper, body_fun, init_val):
  """Equivalent to :func:`jax.lax.fori_loop` with Haiku state passed in/out."""
  if not base.inside_transform():
    raise ValueError(
        "hk.fori_loop() should not be used outside of hk.transform(). "
        "Use jax.lax.fori_loop() instead.")

  @functools.wraps(body_fun)
  def pure_body_fun(i, val):
    state, val = val
    with temporary_internal_state(state):
      val = body_fun(i, val)
      reserve_up_to_full_rng_block()
      state = internal_state()
      return state, val

  if not base.params_frozen():
    # During init we need to unwind one step of the loop to ensure the Haiku
    # state before and after the body has the same structure.
    init_val = body_fun(lower, init_val)
    lower += 1
    try:
      if upper - lower == 0:
        return init_val
    except jax.errors.ConcretizationTypeError:
      # upper or lower might be tracers, which jax.lax.fori_loop can handle.
      pass

  reserve_up_to_full_rng_block()
  state = internal_state()
  init_val = state, init_val
  state, val = jax.lax.fori_loop(lower, upper, pure_body_fun, init_val)
  update_internal_state(state)
  return val


def vmap(fun, in_axes=0, out_axes=0, axis_name=None):
  """Equivalent to :func:`jax.vmap` with module parameters/state not mapped."""

  # TODO(tomhennigan): Allow configuration of params/state/rng mapping.
  in_axes = in_axes, None
  out_axes = out_axes, None

  @functools.wraps(fun)
  def pure_fun(args, state_in):
    with temporary_internal_state(state_in):
      out = fun(*args)
      state_out = difference(state_in, internal_state())
      return out, state_out

  @functools.wraps(fun)
  def mapped_fun(*args):
    mapped_pure_fun = jax.vmap(pure_fun, in_axes=in_axes, out_axes=out_axes,
                               axis_name=axis_name)
    state = internal_state()
    out, state = mapped_pure_fun(args, state)
    update_internal_state(state)
    return out

  return mapped_fun


def while_loop(cond_fun, body_fun, init_val):
  """Equivalent to jax.lax.while_loop with Haiku state threaded in/out."""

  if not base.params_frozen():
    raise ValueError(
        "hk.while_loop does not support initialization (since we cannot "
        "statically determine if your loop will run at least once). Please "
        "use `hk.running_init` to run the body unconditionally:\n"
        "\n"
        "    if hk.running_init():\n"
        "      # Unconditionally connect the module at init time.\n"
        "      val = module(val)\n"
        "    else:\n"
        "      val = hk.while_loop(lambda val: val.mean() < 1, module, val)\n")

  @functools.wraps(cond_fun)
  def pure_cond_fun(val):
    val, _ = val
    try:
      with base.assert_state_unchanged():
        return cond_fun(val)
    except base.StateChangedError as e:
      # If we find a use case for updating state/using rng in `cond` we would
      # need to make a change in JAX itself (to support aux in/out of the cond).
      raise ValueError(
          "`hk.while_loop` does not support `hk.set_state`, `hk.next_rng_key` "
          "(et al) in `cond_fun`."
      ) from e

  @functools.wraps(body_fun)
  def pure_body_fun(val):
    val, state = val
    with temporary_internal_state(state):
      val = body_fun(val)
      state = internal_state()
      return val, state

  init_val = (init_val, internal_state())
  val, state = jax.lax.while_loop(pure_cond_fun, pure_body_fun, init_val)
  update_internal_state(state)
  return val


def named_call(
    fun: Callable[..., Any],
    *,
    name: Optional[str] = None,
) -> Callable[..., Any]:
  """Wraps a function in an XLA name_scope and maintains Haiku state."""

  @functools.wraps(fun)
  def hide_non_jaxtype_outputs(fun, side_channel):
    @functools.wraps(fun)
    def named_call_hidden_outputs(*args, **kwargs):
      out = fun(*args, **kwargs)
      out_leaves, treedef = jax.tree_flatten(out)

      # Partition the output into valid and invalid JAX output. The invalid
      # output types are not returned from the function, but moved out
      # through a side channel.
      # In order to easily merge the output back later, replace the elements
      # of the other partition with Nones.
      out_leaves = [(x, None) if isinstance(x, jnp.ndarray) else (None, x)
                    for x in out_leaves]
      jax_types, non_jaxtypes = zip(*out_leaves)
      side_channel["non_jaxtypes"] = non_jaxtypes
      side_channel["treedef"] = treedef
      return jax_types
    return named_call_hidden_outputs

  @functools.wraps(fun)
  def wrapper(*args, **kwargs):
    side_channel = {"non_jaxtypes": [], "treedef": None}
    wrapped_fun = hide_non_jaxtype_outputs(fun, side_channel)
    if base.inside_transform():
      wrapped_fun = thread_hk_state_in_kwargs(jax.named_call)(wrapped_fun,
                                                              name=name)
    else:
      wrapped_fun = jax.named_call(wrapped_fun, name=name)

    jax_types = wrapped_fun(*args, **kwargs)

    non_jaxtypes = side_channel["non_jaxtypes"]
    out_leaves = [y if x is None else x
                  for x, y in zip(jax_types, non_jaxtypes)]
    out = jax.tree_unflatten(side_channel["treedef"], out_leaves)

    return out
  return wrapper


def eval_shape(fun, *args, **kwargs):
  """Equivalent to jax.eval_shape with any changed Haiku state discarded."""
  if not base.inside_transform():
    raise ValueError(
        "hk.eval_shape() should not be used outside of hk.transform(). "
        "Use jax.eval_shape() instead.")

  @functools.wraps(fun)
  def stateless_fun(state, *args, **kwargs):
    with temporary_internal_state(state):
      out = fun(*args, **kwargs)
      # Don't return changed state
      return out

  out_shape = jax.eval_shape(stateless_fun, internal_state(), *args, **kwargs)
  return out_shape
