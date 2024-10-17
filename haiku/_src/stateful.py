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
import collections.abc
from collections.abc import Callable, Mapping, MutableMapping
import functools
import inspect
from typing import Any, TypeVar

from haiku._src import base
import jax
import jax.numpy as jnp

InternalState = collections.namedtuple("InternalState", "params,state,rng")
Bundle = Mapping[str, Mapping[str, Any]]
T = TypeVar("T")
python_map = map  # pylint: disable=invalid-name


def copy_structure(bundle: T) -> T:
  return jax.tree.map(lambda x: x, bundle)


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
  >>> print(state["my_module"]["last"])
  4.0

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
    with temporary_internal_state(state_in), \
         base.push_jax_trace_level():
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
) -> tuple[TwoLevelMappingToBox, TwoLevelMappingToBox]:
  """Returns boxed two level mappings with the same structure.

  It is assumed that ``a`` is a subset of ``b``.

  Args:
    a: A two level mapping (e.g. Haiku parameters or state).
    b: A two level mapping (e.g. Haiku parameters or state).

  Returns:
    A pair of two level mappings with ``Box`` wrapped leaves (suitable for use
    with ``jax.tree_util.tree_*``). The mappings have the contents of ``a`` and
    ``b`` respectively. Both mappings have the structure from ``b``. Any missing
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
  params_after = jax.tree.map(
      functools.partial(if_changed, is_new_param), params_before, params_after
  )

  # state
  def is_new_state(a: base.StatePair, b: base.StatePair):
    return a.initial is not b.initial or a.current is not b.current

  state_before, state_after = box_and_fill_missing(before.state, after.state)
  state_after = jax.tree.map(
      functools.partial(if_changed, is_new_state), state_before, state_after
  )

  # rng
  def is_new_rng(a: base.PRNGSequenceState | None,
                 b: base.PRNGSequenceState | None):
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
      with temporary_internal_state(state_in, share_python_state=True), \
           base.push_jax_trace_level():
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

# pylint: disable=unnecessary-lambda
# pylint: disable=g-long-lambda
# Calls factory function to create a function then applies it to arguments.
_factory = lambda f: functools.wraps(f)(lambda *a, **k: f()(*a, **k))
# Wraps a function inside a lambda to hide its identity.
_lambda_wrap = lambda f: functools.wraps(f)(lambda *a, **k: f(*a, **k))
# jax.remat foiling the function identity cache.
_remat_no_cache = functools.wraps(jax.remat)(
    lambda f, *args, **kwargs: functools.wraps(f)(
        _factory(lambda: jax.remat(_lambda_wrap(f), *args, **kwargs))))
# Haiku version of jax.remat.
remat = thread_hk_state_in_kwargs(_remat_no_cache)
# pylint: enable=unnecessary-lambda
# pylint: enable=g-long-lambda


def stateful_branch(branch_fun):
  """Calls branch_fun passing internal state in and out."""
  @functools.wraps(branch_fun)
  def new_branch_fun(operand):
    state, operand = operand
    with temporary_internal_state(state), \
         base.push_jax_trace_level():
      out = branch_fun(*operand)
      reserve_up_to_full_rng_block()
      # TODO(tomhennigan) Return difference of state in/out here.
      return out, internal_state()
  return new_branch_fun

SENTINEL = object()


def _new_cond(pred, true_fun, false_fun, *operands, operand=SENTINEL):
  del pred, true_fun, false_fun, operands, operand


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


RUNNING_INIT_HINT = """
Hint: A common mistake is to use hk.cond(..) or `hk.switch(..)` at init time and
      create module parameters in one of the branches. At init time you should
      unconditionally create the parameters of all modules you might want to use
      at apply.

For hk.cond():

    if hk.running_init():
      # At init time unconditionally create parameters in my_module.
      my_other_module(x)
      out = my_module(x)
    else:
      out = hk.cond(pred, my_module, my_other_module)

For hk.switch():

    branches = [my_module, lambda x: x]
    if hk.running_init():
      # At init time unconditionally create parameters in all branches.
      for branch in branches:
        out = my_module(x)
    else:
      out = hk.switch(idx, branches, x)
""".strip()


def with_output_structure_hint(f):
  """Adds a helpful hint to branch structure errors."""
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except TypeError as e:
      if not base.params_frozen() and "must have same type structure" in str(e):
        raise TypeError(RUNNING_INIT_HINT) from e
      else:
        raise e
  return wrapper


# pylint: disable=g-doc-args
@functools.wraps(jax.lax.cond)
@with_output_structure_hint
def cond(*args, **kwargs):
  """Equivalent to :func:`jax.lax.cond` but with Haiku state passed in/out.

  >>> true_fn = hk.nets.ResNet50(10)
  >>> false_fn = hk.Sequential([hk.Flatten(), hk.nets.MLP([300, 100, 10])])
  >>> x = jnp.ones([1, 224, 224, 3])
  >>> if hk.running_init():
  ...   # At `init` run both branches to create parameters everywhere.
  ...   true_fn(x)
  ...   out = false_fn(x)
  ... else:
  ...   # At `apply` conditionally call one of the modules.
  ...   i = jax.random.randint(hk.next_rng_key(), [], 0, 100)
  ...   out = hk.cond(i > 50, true_fn, false_fn, x)

  Args:
    pred: Boolean scalar type.
    true_fun: Function (A -> B), to be applied if ``pred`` is ``True``.
    false_fun: Function (A -> B), to be applied if ``pred`` is ``False``.
    operands: Operands (A) input to either branch depending on ``pred``. The
      type can be a scalar, array, or any pytree (nested Python tuple/list/dict)
      thereof.

  Returns:
    Value (B) of either ``true_fun(*operands)`` or ``false_fun(*operands)``,
    depending on the value of ``pred``. The type can be a scalar, array, or any
    pytree (nested Python tuple/list/dict) thereof.
  """
# pylint: enable=g-doc-args
  if not base.inside_transform():
    raise ValueError("hk.cond() should not be used outside of hk.transform(). "
                     "Use jax.cond() instead.")

  try:
    bound_args = inspect.signature(_old_cond).bind(*args, **kwargs)
    pred, true_operand, true_fun, false_operand, false_fun = bound_args.args
    if not callable(true_fun) or not callable(false_fun):
      # Two operand new cond case: cond(pred, tf, ff, 1, 2)
      raise TypeError
  except TypeError:
    bound_args = inspect.signature(_new_cond).bind(*args, **kwargs)
    bound_args.apply_defaults()
    pred, true_fun, false_fun, *operands = bound_args.args
    operand = bound_args.kwargs["operand"]
    if operand is not SENTINEL:
      if operands:
        raise ValueError("When the operand keyword argument is used you cannot "  # pylint: disable=raise-missing-from
                         "also pass operands positionally. Got "
                         f"operand={operand} and *operands={tuple(operands)}")
      operands = (operand,)
      del operand
  else:
    true_fun = lambda op, f=true_fun: f(op[0])
    false_fun = lambda op, f=false_fun: f(op[1])
    operands = ((true_operand, false_operand),)

  reserve_up_to_full_rng_block()
  stateful_branch_mem = _memoize_by_id(stateful_branch)
  state = internal_state()
  out, state = jax.lax.cond(pred,
                            true_fun=stateful_branch_mem(true_fun),
                            false_fun=stateful_branch_mem(false_fun),
                            operand=(state, operands))
  update_internal_state(state)
  return out


@with_output_structure_hint
def switch(index, branches, *operands):
  r"""Equivalent to :func:`jax.lax.switch` but with Haiku state passed in/out.

  Note that creating parameters inside a switch branch is not supported, as such
  at init time we recommend you unconditionally evaluate all branches of your
  switch and only use the switch at apply. For example:

  >>> experts = [hk.nets.MLP([300, 100, 10]) for _ in range(5)]
  >>> x = jnp.ones([1, 28 * 28])
  >>> if hk.running_init():
  ...   # During init unconditionally create params/state for all experts.
  ...   for expert in experts:
  ...     out = expert(x)
  ... else:
  ...   # During apply conditionally apply (and update) only one expert.
  ...   index = jax.random.randint(hk.next_rng_key(), [], 0, len(experts) - 1)
  ...   out = hk.switch(index, experts, x)

  Args:
    index: Integer scalar type, indicating which branch function to apply.
    branches: Sequence of functions (A -> B) to be applied based on index.
    operands: Operands (A) input to whichever branch is applied.

  Returns:
    Value (B) of branch(\*operands) for the branch that was selected based on
    index.
  """
  if not base.inside_transform():
    raise ValueError(
        "hk.switch() should not be used outside of hk.transform(). "
        "Use jax.switch() instead.")

  reserve_up_to_full_rng_block()
  stateful_branch_mem = _memoize_by_id(stateful_branch)
  state = internal_state()
  out, state = jax.lax.switch(
      index, tuple(python_map(stateful_branch_mem, branches)), (state, operands)
  )
  update_internal_state(state)
  return out


def scan(f, init, xs, length=None, reverse=False, unroll=1):
  """Equivalent to :func:`jax.lax.scan` but with Haiku state passed in/out."""
  if not base.inside_transform():
    raise ValueError("hk.scan() should not be used outside of hk.transform(). "
                     "Use jax.lax.scan() instead.")

  if length is None:
    length = jax.tree.leaves(xs)[0].shape[0]

  running_init_fn = not base.params_frozen()

  if running_init_fn:
    # During `init` we need to unroll one step of the scan, this is because our
    # carry contains the Haiku state and during `init` this may change structure
    # (e.g. as state is created).
    if not length:
      x0 = jax.tree.map(lambda x: jnp.zeros(x.shape[1:], x.dtype), xs)
      _, y0 = f(init, x0)
      y0 = jax.tree.map(lambda y: jnp.zeros((0,) + y.shape, y.dtype), y0)
      return init, y0

    if reverse:
      x0 = jax.tree.map(lambda x: x[-1], xs)
      xs = jax.tree.map(lambda x: x[:-1], xs)
    else:
      x0 = jax.tree.map(lambda x: x[0], xs)
      xs = jax.tree.map(lambda x: x[1:], xs)
    init, y0 = f(init, x0)
    y0 = jax.tree.map(lambda y: jnp.expand_dims(y, 0), y0)
    length -= 1
    if not length:
      return init, y0

  @functools.wraps(f)
  def stateful_fun(carry, x):
    carry, state = carry
    with temporary_internal_state(state):
      with base.assert_no_new_parameters(), \
           base.push_jax_trace_level():
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
      ys = jax.tree.map(lambda y0, ys: jnp.concatenate([ys, y0]), y0, ys)
    else:
      ys = jax.tree.map(lambda y0, ys: jnp.concatenate([y0, ys]), y0, ys)

  return carry, ys


def map(f, xs):  # pylint: disable=redefined-builtin
  """Equivalent to :func:`jax.lax.map` but with Haiku state passed in/out."""
  g = lambda _, x: ((), f(x))
  _, ys = scan(g, (), xs)
  return ys


def fori_loop(lower, upper, body_fun, init_val):
  """Equivalent to :func:`jax.lax.fori_loop` with Haiku state passed in/out."""
  if not base.inside_transform():
    raise ValueError(
        "hk.fori_loop() should not be used outside of hk.transform(). "
        "Use jax.lax.fori_loop() instead.")

  @functools.wraps(body_fun)
  def pure_body_fun(i, val):
    state, val = val
    with temporary_internal_state(state), \
         base.push_jax_trace_level():
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


def maybe_get_axis(axis: int | None, arrays: Any) -> int | None:
  """Returns `array.shape[axis]` for one of the arrays in the input."""
  if axis is None: return None
  shapes = [a.shape for a in jax.tree.leaves(arrays)]
  sizes = {s[axis] for s in shapes}
  if len(sizes) != 1:
    raise ValueError("Arrays must have the same mapped axis size, found "
                     f"sizes {sizes} for input shapes {shapes}")
  size, = sizes
  return size

# Uniq but maintaining insertion order.
uniq = lambda x: tuple({k: None for k in x}.keys())


def get_mapped_axis_size(args: tuple[Any], in_axes: Any) -> int:
  sizes = uniq(
      jax.tree.leaves(
          jax.tree.map(
              maybe_get_axis, in_axes, args, is_leaf=lambda x: x is None
          )
      )
  )
  assert sizes, "hk.vmap should guarantee non-empty in_axes"
  # NOTE: We use the first in_axes regardless of how many non-unique values
  # there are to allow JAX to handle multiple conflicting sizes.
  return sizes[0]


def add_split_rng_error(f):
  """Adds a nice error message when split_rng is missing."""

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    if "split_rng" not in kwargs and not wrapper.require_split_rng:
      kwargs["split_rng"] = False

    if "split_rng" not in kwargs:
      try:
        return f(*args, **kwargs)
      except TypeError as e:
        raise TypeError("Haiku now requires the split_rng argument to be "
                        "passed to hk.vmap. If you have code using the old "
                        "API which you cannot change, you can opt-out of this "
                        "requirement by using "
                        "`hk.vmap.require_split_rng = False`.") from e

    return f(*args, **kwargs)

  wrapper.require_split_rng = True
  return wrapper

list_to_tuple = lambda x: tuple(x) if isinstance(x, list) else x


@add_split_rng_error
def vmap(
    fun: Callable[..., Any],
    in_axes=0,
    out_axes=0,
    axis_name: str | None = None,
    axis_size: int | None = None,
    *,
    split_rng: bool,
) -> Callable[..., Any]:
  """Equivalent to :func:`jax.vmap` with module parameters/state not mapped.

  The behaviour of Haiku random key APIs under :func:`vmap` is controlled by the
  ``split_rng`` argument:

  >>> x = jnp.arange(2)
  >>> f = hk.vmap(lambda _: hk.next_rng_key(), split_rng=False)
  >>> key1, key2 = f(x)
  >>> assert (key1 == key2).all()

  >>> f = hk.vmap(lambda _: hk.next_rng_key(), split_rng=True)
  >>> key1, key2 = f(x)
  >>> assert not (key1 == key2).all()

  Random numbers in Haiku are typically used for two things, firstly for
  initialising model parameters, and secondly for creating random samples as
  part of the forward pass of a neural network (e.g. for dropout). If you are
  using :func:`vmap` with a module that uses Haiku random keys for both (e.g.
  you don't pass keys explicitly into the network), then it is quite likely that
  you will want to vary the value of ``split_rng`` depending on whether we are
  initalizing (e.g. creating model parameters) or applying the model. An easy
  way to do this is to set ``split_rng=(not hk.running_init())``.

  For more advanced use cases, such as mapping module parameters, we suggest
  users instead use :func:`lift` or :func:`~haiku.experimental.transparent_lift`
  in combination with :func:`jax.vmap`.

  Args:
    fun: See :func:`jax.vmap`.
    in_axes: See :func:`jax.vmap`.
    out_axes: See :func:`jax.vmap`.
    axis_name: See :func:`jax.vmap`.
    axis_size: See :func:`jax.vmap`.
    split_rng: Controls whether random key APIs in Haiku (e.g.
      :func:`next_rng_key`) return different (aka. the internal key is split
      before calling your mapped function) or the same (aka. the internal key
      is broadcast before calling your mapped fucntion) key. See the docstring
      for examples.

  Returns:
    See :func:`jax.vmap`.
  """

  if not jax.tree.leaves(in_axes):
    raise ValueError(
        f"{fun.__name__} must have at least one non-None value in in_axes "
        "to use with `hk.vmap`.")

  params_axes = state_axes = None
  rng_axes = (0 if split_rng else None)
  haiku_state_axes = InternalState(params_axes, state_axes, rng_axes)
  in_axes = list_to_tuple(in_axes), haiku_state_axes
  out_axes = out_axes, haiku_state_axes

  @functools.wraps(fun)
  def pure_fun(args, state_in):
    if split_rng:
      # NOTE: In the case of split_rng we recieve an RNG key (rather than the
      # internal state of a PRNGSequence) so we need to construct that here.
      rng = base.PRNGSequence(state_in.rng).internal_state
      state_in = InternalState(state_in.params, state_in.state, rng)

    with temporary_internal_state(state_in), \
         base.push_jax_trace_level():
      out = fun(*args)
      state_out = difference(state_in, internal_state())
      return out, state_out

  @functools.wraps(fun)
  def mapped_fun(*args):
    base.assert_context("vmap")

    mapped_pure_fun = jax.vmap(pure_fun, in_axes=in_axes, out_axes=out_axes,
                               axis_name=axis_name, axis_size=axis_size)
    state = internal_state()

    if split_rng:
      # Need to take a new key and split.
      num = get_mapped_axis_size(args, in_axes[0])
      rng = base.next_rng_keys(num)
      state = internal_state()  # Needed since we mutated internal RNG.
      saved_rng = state.rng
      state = InternalState(state.params, state.state, rng)

    try:
      out, state = mapped_pure_fun(args, state)
    except ValueError as err:
      if split_rng and not base.params_frozen() and "out_axes" in str(err):
        # TODO(lenamartens): add error for state too.
        raise ValueError("hk.vmap does not support setting split_rng to True "
                         "during initialization because it assumes parameters "
                         "are always shared along the mapped dimension. "
                         "Consider switching the value of `split_rng` to False "
                         "during initialization through "
                         "`split_rng=(not hk.running_init())`."
                         ) from err
      else:
        raise err

    if split_rng:
      state = InternalState(state.params, state.state, saved_rng)

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
    with temporary_internal_state(state), \
         base.push_jax_trace_level():
      val = body_fun(val)
      reserve_up_to_full_rng_block()
      state = internal_state()
      return val, state

  reserve_up_to_full_rng_block()
  init_val = (init_val, internal_state())
  val, state = jax.lax.while_loop(pure_cond_fun, pure_body_fun, init_val)
  update_internal_state(state)
  return val


def eval_shape(fun, *args, **kwargs):
  """Equivalent to jax.eval_shape with any changed Haiku state discarded."""
  if not base.inside_transform():
    raise ValueError(
        "hk.eval_shape() should not be used outside of hk.transform(). "
        "Use jax.eval_shape() instead.")

  @functools.wraps(fun)
  def stateless_fun(state, *args, **kwargs):
    with temporary_internal_state(state), \
         base.push_jax_trace_level():
      out = fun(*args, **kwargs)
      # Don't return changed state
      return out

  out_shape = jax.eval_shape(stateless_fun, internal_state(), *args, **kwargs)
  return out_shape
