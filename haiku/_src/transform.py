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
"""Base Haiku module."""

from typing import Any, Callable, NamedTuple, Optional, Tuple, TypeVar, Union
import warnings

from haiku._src import analytics
from haiku._src import base
from haiku._src.typing import Params, State, PRNGKey, PRNGSeed  # pylint: disable=g-multiple-import

T = TypeVar("T")

# TODO(tomhennigan): Use protocols to describe *args when we are 3.8+.
# https://www.python.org/dev/peps/pep-0544/#callback-protocols


class Transformed(NamedTuple):
  """Holds a pair of pure functions.

  Attributes:
    init: A pure function: ``params = init(rng, *a, **k)``
    apply: A pure function: ``out = apply(params, rng, *a, **k)``
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., Params]

  # Args: [Params, Optional[PRNGKey], ...]
  apply: Callable[..., Any]


class TransformedWithState(NamedTuple):
  """Holds a pair of pure functions.

  Attributes:
    init: A pure function: ``params, state = init(rng, *a, **k)``
    apply: A pure function: ``out, state = apply(params, state, rng, *a, **k)``
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., Tuple[Params, State]]

  # Args: [Params, State, Optional[PRNGKey], ...]
  apply: Callable[..., Tuple[Any, State]]


def to_prng_sequence(rng, err_msg) -> Optional[base.PRNGSequence]:
  if rng is not None:
    try:
      rng = base.PRNGSequence(rng)
    except Exception as e:
      raise ValueError(err_msg) from e
  return rng

RNG_ERROR_TPL = (
    "{f} must be called with an RNG as the {position} argument, "
    "the required signature is: `{signature}`"
)
INIT_RNG_ERROR = RNG_ERROR_TPL.format(
    f="Init", position="first", signature="init(rng, *a, **k)")
APPLY_RNG_ERROR = RNG_ERROR_TPL.format(
    f="Apply", position="second", signature="apply(params, rng, *a, **k)")
APPLY_RNG_STATE_ERROR = RNG_ERROR_TPL.format(
    f="Apply", position="third", signature="apply(params, state, rng, *a, **k)")


def without_state(f: TransformedWithState) -> Transformed:
  """Wraps a transformed tuple and ignores state in/out.

  >>> def f(x):
  ...   mod = hk.Linear(10)
  ...   return mod(x)

  >>> f = hk.without_state(hk.transform_with_state(f))
  >>> # NOTE: This is equivalent to `f = hk.transform(f, apply_rng=True)`.

  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.zeros([1, 1])
  >>> params = f.init(rng, x)
  >>> out = f.apply(params, rng, x)
  >>> out
  DeviceArray([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)

  Args:
    f: A transformed function.

  Returns:
    A transformed function that does not take or return state.
  """

  def init_fn(*args, **kwargs):
    params, state = f.init(*args, **kwargs)
    if state:
      raise ValueError("If your transformed function uses `hk.{get,set}_state` "
                       "then use `hk.transform_with_state`.")
    return params

  def apply_fn(params, *args, **kwargs):
    out, state = f.apply(params, {}, *args, **kwargs)
    if state:
      raise ValueError("If your transformed function uses `hk.{get,set}_state` "
                       "then use `hk.transform_with_state`.")
    return out

  return Transformed(init=init_fn, apply=apply_fn)


TransformedT = TypeVar("TransformedT", Transformed, TransformedWithState)


def without_apply_rng(f: TransformedT) -> TransformedT:
  """Removes the rng argument from the apply function."""
  if isinstance(f, TransformedWithState):
    def apply_fn(params, state, *args, **kwargs):
      return f.apply(params, state, None, *args, **kwargs)
    return TransformedWithState(init=f.init, apply=apply_fn)

  elif isinstance(f, Transformed):
    def apply_fn(params, *args, **kwargs):
      return f.apply(params, None, *args, **kwargs)
    return Transformed(init=f.init, apply=apply_fn)

  else:
    raise ValueError("Must be called with the reuslt of `hk.transformed` or "
                     f"`hk.transformed_with_state`, actual {type(f)}")


# TODO(tomhennigan) Remove apply_rng.
def transform(f, *, apply_rng=False) -> Transformed:
  """Transforms a function using Haiku modules into a pair of pure functions.

  For a function ``out = f(*a, **k)`` this function returns a pair of two pure
  functions that call ``f(*a, **k)`` explicitly collecting and injecting
  parameter values::

      params = init(rng, *a, **k)
      out = apply(params, rng, *a, **k)

  Note that the ``rng`` argument is typically not required for `apply` and
  passing ``None`` is accepted.

  The first thing to do is to define a `Module`. A module encapsulates some
  parameters and a computation on those parameters:

  >>> class MyModule(hk.Module):
  ...   def __call__(self, x):
  ...     w = hk.get_parameter("w", [], init=jnp.zeros)
  ...     return x + w

  Next, define some function that creates and applies modules. We use
  :func:`transform` to transform that function into a pair of functions that
  allow us to lift all the parameters out of the function (``f.init``) and
  apply the function with a given set of parameters (``f.apply``):

  >>> def f(x):
  ...   a = MyModule()
  ...   b = MyModule()
  ...   return a(x) + b(x)

  >>> f = hk.transform(f)

  To get the initial state of the module call ``init`` with an example input:

  >>> params = f.init(None, 1)
  >>> params
  frozendict({
    'my_module': frozendict({'w': DeviceArray(0., dtype=float32)}),
    'my_module_1': frozendict({'w': DeviceArray(0., dtype=float32)}),
  })

  You can then apply the function with the given parameters by calling
  ``apply``:

  >>> f.apply(params, 1)
  DeviceArray(2., dtype=float32)

  It is expected that your program will at some point produce updated parameters
  and you will want to re-apply ``apply``. You can do this by calling ``apply``
  with different parameters:

  >>> new_params = {"my_module": {"w": jnp.array(2.)},
  ...               "my_module_1": {"w": jnp.array(3.)}}
  >>> f.apply(new_params, 2)
  DeviceArray(9., dtype=float32)

  If your transformed function needs to maintain internal state (e.g. moving
  averages in batch norm) then see :func:`transform_with_state`.

  Args:
    f: A function closing over :class:`Module` instances.
    apply_rng: Whether ``apply`` should accept `rng` as an argument.

  Returns:
    A :class:`Transformed` tuple with ``init`` and ``apply`` pure functions.
  """
  analytics.log_once("transform")

  if not apply_rng:
    warnings.warn("Apply_rng will soon be removed and defaulted to True",
                  DeprecationWarning)

  pair = transform_with_state(f)
  if not apply_rng:
    pair = without_apply_rng(pair)
  return without_state(pair)


def transform_with_state(f) -> TransformedWithState:
  """Transforms a function using Haiku modules into a pair of pure functions.

  See :func:`transform` for general details on Haiku transformations.

  For a function ``out = f(*a, **k)`` this function returns a pair of two pure
  functions that call ``f(*a, **k)`` explicitly collecting and injecting
  parameter values and state::

      params, state = init(rng, *a, **k)
      out, state = apply(params, state, rng, *a, **k)

  Note that the ``rng`` argument is typically not required for `apply` and
  passing ``None`` is accepted.

  This function is equivalent to :func:`transform`, however it allows you to
  maintain and update internal state (e.g. moving averages in batch norm) via
  :func:`get_state` and :func:`set_state`.

  >>> def f():
  ...   counter = hk.get_state("counter", shape=[], dtype=jnp.int32,
  ...                          init=jnp.zeros)
  ...   hk.set_state("counter", counter + 1)
  ...   return counter

  >>> f = hk.transform_with_state(f)

  >>> params, state = f.init(None)
  >>> for _ in range(10):
  ...   counter, state = f.apply(params, state, None)
  >>> counter
  DeviceArray(9, dtype=int32)

  Args:
    f: A function closing over :class:`Module` instances.

  Returns:
    A :class:`TransformedWithState` tuple with `init` and `apply` properties.
  """
  analytics.log_once("transform_with_state")

  def init_fn(
      rng: Optional[Union[PRNGKey, PRNGSeed]],
      *args,
      **kwargs,
  ) -> Tuple[Params, State]:
    """Initializes your function collecting parameters and state."""
    rng = to_prng_sequence(rng, err_msg=INIT_RNG_ERROR)
    with base.new_context(rng=rng) as ctx:
      f(*args, **kwargs)
    return ctx.collect_params(), ctx.collect_initial_state()

  def apply_fn(
      params: Params,
      state: State,
      rng: Optional[Union[PRNGKey, PRNGSeed]],
      *args,
      **kwargs,
  ) -> Tuple[Any, State]:
    """Applies your function injecting parameters and state."""
    rng = to_prng_sequence(
        rng, err_msg=(APPLY_RNG_STATE_ERROR if state else APPLY_RNG_ERROR))
    with base.new_context(params=params, state=state, rng=rng) as ctx:
      out = f(*args, **kwargs)
    return out, ctx.collect_state()

  # EXPERIMENTAL: Expose the original function as a private attribute.
  init_fn._original_fn = f  # pylint: disable=protected-access
  apply_fn._original_fn = f  # pylint: disable=protected-access

  return TransformedWithState(init_fn, apply_fn)
