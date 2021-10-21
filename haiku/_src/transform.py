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

import types
from typing import Any, Callable, Mapping, NamedTuple, Optional, Tuple, TypeVar, Union

from haiku._src import analytics
from haiku._src import base
from haiku._src import data_structures
from haiku._src import typing
import jax

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.PRNGSequence = base.PRNGSequence
hk.Params = typing.Params
hk.State = typing.State
PRNGKey = typing.PRNGKey
del typing

T = TypeVar("T")

# TODO(b/161684853): Use protocols for transform if/when PEP-612 is implemented.
# https://www.python.org/dev/peps/pep-0612/


class Transformed(NamedTuple):
  """Holds a pair of pure functions.

  Attributes:
    init: A pure function: ``params = init(rng, *a, **k)``
    apply: A pure function: ``out = apply(params, rng, *a, **k)``
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., hk.Params]

  # Args: [Params, Optional[PRNGKey], ...]
  apply: Callable[..., Any]


class TransformedWithState(NamedTuple):
  """Holds a pair of pure functions.

  Attributes:
    init: A pure function: ``params, state = init(rng, *a, **k)``
    apply: A pure function: ``out, state = apply(params, state, rng, *a, **k)``
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., Tuple[hk.Params, hk.State]]

  # Args: [hk.Params, hk.State, Optional[PRNGKey], ...]
  apply: Callable[..., Tuple[Any, hk.State]]


def to_prng_sequence(rng, err_msg) -> Optional[hk.PRNGSequence]:
  if rng is not None:
    try:
      rng = hk.PRNGSequence(rng)
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

  The example below is equivalent to ``f = hk.transform(f)``:

  >>> def f(x):
  ...   mod = hk.Linear(10)
  ...   return mod(x)
  >>> f = hk.without_state(hk.transform_with_state(f))
  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.zeros([1, 1])
  >>> params = f.init(rng, x)
  >>> f.apply(params, rng, x)
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
    if "state" in kwargs:
      raise TypeError(
          "Haiku transform adds three arguments (params, state, rng) to apply. "
          "If the functions you are transforming use the same names you must "
          "pass them positionally (e.g. `f.apply(.., my_state)` and not by "
          "name (e.g. `f.apply(.., state=my_state)`)")

    out, state = f.apply(params, {}, *args, **kwargs)
    if state:
      raise ValueError("If your transformed function uses `hk.{get,set}_state` "
                       "then use `hk.transform_with_state`.")
    return out

  tie_in_original_fn(f, init_fn, apply_fn)

  return Transformed(init=init_fn, apply=apply_fn)


def with_empty_state(f: Transformed) -> TransformedWithState:
  """Wraps a transformed tuple and passes empty state in/out.

  The example below is equivalent to ``f = hk.transform_with_state(f)``:

  >>> def f(x):
  ...   mod = hk.Linear(10)
  ...   return mod(x)
  >>> f = hk.with_empty_state(hk.transform(f))
  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.zeros([1, 1])
  >>> params, state = f.init(rng, x)
  >>> state
  {}
  >>> out, state = f.apply(params, state, rng, x)
  >>> out
  DeviceArray([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)
  >>> state
  {}

  Args:
    f: A transformed function.

  Returns:
    A transformed function that does accepts and returns state.
  """

  def init_fn(*args, **kwargs):
    params = f.init(*args, **kwargs)
    state = data_structures.to_haiku_dict({})
    return params, state

  def apply_fn(params, state, *args, **kwargs):
    del state
    out = f.apply(params, *args, **kwargs)
    state = data_structures.to_haiku_dict({})
    return out, state

  tie_in_original_fn(f, init_fn, apply_fn)

  return TransformedWithState(init=init_fn, apply=apply_fn)


TransformedT = TypeVar("TransformedT", Transformed, TransformedWithState)


def without_apply_rng(f: TransformedT) -> TransformedT:
  """Removes the rng argument from the apply function.

  This is a convenience wrapper that makes the ``rng`` argument to
  ``f.apply`` default to ``None``. This is useful when ``f`` doesn't actually
  use random numbers as part of its computation, such that the ``rng`` argument
  wouldn't be used. Note that if ``f`` `does` use random numbers, this will
  cause an error to be thrown complaining that ``f`` needs a non-None PRNGKey.

  Args:
    f: A transformed function.

  Returns:
    The same transformed function, with a modified ``apply``.
  """
  def check_rng_kwarg(kwargs):
    if "rng" in kwargs:
      raise TypeError(
          "Haiku transform adds three arguments (params, state, rng) to apply. "
          "If the functions you are transforming use the same names you must "
          "pass them positionally (e.g. `f.apply(.., my_rng)` and not by "
          "name (e.g. `f.apply(.., rng=my_rng)`)")

  if isinstance(f, TransformedWithState):
    def apply_fn(params, state, *args, **kwargs):
      check_rng_kwarg(kwargs)
      return f.apply(params, state, None, *args, **kwargs)
    f_new = TransformedWithState(init=f.init, apply=apply_fn)

  elif isinstance(f, Transformed):
    def apply_fn(params, *args, **kwargs):
      check_rng_kwarg(kwargs)
      return f.apply(params, None, *args, **kwargs)
    f_new = Transformed(init=f.init, apply=apply_fn)

  else:
    raise ValueError("Must be called with the result of `hk.transformed` or "
                     f"`hk.transformed_with_state`, actual {type(f)}")

  tie_in_original_fn(f, f_new.init, f_new.apply)
  return f_new


# TODO(tomhennigan) Remove apply_rng.
def transform(f, *, apply_rng=True) -> Transformed:
  """Transforms a function using Haiku modules into a pair of pure functions.

  For a function ``out = f(*a, **k)`` this function returns a pair of two pure
  functions that call ``f(*a, **k)`` explicitly collecting and injecting
  parameter values::

      params = init(rng, *a, **k)
      out = apply(params, rng, *a, **k)

  Note that the ``rng`` argument is typically not required for ``apply`` and
  passing ``None`` is accepted.

  The first thing to do is to define a :class:`Module`. A module encapsulates
  some parameters and a computation on those parameters:

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
  {'my_module': {'w': DeviceArray(0., dtype=float32)},
   'my_module_1': {'w': DeviceArray(0., dtype=float32)}}

  You can then apply the function with the given parameters by calling
  ``apply`` (note that since we don't use Haiku's random number APIs to apply
  our network we pass ``None`` as an RNG key):

  >>> f.apply(params, None, 1)
  DeviceArray(2., dtype=float32)

  It is expected that your program will at some point produce updated parameters
  and you will want to re-apply ``apply``. You can do this by calling ``apply``
  with different parameters:

  >>> new_params = {"my_module": {"w": jnp.array(2.)},
  ...               "my_module_1": {"w": jnp.array(3.)}}
  >>> f.apply(new_params, None, 2)
  DeviceArray(9., dtype=float32, weak_type=True)

  If your transformed function needs to maintain internal state (e.g. moving
  averages in batch norm) then see :func:`transform_with_state`.

  Args:
    f: A function closing over :class:`Module` instances.
    apply_rng: In the process of being removed. Can only value `True`.

  Returns:
    A :class:`Transformed` tuple with ``init`` and ``apply`` pure functions.
  """
  analytics.log_once("transform")

  if not apply_rng:
    raise ValueError(
        "The apply_rng argument has been removed and k.transform "
        "now *always* applies an rng.\n"
        "Replace hk.transform(..., apply_rng=False) with "
        "hk.without_apply_rng(hk.transform(...)).\n"
        "Replace hk.transform(..., apply_rng=True) with hk.transform(...).")

  return without_state(transform_with_state(f))


def transform_with_state(f) -> TransformedWithState:
  """Transforms a function using Haiku modules into a pair of pure functions.

  See :func:`transform` for general details on Haiku transformations.

  For a function ``out = f(*a, **k)`` this function returns a pair of two pure
  functions that call ``f(*a, **k)`` explicitly collecting and injecting
  parameter values and state::

      params, state = init(rng, *a, **k)
      out, state = apply(params, state, rng, *a, **k)

  Note that the ``rng`` argument is typically not required for ``apply`` and
  passing ``None`` is accepted.

  This function is equivalent to :func:`transform`, however it allows you to
  maintain and update internal state (e.g. :class:`ExponentialMovingAverage` in
  :class:`BatchNorm`) via :func:`get_state` and :func:`set_state`:

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
    A :class:`TransformedWithState` tuple with ``init`` and ``apply`` pure
    functions.
  """
  analytics.log_once("transform_with_state")

  unexpected_tracer_hint = (
      "An UnexpectedTracerError was raised while inside a Haiku transformed "
      "function (see error above).\n"
      "Hint: are you using a JAX transform or JAX control-flow function "
      "(jax.vmap/jax.scan/...) inside a Haiku transform? You might want to use "
      "the Haiku version of the transform instead (hk.vmap/hk.scan/...).\n"
      "See https://dm-haiku.readthedocs.io/en/latest/notebooks/transforms.html "
      "on why you can't use JAX transforms inside a Haiku module.")
  def init_fn(
      rng: Optional[Union[PRNGKey, int]],
      *args,
      **kwargs,
  ) -> Tuple[hk.Params, hk.State]:
    """Initializes your function collecting parameters and state."""
    rng = to_prng_sequence(rng, err_msg=INIT_RNG_ERROR)
    with base.new_context(rng=rng) as ctx:
      try:
        f(*args, **kwargs)
      except jax.errors.UnexpectedTracerError as e:
        raise jax.errors.UnexpectedTracerError(unexpected_tracer_hint) from e
    return ctx.collect_params(), ctx.collect_initial_state()

  def apply_fn(
      params: Optional[hk.Params],
      state: Optional[hk.State],
      rng: Optional[Union[PRNGKey, int]],
      *args,
      **kwargs,
  ) -> Tuple[Any, hk.State]:
    """Applies your function injecting parameters and state."""
    params = check_mapping("params", params)
    state = check_mapping("state", state)
    rng = to_prng_sequence(
        rng, err_msg=(APPLY_RNG_STATE_ERROR if state else APPLY_RNG_ERROR))
    with base.new_context(params=params, state=state, rng=rng) as ctx:
      try:
        out = f(*args, **kwargs)
      except jax.errors.UnexpectedTracerError as e:
        raise jax.errors.UnexpectedTracerError(unexpected_tracer_hint) from e
    return out, ctx.collect_state()

  tie_in_original_fn(f, init_fn, apply_fn)

  return TransformedWithState(init_fn, apply_fn)


def tie_in_original_fn(f, init_fn, apply_fn):
  # EXPERIMENTAL: Expose the original function as a private attribute.
  if isinstance(f, (Transformed, TransformedWithState)):
    f = getattr(f.init, "_original_fn")
  init_fn._original_fn = f  # pylint: disable=protected-access
  apply_fn._original_fn = f  # pylint: disable=protected-access


def get_original_fn(f: Union[TransformedT, Callable[..., Any]]):
  if isinstance(f, (Transformed, TransformedWithState)):
    f = f.init
  return getattr(f, "_original_fn")


def check_mapping(name: str, mapping: Optional[T]) -> T:
  """Cleans inputs to apply_fn, providing better errors."""
  # TODO(tomhennigan) Remove support for empty non-Mappings.
  if mapping is None:
    # Convert None to empty dict.
    mapping = dict()
  if not isinstance(mapping, Mapping):
    raise TypeError(f"{name} argument does not appear valid: {mapping!r}. "
                    "For reference the parameters for apply are "
                    "`apply(params, rng, ...)`` for `hk.transform` and "
                    "`apply(params, state, rng, ...)` for "
                    "`hk.transform_with_state`.")
  return mapping


def running_init() -> bool:
  """Return True if running the ``init`` function of a Haiku transform.

  In general you should not need to gate behaviour of your module based on
  whether you are running ``init`` or ``apply``, but sometimes (e.g. when making
  use of JAX control flow) this is required.

  For example, if you want to use :func:`switch` to pick between experts, when
  we run your init function we need to ensure that params/state for all experts
  are created (unconditionally) but during apply we want to conditionally apply
  (and perhaps update the internal state) of only one of our experts:

  >>> experts = [hk.nets.ResNet50(10) for _ in range(5)]
  >>> x = jnp.ones([1, 224, 224, 3])
  >>> if hk.running_init():
  ...   # During init unconditionally create params/state for all experts.
  ...   for expert in experts:
  ...     out = expert(x, is_training=True)
  ... else:
  ...   # During apply conditionally apply (and update) only one expert.
  ...   index = jax.random.randint(hk.next_rng_key(), [], 0, len(experts) - 1)
  ...   out = hk.switch(index, experts, x)

  Returns:
    True if running ``init`` otherwise False.
  """
  base.assert_context("running_init")
  return not base.params_frozen()
