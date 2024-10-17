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

from collections.abc import Callable, Mapping
import inspect
from typing import Any, NamedTuple, Optional, TypeVar, Union

from haiku._src import analytics
from haiku._src import base
from haiku._src import data_structures
from haiku._src import typing
import jax


# If you are forking replace this with `import haiku as hk`.
# pylint: disable=invalid-name
class hk:
  PRNGSequence = base.PRNGSequence
  Params = typing.Params
  State = typing.State
  MutableParams = typing.MutableParams
  MutableState = typing.MutableState
# pylint: enable=invalid-name
# TODO(slebedev): This makes the module non-forkable.
PRNGKey = typing.PRNGKey
del typing

T = TypeVar("T")

# TODO(b/161684853): Use protocols for transform if/when PEP-612 is implemented.
# https://www.python.org/dev/peps/pep-0612/


def sig_replace_leading_parameters(
    s: inspect.Signature, n: int, new_params: list[inspect.Parameter]
) -> inspect.Signature:
  """Replace the first n positional parameters of a signature."""
  p = list(s.parameters.values())
  for i in range(n):
    if i >= len(p) or p[i].kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    }:
      break  # not enough arguments (or args in VARARGS that can't be counted)
  else:
    i = n
  return inspect.Signature(
      parameters=new_params + p[i:], return_annotation=s.return_annotation,
      __validate_parameters__=False)


def sig_remove_state(s: inspect.Signature) -> inspect.Signature:
  """Remove hk.State from the return type of a signature."""
  ret = s.return_annotation
  # Extract the tuple element types from `typing._GenericAlias` or
  # `types.GenericAlias`.
  ret_generic = getattr(ret, "__origin__", None)
  ret_type_args = getattr(ret, "__args__", ())
  if ret_generic is tuple and len(ret_type_args) == 2:
    ret = ret_type_args[0]
  else:
    ret = Any
  return inspect.Signature(
      parameters=list(s.parameters.values()), return_annotation=ret,
      __validate_parameters__=False)


def sig_add_state(s: inspect.Signature) -> inspect.Signature:
  """Add hk.State to the return type of a signature."""
  if s.return_annotation is inspect.Parameter.empty:
    ret = Any
  else:
    ret = s.return_annotation
  return inspect.Signature(
      parameters=list(s.parameters.values()),
      return_annotation=tuple[ret, hk.State],
      __validate_parameters__=False)


class Transformed(NamedTuple):
  """Holds a pair of pure functions.

  Attributes:
    init: A pure function: ``params = init(rng, *a, **k)``
    apply: A pure function: ``out = apply(params, rng, *a, **k)``
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., hk.MutableParams]

  # Args: [Params, Optional[PRNGKey], ...]
  apply: Callable[..., Any]


class TransformedWithState(NamedTuple):
  """Holds a pair of pure functions.

  Attributes:
    init: A pure function: ``params, state = init(rng, *a, **k)``
    apply: A pure function: ``out, state = apply(params, state, rng, *a, **k)``
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., tuple[hk.MutableParams, hk.MutableState]]

  # Args: [hk.Params, hk.State, Optional[PRNGKey], ...]
  apply: Callable[..., tuple[Any, hk.MutableState]]


def to_prng_sequence(rng, err_msg) -> hk.PRNGSequence | None:
  if rng is not None:
    try:
      rng = hk.PRNGSequence(rng)
    except Exception as e:
      raise ValueError(
          f"{err_msg}. The object was of type {type(rng)}: {rng}") from e
  return rng


RNG_ERROR_TPL = ("{f} must be called with an RNG as the {position} argument, "
                 "the required signature is: `{signature}`")
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
  >>> print(f.apply(params, rng, x))
  [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

  Args:
    f: A transformed function.

  Returns:
    A transformed function that does not take or return state.
  """

  def init_fn(*args, **kwargs) -> hk.MutableParams:
    params, state = f.init(*args, **kwargs)
    if state:
      raise base.NonEmptyStateError(
          "If your transformed function uses `hk.{get,set}_state` then use "
          "`hk.transform_with_state`.")
    return params

  init_fn.__signature__ = sig_remove_state(inspect.signature(f.init))

  def apply_fn(params, *args, **kwargs):
    if "state" in kwargs:
      raise TypeError(
          "Haiku transform adds three arguments (params, state, rng) to apply. "
          "If the functions you are transforming use the same names you must "
          "pass them positionally (e.g. `f.apply(.., my_state)` and not by "
          "name (e.g. `f.apply(.., state=my_state)`)")

    out, state = f.apply(params, None, *args, **kwargs)
    if state:
      raise base.NonEmptyStateError(
          "If your transformed function uses `hk.{get,set}_state` then use "
          "`hk.transform_with_state`.")
    return out

  apply_fn.__signature__ = sig_remove_state(
      sig_replace_leading_parameters(
          inspect.signature(f.apply), 2, [
              inspect.Parameter(
                  "params",
                  inspect.Parameter.POSITIONAL_OR_KEYWORD,
                  annotation=Optional[hk.Params])
          ]))

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
  >>> print(out)
  [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
  >>> state
  {}

  Args:
    f: A transformed function.

  Returns:
    A transformed function that does accepts and returns state.
  """

  def init_fn(*args, **kwargs) -> tuple[hk.MutableParams, hk.MutableState]:
    params = f.init(*args, **kwargs)
    state = data_structures.to_haiku_dict({})
    return params, state

  init_fn.__signature__ = sig_add_state(inspect.signature(f.init))

  def apply_fn(
      params: hk.Params, state: hk.State | None, *args, **kwargs
  ) -> tuple[Any, hk.MutableState]:
    del state
    out = f.apply(params, *args, **kwargs)
    state = data_structures.to_haiku_dict({})
    return out, state

  apply_fn.__signature__ = sig_add_state(sig_replace_leading_parameters(
      inspect.signature(f.apply), 1, [
          inspect.Parameter(
              "param",
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              annotation=hk.Params),
          inspect.Parameter(
              "state",
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              annotation=hk.Params)
      ]))

  tie_in_original_fn(f, init_fn, apply_fn)

  return TransformedWithState(init=init_fn, apply=apply_fn)


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
  {'my_module': {'w': ...Array(0., dtype=float32)},
   'my_module_1': {'w': ...Array(0., dtype=float32)}}

  You can then apply the function with the given parameters by calling
  ``apply`` (note that since we don't use Haiku's random number APIs to apply
  our network we pass ``None`` as an RNG key):

  >>> print(f.apply(params, None, 1))
  2.0

  It is expected that your program will at some point produce updated parameters
  and you will want to re-apply ``apply``. You can do this by calling ``apply``
  with different parameters:

  >>> new_params = {"my_module": {"w": jnp.array(2.)},
  ...               "my_module_1": {"w": jnp.array(3.)}}
  >>> print(f.apply(new_params, None, 2))
  9.0

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
        "The apply_rng argument has been removed and hk.transform "
        "now *always* applies an rng.\n"
        "Replace hk.transform(..., apply_rng=False) with "
        "hk.without_apply_rng(hk.transform(...)).\n"
        "Replace hk.transform(..., apply_rng=True) with hk.transform(...).")

  return without_state(transform_with_state(f))

COMPILED_FN_TYPES = (jax.lib.xla_extension.PjitFunction,
                     jax.lib.xla_extension.PmapFunction)  # pytype: disable=name-error


def check_not_jax_transformed(f):
  # TODO(tomhennigan): Consider `CompiledFunction = type(jax.jit(lambda: 0))`.
  if isinstance(f, COMPILED_FN_TYPES):
    raise ValueError("A common error with Haiku is to pass an already jit "
                     "(or pmap) decorated function into hk.transform (e.g. "
                     "`hk.transform(jax.jit(f)))`. You should instead jit/pmap "
                     "the init or apply function you get back from Haiku (e.g. "
                     "`jax.jit(hk.transform(f).apply)`).\n\n"
                     "This is because the function you pass into hk.transform "
                     "is not a pure function (because you don't explicitly "
                     "pass in/out params/rng). jit and pmap require you to "
                     "pass in a pure function (such as the init or apply "
                     "functions Haiku gives you back from hk.transform).")


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
  >>> print(counter)
  9

  Args:
    f: A function closing over :class:`Module` instances.

  Returns:
    A :class:`TransformedWithState` tuple with ``init`` and ``apply`` pure
    functions.
  """
  analytics.log_once("transform_with_state")
  check_not_jax_transformed(f)

  unexpected_tracer_hint = (
      "An UnexpectedTracerError was raised while inside a Haiku transformed "
      "function (see error above).\n"
      "Hint: are you using a JAX transform or JAX control-flow function "
      "(jax.vmap/jax.lax.scan/...) inside a Haiku transform? You might want to use "
      "the Haiku version of the transform instead (hk.vmap/hk.scan/...).\n"
      "See https://dm-haiku.readthedocs.io/en/latest/notebooks/transforms.html "
      "on why you can't use JAX transforms inside a Haiku module.")

  f_sig = inspect.signature(f)

  def init_fn(
      rng: PRNGKey | int | None,
      *args,
      **kwargs,
  ) -> tuple[hk.MutableParams, hk.MutableState]:
    """Initializes your function collecting parameters and state."""
    rng = to_prng_sequence(rng, err_msg=INIT_RNG_ERROR)
    with base.new_context(rng=rng) as ctx:
      try:
        f(*args, **kwargs)
      except jax.errors.UnexpectedTracerError as e:
        raise jax.errors.UnexpectedTracerError(unexpected_tracer_hint) from e
    return ctx.collect_params(), ctx.collect_initial_state()

  init_fn.__signature__ = inspect.Signature(
      parameters=[
          inspect.Parameter(
              "rng",
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              annotation=Optional[Union[PRNGKey, int]],
          ),
      ]
      + list(f_sig.parameters.values()),
      return_annotation=tuple[hk.Params, hk.State],
      __validate_parameters__=False,
  )

  def apply_fn(
      params: hk.Params | None,
      state: hk.State | None,
      rng: PRNGKey | int | None,
      *args,
      **kwargs,
  ) -> tuple[Any, hk.MutableState]:
    """Applies your function injecting parameters and state."""
    uses_state = state is not None
    params = check_mapping("params", params)
    state = check_mapping("state", state)
    rng = to_prng_sequence(
        rng,
        err_msg=(APPLY_RNG_STATE_ERROR if uses_state else APPLY_RNG_ERROR))
    with base.new_context(params=params, state=state, rng=rng) as ctx:
      try:
        out = f(*args, **kwargs)
      except jax.errors.UnexpectedTracerError as e:
        raise jax.errors.UnexpectedTracerError(unexpected_tracer_hint) from e
    return out, ctx.collect_state()

  apply_fn.__signature__ = sig_add_state(inspect.Signature(
      parameters=[
          inspect.Parameter("params", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=Optional[hk.Params]),
          inspect.Parameter("state", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=Optional[hk.State]),
          inspect.Parameter("rng", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            annotation=Optional[Union[PRNGKey, int]]),
      ] + list(f_sig.parameters.values()),
      return_annotation=f_sig.return_annotation,
      __validate_parameters__=False
  ))

  tie_in_original_fn(f, init_fn, apply_fn)

  return TransformedWithState(init_fn, apply_fn)


def tie_in_original_fn(f, init_fn, apply_fn):
  # EXPERIMENTAL: Expose the original function as a private attribute.
  if isinstance(f, (Transformed, TransformedWithState)):
    f = getattr(f.init, "_original_fn")
  init_fn._original_fn = f  # pylint: disable=protected-access
  apply_fn._original_fn = f  # pylint: disable=protected-access


def get_original_fn(f: Transformed | TransformedWithState | Callable[..., Any]):
  if isinstance(f, (Transformed, TransformedWithState)):
    f = f.init
  return getattr(f, "_original_fn")


def check_mapping(name: str, mapping: T | None) -> T:
  """Cleans inputs to apply_fn, providing better errors."""
  if mapping is None:
    # Convert None to empty dict.
    mapping = dict()
  if not isinstance(mapping, Mapping):
    if type(mapping).__name__ == "_DictWrapper":
      # TensorFlow's checkpointing infrastructure replaces `dict` instances on
      # `tf.Module`s with a type that is not a `Mapping` instance.
      return mapping

    raise TypeError(f"{name} argument does not appear valid. It should be a "
                    f"mapping but is of type {type(mapping)}. "
                    "For reference the parameters for apply are "
                    "`apply(params, rng, ...)`` for `hk.transform` and "
                    "`apply(params, state, rng, ...)` for "
                    "`hk.transform_with_state`.\n"
                    f"The argument was: {mapping!r}.")
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
