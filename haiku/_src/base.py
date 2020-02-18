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
"""Base Haiku module."""

import collections
import contextlib
import functools
from typing import (Any, Callable, Iterator, MutableMapping, NamedTuple,
                    Optional, Text, Tuple, Set, TypeVar, Union)
import warnings

from haiku._src import analytics
from haiku._src import data_structures
from haiku._src.typing import (Shape, DType, ParamName, Initializer, Params,  # pylint: disable=g-multiple-import
                               State, ParamCreator, PRNGKey, PRNGSeed)
import jax
import jax.numpy as jnp
import numpy as np

namedtuple = collections.namedtuple
frozendict = data_structures.frozendict
Stack = data_structures.Stack
ThreadLocalStack = data_structures.ThreadLocalStack

T = TypeVar("T")

ModuleState = namedtuple("ModuleState", ("module", "method_name"))
StatePair = namedtuple("StatePair", ("initial", "current"))
MutableParams = MutableMapping[Text, MutableMapping[ParamName, jnp.ndarray]]
MutableState = MutableMapping[Text, MutableMapping[Text, StatePair]]

# TODO(tomhennigan) Add types to attributes once state/apply_rng are removed.
Transformed = namedtuple("Transformed", ("init", "apply"))


class TransformedWithState(NamedTuple):
  """Holds a pair of pure functions."""

  # TODO(tomhennigan): Use protocols to describe *args when we are 3.8+.
  # https://www.python.org/dev/peps/pep-0544/#callback-protocols

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., Tuple[Params, State]]

  # Args: [Params, State, Optional[PRNGKey], ...]
  apply: Callable[..., Tuple[Any, State]]

# TODO(tomhennigan) Should creator_stack be part of frame?
frame_stack = ThreadLocalStack()  # type: ThreadLocalStack["Frame"]
creator_stack = ThreadLocalStack()  # type: ThreadLocalStack[ParamCreator]


class Frame(NamedTuple):
  """A frame represents all of the per-transform values in Haiku."""

  # JAX values.
  params: Union[Params, MutableParams]
  state: Optional[MutableState]
  rng_stack: Stack[Optional["PRNGSequence"]]

  # Pure python values.
  module_stack: Stack[ModuleState]
  counter_stack: Stack[collections.Counter]
  used_names_stack: Stack[Set[Text]]

  @property
  def params_frozen(self):
    return isinstance(self.params, data_structures.frozendict)

  @classmethod
  def create(cls, params, state, rng: Optional["PRNGSequence"]):
    """Creates a new frame."""
    frame = Frame(params=params,
                  state=state,
                  rng_stack=Stack(),
                  module_stack=Stack(),
                  counter_stack=Stack(),
                  used_names_stack=Stack())
    frame.rng_stack.push(rng)
    frame.counter_stack.push(collections.Counter())
    frame.used_names_stack.push(set())
    return frame

  def evolve(self, params, state, rng):
    rng_stack = self.rng_stack.clone()
    rng_stack.push(rng)
    return Frame(params=params,
                 state=state,
                 rng_stack=rng_stack,
                 module_stack=self.module_stack.clone(),
                 counter_stack=self.counter_stack.clone(),
                 used_names_stack=self.used_names_stack.clone())

  @contextlib.contextmanager
  def module(self, module_state: ModuleState):
    with self.module_stack(module_state), \
         self.counter_stack(collections.Counter()), \
         self.used_names_stack(set()):
      yield


current_frame = frame_stack.peek


def safe_get_module_name(module) -> Text:
  # TODO(tomhennigan) Module specific code should be part of `module.py`.
  if not hasattr(module, "module_name"):
    raise ValueError("The super constructor must be called before you create "
                     "parameters or submodules.")
  return module.module_name


def current_bundle_name():
  frame = current_frame()
  if frame.module_stack:
    module = frame.module_stack.peek().module
    module_name = safe_get_module_name(module)
    return module_name
  else:
    # Any parameters defined outside an `hk.Module` are put in the same group.
    return "~"


def assert_transformed(public_symbol_name):
  if not frame_stack:
    raise ValueError(
        "`hk.{}` must be used as part of an `hk.transform`".format(
            public_symbol_name))


def get_parameter(
    name: ParamName,
    shape: Shape,
    dtype: DType = jnp.float32,
    init: Initializer = None,
) -> jnp.ndarray:
  """Creates or reuses a parameter for the given transformed function.

  >>> hk.get_parameter("w", [], init=jnp.ones)
  DeviceArray(1., dtype=float32)

  Parameters within the same `hk.transform` and/or `hk.Module` with the same
  name have the same value:

  >>> w1 = hk.get_parameter("w", [], init=jnp.zeros)
  >>> w2 = hk.get_parameter("w", [], init=jnp.zeros)
  >>> assert w1 is w2

  Args:
    name: A name for the parameter.
    shape: The shape of the parameter.
    dtype: The dtype of the parameter.
    init: A callable of shape, dtype to generate an initial value for the
      parameter.

  Returns:
    A jnp.ndarray with the parameter of the given shape.
  """
  assert_transformed("get_parameter")
  assert init is not None, "Initializer must be specified."

  bundle_name = current_bundle_name()
  frame = current_frame()

  if frame.params_frozen and bundle_name not in frame.params:
    raise ValueError(
        "Unable to retrieve parameter {!r} for module {!r}. "
        "All parameters must be created as part of `init_fn`.".format(
            name, bundle_name))

  params = frame.params[bundle_name]
  param = params.get(name)
  if param is None:
    if frame.params_frozen:
      raise ValueError(
          "Unable to retrieve parameter {!r} for module {!r}. "
          "All parameters must be created as part of `init_fn`.".format(
              name, bundle_name))

    fq_name = bundle_name + "/" + name
    param = create_parameter(fq_name, shape, dtype, init)
    params[name] = param  # pytype: disable=unsupported-operands

  assert param.shape == tuple(shape), (
      "{!r} with shape {!r} does not match shape={!r} dtype={!r}".format(
          param, param.shape, shape, dtype))

  return param


def create_parameter(
    original_name: ParamName,
    shape: Shape,
    dtype: DType = jnp.float32,
    init: Initializer = None,
) -> jnp.ndarray:
  """Creates a parameter by running user defined creators then init.

  >>> def fp16_creator(next_creator, name, shape, dtype):
  ...   return next_creator(name, shape, jnp.float16)

  >>> with hk.custom_creator(fp16_creator):
  ...   w = hk.get_parameter("w", [], jnp.float32, init=jnp.ones)
  >>> w.dtype
  dtype('float16')

  Args:
    original_name: Name of the parameter, including parent module name.
    shape: The shape of the parameter.
    dtype: The dtype of the parameter.
    init: A callable of shape, dtype to generate an initial value for the
      parameter.

  Returns:
    A jnp.ndarray with the parameter of the given shape/dtype.
  """
  if not creator_stack:
    return init(shape, dtype)

  def next_creator(name, shape, dtype, init):
    if name != original_name:
      raise ValueError(
          "Modifying variable `name` in a custom creator is not supported.")

    if creator_stack_copy:
      return creator_stack_copy.popleft()(name, shape, dtype, init)
    else:
      return init(shape, dtype)

  creator_stack_copy = creator_stack.map(
      lambda c: functools.partial(c, next_creator))

  return creator_stack_copy.popleft()(original_name, shape, dtype, init)


def custom_creator(creator: ParamCreator):
  """Registers a custom parameter creator.

  When new parameters are created via `hk.get_parameter` we first run custom
  creators passing user defined values through. For example:

  >>> def zeros_creator(next_creator, name, shape, dtype, init):
  ...   return next_creator(name, shape, dtype, init=jnp.zeros)

  >>> with hk.custom_creator(zeros_creator):
  ...   w = hk.get_parameter("w", [], jnp.float32, jnp.ones)
  >>> w
  DeviceArray(0., dtype=float32)

  Args:
    creator: A parameter creator.

  Returns:
    Context manager under which the creator is active.
  """
  return creator_stack(creator)


def mk_init_fn(f: Callable[..., Any]) -> Callable[..., Tuple[Params, State]]:
  """Rewrites `f` to return initial values for parameters and state.

  See :func:`transform` for more details.

  The signature of the resulting function is:

      init_fn(rng, ...) -> params, state

  Args:
    f: `f(*args, **kwargs) -> Out`

  Returns:
    A function that given args/kwargs returns the initial state for f.
  """
  def init_fn(
      rng: Optional[Union[PRNGKey, PRNGSeed]],
      *args,
      **kwargs,
  ):
    """Initializes your function collecting parameters and state."""
    params = collections.defaultdict(dict)
    state = collections.defaultdict(dict)

    frame = Frame.create(params=params,
                         state=state,
                         rng=(PRNGSequence(rng) if rng is not None else None))

    with frame_stack(frame):
      f(*args, **kwargs)

    params = data_structures.to_immutable_dict(params)
    state = _extract_state(state, initial=True)
    return params, state

  return init_fn


def mk_apply_fn(f: Callable[..., T]) -> Callable[..., Tuple[T, State]]:
  """Rewrites `f` to accept parameters, state and rng as input.

  See :func:`transform` for more details.

  Args:
    f: `f(...) -> Out`

  Returns:
    A function that accepts parameters/state as input and computes f(*a, **k).
  """
  def apply_fn(
      params: Params,
      state: State,
      rng: Optional[Union[PRNGKey, PRNGSeed]],
      *args,
      **kwargs,
  ):
    """Applies your function injecting parameters and state."""
    # TODO(tomhennigan) Remove support for `None` params (used in tests).
    params = data_structures.to_immutable_dict(params)
    state = {m: {k: StatePair(v, v) for k, v in p.items()}
             for m, p in state.items()}

    frame = Frame.create(
        params=params,
        state=state,
        rng=(PRNGSequence(rng) if rng is not None else None))

    with frame_stack(frame):
      out = f(*args, **kwargs)

    state = _extract_state(state, initial=False)
    return out, state

  return apply_fn


# TODO(tomhennigan) Should be TransformedWithState -> Transformed.
def without_state(
    f: Union[Transformed, TransformedWithState],
) -> Transformed:
  """Wraps a transformed tuple and ignores state in/out.

  >>> def f(x):
  ...   mod = hk.Linear(10)
  ...   return mod(x)

  >>> f = hk.without_state(hk.transform_with_state(f))

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
      raise ValueError("Function wrapped with `hk.without_state` used state.")
    return params

  def apply_fn(params, *args, **kwargs):
    out, state = f.apply(params, {}, *args, **kwargs)
    if state:
      raise ValueError("Function wrapped with `hk.without_state` used state.")
    return out

  return Transformed(init=init_fn, apply=apply_fn)


def without_apply_rng(
    f: Union[Transformed, TransformedWithState],
) -> Transformed:

  def apply_fn(params, state, *args, **kwargs):
    return f.apply(params, state, None, *args, **kwargs)

  return Transformed(init=f.init, apply=apply_fn)


# TODO(tomhennigan) Remove apply_rng and state.
def transform(
    f,
    apply_rng=False,
    state=False,
) -> Transformed:
  """Transforms a function using Haiku modules into a pair of pure functions.

  The first thing to do is to define a `Module`. A module encapsulates some
  parameters and a computation on those parameters:

  >>> class MyModule(hk.Module):
  ...   def __call__(self, x):
  ...     w = hk.get_parameter("w", [], init=jnp.zeros)
  ...     return x + w

  Next, define some function that creates and applies modules. We use
  `hk.transform` to transform that function into a pair of functions that allow
  us to lift all the parameters out of the function (`f.init`) and apply the
  function with a given set of parameters (`f.apply`):

  >>> def f(x):
  ...   a = MyModule()
  ...   b = MyModule()
  ...   return a(x) + b(x)

  >>> f = hk.transform(f)

  To get the initial state of the module call the `init_fn` with an example
  input:

  >>> params = f.init(None, 1)
  >>> params
  frozendict({
    'my_module': frozendict({'w': DeviceArray(0., dtype=float32)}),
    'my_module_1': frozendict({'w': DeviceArray(0., dtype=float32)}),
  })

  You can then apply the function with the given parameters by calling
  `f.apply`:

  >>> f.apply(params, 1)
  DeviceArray(2., dtype=float32)

  It is expected that your program will at some point produce updated parameters
  and you will want to re-apply `f.apply`. You can do this by calling `f.apply`
  with different parameters:

  >>> new_params = {"my_module": {"w": jnp.array(2.)},
  ...               "my_module_1": {"w": jnp.array(3.)}}
  >>> f.apply(new_params, 2)
  DeviceArray(9., dtype=float32)

  If your transformed function needs to maintain internal state (e.g. moving
  averages in batch norm) then see :func:`transform_with_state`.

  Args:
    f: A function closing over `Module` instances.
    apply_rng: Whether `apply` should accept `rng` as an argument.
    state: *Deprecated:* use `hk.transform_with_state`.

  Returns:
    A named tuple with `init` and `apply` pure functions.
  """
  analytics.log_once("transform")

  if state:
    warnings.warn(
        "Prefer using hk.transform_with_state(f) vs. passing state=True.",
        DeprecationWarning)

  if apply_rng:
    warnings.warn("Apply_rng will soon be removed and defaulted to True",
                  DeprecationWarning)

  pair = transform_with_state(f)  # type: Transformed
  if not apply_rng:
    pair = without_apply_rng(pair)
  if not state:
    pair = without_state(pair)
  return pair


def transform_with_state(f) -> TransformedWithState:
  """Transforms a function using Haiku modules into a pair of pure functions.

  See :func:`transform` for general details on Haiku transformations.

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
    f: A function closing over `Module` instances.

  Returns:
    A named tuple with `init` and `apply` properties.
  """
  analytics.log_once("transform_with_state")
  return TransformedWithState(mk_init_fn(f), mk_apply_fn(f))


class PRNGSequence(Iterator[PRNGKey]):
  """Iterator of PRNGKeys.

      >>> seq = hk.PRNGSequence(42)  # OR pass a jax.random.PRNGKey
      >>> key1 = next(seq)
      >>> key2 = next(seq)
      >>> assert key1 is not key2
  """

  def __init__(self, key_or_seed: Union[PRNGKey, int]):
    if np.isscalar(key_or_seed):
      key = jax.random.PRNGKey(key_or_seed)
    else:
      key = key_or_seed
    self._key = key

  def peek(self):
    return self._key

  def replace(self, key: PRNGKey):
    self._key = key

  def __next__(self) -> PRNGKey:
    key, subkey = jax.random.split(self._key)
    self._key = key
    return subkey

  next = __next__  # For Python 2.


def next_rng_key() -> PRNGKey:
  """Returns a unique `PRNGKey` split from the current global key."""
  assert_transformed("next_rng_key")

  rng_seq = current_frame().rng_stack.peek()
  if rng_seq is None:
    raise ValueError("You must pass a non-None PRNGKey to init and/or apply "
                     "if you make use of random numbers.")

  return next(rng_seq)


def _extract_state(state: MutableState, *, initial) -> State:
  state = {m: {k: (v.initial if initial else v.current) for k, v in p.items()}
           for m, p in state.items()}
  state = data_structures.to_immutable_dict(state)
  return state


def get_state(name: ParamName,
              shape: Optional[Shape] = None,
              dtype: Optional[DType] = jnp.float32,
              init: Optional[Initializer] = None) -> jnp.ndarray:
  """Gets the current value for state with an optional initial value."""
  assert_transformed("get_state")
  state = current_frame().state[current_bundle_name()]
  value = state.get(name, None)
  if value is None:
    if init is None:
      raise ValueError(
          "No value for {!r} in {!r}, perhaps set an init function?".format(
              name, current_bundle_name()))
    if shape is None or dtype is None:
      raise ValueError(
          "Must provide shape and dtype to initialize {!r} in {!r}.".format(
              name, current_bundle_name()))

    initial = init(shape, dtype)
    value = state[name] = StatePair(initial, initial)
  return value.current


def set_state(name: ParamName, value):
  """Sets the current value for state."""
  assert_transformed("set_state")
  state = current_frame().state[current_bundle_name()]
  if name in state:
    initial, _ = state[name]
    current = value
  else:
    initial = current = value
  state[name] = StatePair(initial, current)


def with_rng(key: PRNGKey):
  """Provides a new sequence for `hk.next_rng_key` to draw from.

  When `hk.next_rng_key` is called, it draws a new key from the PRNGSequence
  defined by the input key to the transformed function. This context manager
  overrides the sequence for the duration of the scope.

  >>> with hk.with_rng(jax.random.PRNGKey(428)):
  ...   s = jax.random.uniform(hk.next_rng_key(), ())
  >>> s
  DeviceArray(0.501871, dtype=float32)

  Args:
    key: The key to seed the sequence with.

  Returns:
    Context manager under which the given sequence is active.
  """
  return current_frame().rng_stack(PRNGSequence(key))
