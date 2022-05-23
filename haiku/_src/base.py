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
import itertools as it
from typing import (Callable, Iterator, Iterable, MutableMapping, NamedTuple,
                    Optional, Set, Tuple, Union, Any, Sequence, Mapping,
                    FrozenSet, TypeVar)
import warnings

from haiku._src import config
from haiku._src import data_structures
from haiku._src.typing import (  # pylint: disable=g-multiple-import
    Initializer,
    Params,
    State,
    Module,
    PRNGKey,
)
import jax
from jax import config as jax_config
from jax import core as jax_core
import jax.numpy as jnp

try:
  from typing import final  # pylint: disable=g-import-not-at-top
except ImportError:
  # Pre Python 3.8.
  final = lambda cls: cls

DEFAULT_PRNG_RESERVE_SIZE = 1

Stack = data_structures.Stack
ThreadLocalStack = data_structures.ThreadLocalStack

ModuleState = collections.namedtuple("ModuleState", ("module", "method_name"))
StatePair = collections.namedtuple("StatePair", ("initial", "current"))
MutableParams = MutableMapping[str, MutableMapping[str, jnp.ndarray]]
MutableState = MutableMapping[str, MutableMapping[str, StatePair]]

# TODO(tomhennigan) Should creator_stack be part of frame?
frame_stack: ThreadLocalStack["Frame"] = ThreadLocalStack()
context_stack: ThreadLocalStack["HaikuContext"] = ThreadLocalStack()
param_creator_stack: ThreadLocalStack["Creator"] = ThreadLocalStack()
state_creator_stack: ThreadLocalStack["Creator"] = ThreadLocalStack()
param_getter_stack: ThreadLocalStack["Getter"] = ThreadLocalStack()
state_getter_stack: ThreadLocalStack["Getter"] = ThreadLocalStack()
state_setter_stack: ThreadLocalStack["Setter"] = ThreadLocalStack()
closure_boundary_stack: ThreadLocalStack[int] = ThreadLocalStack()


class JaxTraceLevel(NamedTuple):
  """Comparable object capturing trace state in JAX."""
  opaque: Any

  @classmethod
  def current(cls):
    # TODO(tomhennigan): Remove once a version of JAX is released incl PR#9423.
    trace_stack = jax_core.thread_local_state.trace_state.trace_stack.stack
    top_type = trace_stack[0].trace_type
    level = trace_stack[-1].level
    sublevel = jax_core.cur_sublevel()
    return JaxTraceLevel(opaque=(top_type, level, sublevel))

frame_ids = it.count()


class Frame(NamedTuple):
  """A frame represents all of the per-transform values in Haiku."""

  # JAX values.
  params: Union[Params, MutableParams]
  state: Optional[MutableState]
  rng_stack: Stack[Optional["PRNGSequence"]]

  # Pure python values.
  freeze_params: bool
  module_stack: Stack[ModuleState]
  counter_stack: Stack[collections.Counter]
  used_names_stack: Stack[Set[str]]
  jax_trace_stack: Stack[JaxTraceLevel]
  frame_id: int

  @property
  def params_frozen(self):
    return self.freeze_params

  @classmethod
  def create(cls, params, state, rng: Optional["PRNGSequence"],
             freeze_params: bool) -> "Frame":
    """Creates a new frame."""
    frame = Frame(params=params,
                  state=state,
                  rng_stack=Stack(),
                  freeze_params=freeze_params,
                  module_stack=Stack(),
                  counter_stack=Stack(),
                  used_names_stack=Stack(),
                  jax_trace_stack=Stack(),
                  frame_id=next(frame_ids))
    frame.rng_stack.push(rng)
    frame.counter_stack.push(collections.Counter())
    frame.used_names_stack.push(set())
    frame.jax_trace_stack.push(JaxTraceLevel.current())
    return frame

  def evolve(self, params, state, rng, *, decoupled=True) -> "Frame":
    """Creates a new frame with JAX state as passed in."""
    rng_stack = self.rng_stack.clone()
    rng_stack.push(rng)
    if decoupled:
      module_stack = self.module_stack.clone()
      counter_stack = self.counter_stack.map(collections.Counter)
      used_names_stack = self.used_names_stack.map(set)
      jax_trace_stack = self.jax_trace_stack.clone()
    else:
      module_stack = self.module_stack
      counter_stack = self.counter_stack
      used_names_stack = self.used_names_stack
      jax_trace_stack = self.jax_trace_stack
    return Frame(params=params,
                 state=state,
                 rng_stack=rng_stack,
                 freeze_params=self.freeze_params,
                 module_stack=module_stack,
                 counter_stack=counter_stack,
                 used_names_stack=used_names_stack,
                 jax_trace_stack=jax_trace_stack,
                 frame_id=next(frame_ids))

  @contextlib.contextmanager
  def module(self, module_state: ModuleState):
    with self.module_stack(module_state), \
         self.counter_stack(collections.Counter()), \
         self.used_names_stack(set()):
      yield


current_frame = frame_stack.peek
current_context = context_stack.peek


class HaikuContext:
  """Collects and injects values for computations."""

  __slots__ = ("__params", "__state", "__rng", "__freeze_params",
               "__expected_stack", "__names", "__counter",
               "__teardown_callbacks")

  def __init__(
      self,
      params: Union[Params, MutableParams],
      state: Union[State, MutableState],
      rng: Optional["PRNGSequence"],
      freeze_params: bool,
  ):
    # NOTE: Using __ vs. _ since these are "really" private (as in using these
    # properties directly could result in broken behaviour).
    self.__params = params
    self.__state = state
    self.__rng = rng
    self.__freeze_params = freeze_params
    self.__expected_stack = ThreadLocalStack()
    self.__names = set()
    self.__counter = collections.Counter()
    self.__teardown_callbacks = []

  def collect_params(self) -> Params:
    return data_structures.to_haiku_dict(self.__params)

  def collect_initial_state(self) -> State:
    return extract_state(self.__state, initial=True)

  def collect_state(self) -> State:
    return extract_state(self.__state, initial=False)

  def add_teardown_callback(self, f):
    self.__teardown_callbacks.append(f)

  def __enter__(self):
    frame = Frame.create(params=self.__params,
                         state=self.__state,
                         rng=self.__rng,
                         freeze_params=self.__freeze_params)
    frame.used_names_stack.push(self.__names)
    frame.counter_stack.push(self.__counter)
    self.__expected_stack.push(frame)
    frame_stack.push(frame)
    context_stack.push(self)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert frame_stack.pop() is self.__expected_stack.pop()
    assert context_stack.pop() is self
    if exc_type is None:
      for callback in self.__teardown_callbacks:
        callback()


def new_context(
    *,
    params: Optional[Params] = None,
    state: Optional[State] = None,
    rng: Optional[Union[PRNGKey, int]] = None,
) -> HaikuContext:
  """Collects the results of hk.{get,set}_{parameter,state} calls.

  >>> with new_context(rng=jax.random.PRNGKey(42)) as ctx:
  ...   mod = hk.nets.MLP([300, 100, 10])
  ...   y1 = mod(jnp.ones([1, 1]))

  >>> assert len(jax.tree_leaves(ctx.collect_params())) == 6

  >>> with ctx:
  ...   y2 = mod(jnp.ones([1, 1]))

  The same module instance in the same context will produce the same value:

  >>> assert (y1 == y2).all()

  Args:
    params: Optional parameter values to inject.
    state: Optional state values to inject.
    rng: Optional rng to inject.

  Returns:
    Context manager which closes over mutable Haiku internal state.
  """
  ddict = functools.partial(collections.defaultdict, dict)

  if params is None:
    params = ddict()
    freeze_params = False
  else:
    params = data_structures.to_haiku_dict(params)
    freeze_params = True

  if state is None:
    state = ddict()
  else:
    state = ddict({m: ddict({k: StatePair(v, v) for k, v in p.items()})
                   for m, p in state.items()})

  if rng is not None and not isinstance(rng, PRNGSequence):
    rng = PRNGSequence(rng)

  return HaikuContext(params, state, rng, freeze_params)


def inside_transform():
  return bool(frame_stack)


def safe_get_module_name(module: Module) -> str:
  """Checks if parameters/state can be safely created before returning the module name."""
  # TODO(tomhennigan) Module specific code should be part of `module.py`.
  if not hasattr(module, "module_name"):
    raise ValueError("The super constructor must be called before you create "
                     "parameters or submodules.")

  if (closure_boundary_stack
      and module._creation_frame_id < closure_boundary_stack.peek()):  # pylint: disable=protected-access
    raise ValueError("You can't functionally close over a module which has "
                     "been intitialized outside of the function wrapped in "
                     "hk.experimental.transparent_lift or "
                     "hk.experimental.layer_stack.\n"
                     "The module you closed over is called "
                     f"'{module.module_name}'.")
  return module.module_name


def current_module_state() -> Optional[ModuleState]:
  frame = current_frame()
  if frame.module_stack:
    return frame.module_stack.peek()
  else:
    return None


@contextlib.contextmanager
def maybe_push_module_state(module_state: Optional[ModuleState]):
  if module_state is not None:
    frame = current_frame()
    with frame.module_stack(module_state):
      yield
  else:
    yield


def current_module() -> Optional[Module]:
  module_state = current_module_state()
  return module_state.module if module_state is not None else None


def current_name() -> str:
  """Returns the currently active module name.

  Outside of a Haiku module (but inside a Haiku transform) this will return
  ``~`` which matches the key in the params/state dict where top level values
  are stored.

  >>> hk.experimental.current_name()
  '~'

  Inside a module this returns the current module name:

  >>> class ExampleModule(hk.Module):
  ...   def __call__(self):
  ...     return hk.experimental.current_name()
  >>> ExampleModule()()
  'example_module'

  Inside a name scope this returns the current name scope:

  >>> with hk.experimental.name_scope('example_name_scope'):
  ...   print(hk.experimental.current_name())
  example_name_scope

  Returns:
    The currently active module or name scope name. If modules or name scopes
    are in use returns ``~``.
  """
  assert_context("experimental.current_name")
  module = current_module()
  if module is not None:
    return safe_get_module_name(module)
  else:
    # Any parameters defined outside an `hk.Module` are put in the same group.
    return "~"

# TODO(tomhennigan): Remove this alias.
current_bundle_name = current_name


def assert_context(public_symbol_name):
  if not frame_stack:
    raise ValueError(
        "`hk.{}` must be used as part of an `hk.transform`".format(
            public_symbol_name))


def params_frozen():
  """Returns true at apply time, false at init time."""
  assert_context("params_frozen")
  return current_frame().params_frozen


@final
class DoNotStore:
  r"""Causes a parameter or state value to not be stored.

  By default, Haiku will put the value returned from
  :func:`~haiku.get_parameter`, :func:`~haiku.get_state` and
  :func:`~haiku.set_state` into the dictionaries returned by ``init``. This is
  not always desirable.

  For example, a user may want to have part of their network come from a
  pretrained checkpoint, and they may want to freeze those values (aka. have
  them not appear in the params dict passed later to ``grad``). You can achieve
  this by manipulating the params dict, however sometimes it is more convenient
  to do this using custom creators/getters/setters.

  Consider the following function:

  >>> def f(x):
  ...   x = hk.Linear(300, name='torso')(x)
  ...   x = hk.Linear(10, name='tail')(x)
  ...   return x

  Imagine you have a pre-trained set of weights for the torso:

  >>> pretrained = {'torso': {'w': jnp.ones([28 * 28, 300]),
  ...                         'b': jnp.ones([300])}}

  First we define a creator, that tells Haiku to not store any parameters that
  are part of the pretrained dict:

  >>> def my_creator(next_creator, shape, dtype, init, context):
  ...   if context.module_name in pretrained:
  ...     return hk.experimental.DO_NOT_STORE
  ...   return next_creator(shape, dtype, init)

  Then we need a getter that provides the parameter value from the pretrained
  dict:

  >>> def my_getter(next_getter, value, context):
  ...   if context.module_name in pretrained:
  ...     assert value is hk.experimental.DO_NOT_STORE
  ...     value = pretrained[context.module_name][context.name]
  ...   return next_getter(value)

  Finally we'll wrap our function in context managers activating our creator and
  getter:

  >>> def f_with_pretrained_torso(x):
  ...   with hk.custom_creator(my_creator), \
  ...        hk.custom_getter(my_getter):
  ...     return f(x)

  You can see that when we run our function we only get parameters from modules
  that were not in the pretrained dict:

  >>> f_with_pretrained_torso = hk.transform(f_with_pretrained_torso)
  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.ones([1, 28 * 28])
  >>> params = f_with_pretrained_torso.init(rng, x)
  >>> assert list(params) == ['tail']

  This value can be used in initialisers, :func:`~haiku.custom_creator` or
  :func:`~haiku.custom_setter`.
  """

  @property
  def shape(self):
    raise ValueError("DO_NOT_STORE does not have a shape.")

  @property
  def dtype(self):
    raise ValueError("DO_NOT_STORE does not have a dtype.")

DO_NOT_STORE = DoNotStore()

T = TypeVar("T")


def check_not_none(value: Optional[T], msg: str) -> T:
  if value is None:
    raise ValueError(msg)
  return value


def get_parameter(
    name: str,
    shape: Sequence[int],
    dtype: Any = jnp.float32,
    init: Optional[Initializer] = None,
) -> jnp.ndarray:
  """Creates or reuses a parameter for the given transformed function.

  >>> hk.get_parameter("w", [], init=jnp.ones)
  DeviceArray(1., dtype=float32)

  Parameters within the same :func:`transform` and/or :class:`Module` with the
  same name have the same value:

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
  assert_context("get_parameter")
  assert_jax_usage("get_parameter")

  init = check_not_none(init, "Initializer must be specified.")

  bundle_name = current_bundle_name()
  frame = current_frame()

  fq_name = bundle_name + "/" + name
  context = GetterContext(full_name=fq_name, module=current_module(),
                          original_dtype=dtype, original_shape=shape,
                          original_init=init)

  if bundle_name not in frame.params:
    param = None
  else:
    param = frame.params[bundle_name].get(name)

  if param is None:
    if param_creator_stack:
      param = run_creators(param_creator_stack, context, shape, dtype, init)
    else:
      param = init(shape, dtype)

    if param is DO_NOT_STORE:
      # Initializers or custom creators that return `DO_NOT_STORE` are required
      # to produce a value for the parameter via a custom getter.
      remove_if_empty(frame.params, bundle_name)
    else:
      if frame.params_frozen:
        # Throw if we needed to re-init the parameter during apply.
        raise ValueError(
            f"Unable to retrieve parameter {name!r} for module "
            f"{bundle_name!r} All parameters must be created as part of "
            "`init`.")

      param = check_not_none(param, "Parameters cannot be `None`.")
      frame.params[bundle_name][name] = param

  # Custom getters allow a hook for users to customize the value returned by
  # get_parameter. For example casting values to some dtype.
  if param_getter_stack:
    param = run_getters(param_getter_stack, context, param)

  param = check_not_none(param, "Parameters cannot be `None`.")

  if param.shape != tuple(shape):
    raise ValueError(
        f"{fq_name!r} with retrieved shape {param.shape!r} does not match "
        f"shape={shape!r} dtype={dtype!r}")

  return param


def remove_if_empty(bundle, key):
  if key in bundle and not bundle[key]:
    del bundle[key]


class GetterContext(NamedTuple):
  """Context about where parameters are being created.

  Attributes:
    full_name: The full name of the given parameter (e.g. ``mlp/~/linear_0/w``).
    module: The module that owns the current parameter, ``None`` if this
      parameter exists outside any module.
    original_dtype: The dtype that :func:`~haiku.get_parameter` or
      :func:`~haiku.get_state` was originally called with.
    original_shape: The shape that :func:`~haiku.get_parameter` or
      :func:`~haiku.get_state` was originally called with.
    module_name: The full name of enclosing modules.
    name: The name of this parameter.
  """
  full_name: str
  module: Optional[Module]
  original_dtype: Any
  original_shape: Sequence[int]
  original_init: Optional[Initializer]

  @property
  def module_name(self):
    module_name, _ = self.full_name.rsplit("/", 1)
    return module_name

  @property
  def name(self):
    _, name = self.full_name.rsplit("/", 1)
    return name


NextCreator = Callable[[Sequence[int], Any, Initializer], jnp.ndarray]
Creator = Callable[
    [NextCreator, Sequence[int], Any, Initializer, GetterContext], jnp.ndarray]


def run_creators(
    stack: ThreadLocalStack[Creator],
    context: GetterContext,
    shape: Sequence[int],
    dtype: Any = jnp.float32,
    init: Optional[Initializer] = None,
) -> jnp.ndarray:
  """See :func:`custom_creator` for usage."""
  assert stack
  stack = stack.clone()

  def next_creator(shape, dtype, init):
    if not stack:
      return init(shape, dtype)
    return stack.popleft()(next_creator, shape, dtype, init, context)

  return next_creator(shape, dtype, init)


def custom_creator(
    creator: Creator,
    *,
    params: bool = True,
    state: bool = False,
) -> contextlib.AbstractContextManager:
  """Registers a custom parameter and/or state creator.

  When new parameters are created via :func:`get_parameter` we first run custom
  creators passing user defined values through. For example:

  >>> def zeros_creator(next_creator, shape, dtype, init, context):
  ...   init = jnp.zeros
  ...   return next_creator(shape, dtype, init)

  >>> with hk.custom_creator(zeros_creator):
  ...   z = hk.get_parameter("z", [], jnp.float32, jnp.ones)
  >>> z
  DeviceArray(0., dtype=float32)

  If ``state=True`` then your creator will additionally run on calls to
  :func:`get_state`:

  >>> with hk.custom_creator(zeros_creator, state=True):
  ...   z = hk.get_state("z", [], jnp.float32, jnp.ones)
  >>> z
  DeviceArray(0., dtype=float32)

  Args:
    creator: A parameter creator.
    params: Whether to intercept parameter creation, defaults to ``True``.
    state: Whether to intercept state creation, defaults to ``False``.

  Returns:
    Context manager under which the creator is active.
  """
  assert_context("custom_creator")
  return custom_creator_unsafe(creator, params=params, state=state)


def custom_creator_unsafe(
    creator: Creator,
    *,
    params: bool = True,
    state: bool = False,
) -> contextlib.AbstractContextManager:  # pylint: disable=g-bare-generic
  """See custom_creator."""
  stack = contextlib.ExitStack()
  if params:
    stack.enter_context(param_creator_stack(creator))
  if state:
    stack.enter_context(state_creator_stack(creator))
  return stack

NextGetter = Callable[[str, jnp.ndarray], jnp.ndarray]
Getter = Callable[[NextGetter, jnp.ndarray, GetterContext], jnp.ndarray]


def run_getters(
    stack: Stack[Getter],
    context: GetterContext,
    value: jnp.ndarray,
) -> jnp.ndarray:
  """See :func:`custom_getter` for usage."""
  assert stack
  stack = stack.clone()

  def next_getter(value):
    if not stack:
      return value
    return stack.popleft()(next_getter, value, context)

  return next_getter(value)


def custom_getter(
    getter: Getter,
    *,
    params: bool = True,
    state: bool = False,
) -> contextlib.AbstractContextManager:
  """Registers a custom parameter or state getter.

  When parameters are retrieved using :func:`get_parameter` we always run all
  custom getters before returning a value to the user.

  >>> def bf16_getter(next_getter, value, context):
  ...   value = value.astype(jnp.bfloat16)
  ...   return next_getter(value)

  >>> with hk.custom_getter(bf16_getter):
  ...   w = hk.get_parameter("w", [], jnp.float32, jnp.ones)
  >>> w.dtype
  dtype(bfloat16)

  If ``state=True`` the getter will additionally run for calls to
  :func:`get_state`:

  >>> with hk.custom_getter(bf16_getter, state=True):
  ...   c = hk.get_state("c", [], jnp.float32, jnp.ones)
  >>> c.dtype
  dtype(bfloat16)

  Args:
    getter: A parameter getter.
    params: Whether the getter should run on :func:`get_parameter`
    state: Whether the getter should run on :func:`get_state`.

  Returns:
    Context manager under which the getter is active.
  """
  assert_context("custom_getter")
  stack = contextlib.ExitStack()
  if params:
    stack.enter_context(param_getter_stack(getter))
  if state:
    stack.enter_context(state_getter_stack(getter))
  return stack

T, U = TypeVar("T"), TypeVar("U")
NextSetter = Callable[[str, T], U]
Setter = Callable[[NextSetter, T, "SetterContext"], U]


def run_setters(
    stack: Stack[Setter],
    context: "SetterContext",
    value: T,
) -> U:
  """See :func:`custom_setter` for usage."""
  assert stack
  stack_copy = stack.clone()

  def next_setter(value):
    if not stack_copy:
      return value
    return stack_copy.popleft()(next_setter, value, context)

  return next_setter(value)


class SetterContext(NamedTuple):
  """Context about where state is being set.

  Attributes:
    full_name: The full name of the given state (e.g. ``mlp/~/linear_0/w``).
    module: The module that owns the current state, ``None`` if this
      state exists outside any module.
    original_dtype: The dtype that :func:`~haiku.set_state` was originally
      called with.
    original_shape: The shape that :func:`~haiku.set_state` or
      :func:`~haiku.get_state` was originally called with.
    module_name: The full name of enclosing modules.
    name: The name of this state.
  """
  full_name: str
  module: Optional[Module]
  original_dtype: Any
  original_shape: Sequence[int]

  @property
  def module_name(self):
    module_name, _ = self.full_name.rsplit("/", 1)
    return module_name

  @property
  def name(self):
    _, name = self.full_name.rsplit("/", 1)
    return name


def custom_setter(setter: Setter) -> contextlib.AbstractContextManager:
  """Registers a custom state setter.

  When state is set using :func:`get_parameter` we always run all custom setters
  before saving the value.

  >>> def zero_during_init(next_setter, value, context):
  ...   if hk.running_init():
  ...     value = jnp.zeros_like(value)
  ...   return next_setter(value)

  >>> with hk.custom_setter(zero_during_init):
  ...   hk.set_state("x", jnp.ones([2]))
  ...   x = hk.get_state("x")
  >>> x
  DeviceArray([0., 0.], dtype=float32)

  Args:
    setter: A state setter.

  Returns:
    Context manager under which the setter is active.
  """
  assert_context("custom_setter")
  return state_setter_stack(setter)


def assert_is_prng_key(key: PRNGKey):
  """Asserts that the given input looks like a `jax.random.PRNGKey`."""
  type_error = ValueError("The provided key is not a JAX PRNGKey but a "
                          f"{type(key)}:\n{key}")
  if (hasattr(jax.config, "jax_enable_custom_prng")
      and jax.config.jax_enable_custom_prng):
    if not isinstance(key, jax.random.KeyArray):
      raise type_error
  if not hasattr(key, "shape"):
    raise type_error
  if hasattr(key, "dtype"):
    config_hint = ""
    if hasattr(jax.random, "default_prng_impl"):
      default_impl = jax.random.default_prng_impl()
      expected_shapes = (default_impl.key_shape,)
      if default_impl.key_shape != (2,):
        # Default PRNG impl is set to something different from threefry.
        config_hint = ("\nHint: jax_default_prng_impl has been set to "
                       f"'{jax_config.jax_default_prng_impl}', the shape "
                       "mismatch might be because a jax.random.PRNGKey was "
                       "created before this flag was set.")
    else:
      # Check for the shapes of the known PRNG impls (threefry and RBG)
      expected_shapes = ((2,), (4,))
    if key.shape not in expected_shapes or key.dtype != jnp.uint32:
      expected_shapes_str = " or ".join(map(str, expected_shapes))
      raise ValueError(
          "Provided key did not have expected shape and/or dtype: "
          f"expected=(shape={expected_shapes_str}, dtype=uint32), "
          f"actual=(shape={key.shape}, dtype={key.dtype}){config_hint}")


PRNGSequenceState = Tuple[PRNGKey, Iterable[PRNGKey]]


class PRNGSequence(Iterator[PRNGKey]):
  """Iterator of JAX random keys.

  >>> seq = hk.PRNGSequence(42)  # OR pass a jax.random.PRNGKey
  >>> key1 = next(seq)
  >>> key2 = next(seq)
  >>> assert key1 is not key2

  If you know how many keys you will want then you can use :meth:`reserve` to
  more efficiently split the keys you need::

  >>> seq.reserve(4)
  >>> keys = [next(seq) for _ in range(4)]
  """
  __slots__ = ("_key", "_subkeys")

  def __init__(self, key_or_seed: Union[PRNGKey, int, PRNGSequenceState]):
    """Creates a new :class:`PRNGSequence`."""
    if isinstance(key_or_seed, tuple):
      key, subkeys = key_or_seed
      assert_is_prng_key(key)
      for subkey in subkeys:
        assert_is_prng_key(subkey)
      self._key = key
      self._subkeys = collections.deque(subkeys)
    else:
      if isinstance(key_or_seed, int):
        key_or_seed = jax.random.PRNGKey(key_or_seed)
      # A seed value may also be passed as an int32-typed scalar ndarray.
      elif (hasattr(key_or_seed, "shape") and (not key_or_seed.shape) and
            hasattr(key_or_seed, "dtype") and key_or_seed.dtype == jnp.int32):
        key_or_seed = jax.random.PRNGKey(key_or_seed)
      else:
        assert_is_prng_key(key_or_seed)
      self._key = key_or_seed
      self._subkeys = collections.deque()

  def reserve(self, num):
    """Splits additional ``num`` keys for later use."""
    if num > 0:
      # When storing keys we adopt a pattern of key0 being reserved for future
      # splitting and all other keys being provided to the user in linear order.
      # In terms of jax.random.split this looks like:
      #
      #     key, subkey1, subkey2 = jax.random.split(key, 3)  # reserve(2)
      #     key, subkey3, subkey4 = jax.random.split(key, 3)  # reserve(2)
      #
      # Where subkey1->subkey4 are provided to the user in order when requested.
      new_keys = tuple(jax.random.split(self._key, num + 1))
      self._key = new_keys[0]
      self._subkeys.extend(new_keys[1:])

  def reserve_up_to_full(self):
    num = DEFAULT_PRNG_RESERVE_SIZE - len(self._subkeys)
    if num > 0:
      self.reserve(num)
    else:
      sliced_subkeys = list(self._subkeys)[:DEFAULT_PRNG_RESERVE_SIZE]
      self._subkeys = collections.deque(sliced_subkeys)

  @property
  def internal_state(self) -> PRNGSequenceState:
    return self._key, tuple(self._subkeys)

  def replace_internal_state(self, state: PRNGSequenceState):
    key, subkeys = state
    assert_is_prng_key(key)
    for subkey in subkeys:
      assert_is_prng_key(subkey)
    self._key = key
    self._subkeys = collections.deque(subkeys)

  def __next__(self) -> PRNGKey:
    if not self._subkeys:
      self.reserve(DEFAULT_PRNG_RESERVE_SIZE)
    return self._subkeys.popleft()

  next = __next__

  def take(self, num) -> Tuple[PRNGKey, ...]:
    self.reserve(max(num - len(self._subkeys), 0))
    return tuple(next(self) for _ in range(num))


def rng_seq_or_fail() -> PRNGSequence:
  rng_seq = current_frame().rng_stack.peek()
  if rng_seq is None:
    raise ValueError("You must pass a non-None PRNGKey to init and/or apply "
                     "if you make use of random numbers.")
  return rng_seq


def reserve_rng_keys(num: int):
  """Pre-allocate some number of JAX RNG keys.

  See :func:`next_rng_key`.

  This API offers a way to micro-optimize how RNG keys are split when using
  Haiku. It is unlikely that you need it unless you find compilation time of
  your ``init`` function to be a problem, or you sample a lot of random numbers
  in ``apply``.

  >>> hk.reserve_rng_keys(2)  # Pre-allocate 2 keys for us to consume.
  >>> _ = hk.next_rng_key()   # Takes the first pre-allocated key.
  >>> _ = hk.next_rng_key()   # Takes the second pre-allocated key.
  >>> _ = hk.next_rng_key()   # Splits a new key.

  Args:
    num: The number of JAX rng keys to allocate.
  """
  assert_context("reserve_rng_keys")
  assert_jax_usage("reserve_rng_keys")
  rng_seq = rng_seq_or_fail()
  rng_seq.reserve(num)


def next_rng_key() -> PRNGKey:
  """Returns a unique JAX random key split from the current global key.

  >>> key = hk.next_rng_key()
  >>> _ = jax.random.uniform(key, [])

  Returns:
    A unique (within a call to ``init`` or ``apply``) JAX rng key that can be
    used with APIs such as :func:`jax.random.uniform`.
  """
  assert_context("next_rng_key")
  assert_jax_usage("next_rng_key")
  return next_rng_key_internal()


class JaxUsageError(ValueError):
  pass

JaxUsageError.__module__ = "haiku"


def push_jax_trace_level():
  return current_frame().jax_trace_stack(JaxTraceLevel.current())


def assert_jax_usage(public_symbol_name: str):
  if not config.get_config().check_jax_usage:
    return

  expected_level = current_frame().jax_trace_stack.peek()
  trace_level = JaxTraceLevel.current()
  if trace_level != expected_level:
    raise JaxUsageError(
        "tl;dr - You need to use a Haiku overloaded transform (e.g. `hk.vmap`) "
        "or control flow operator (e.g. `hk.cond`) instead of the `jax.*` "
        "equivalent for untransformed functions using Haiku APIs."
        "\n\n",
        "Some APIs in JAX (e.g. `jit`, `vmap`, `cond`, `switch`) take "
        f"functions that are expected to be pure. `hk.{public_symbol_name}` "
        "has a side effect, and using it inside a function (without also using "
        "`hk.transform`) makes that function 'impure' (the function has a side "
        "effect)."
        "\n\n"
        "Haiku includes drop-in replacements for these JAX APIs (e.g. "
        "`hk.vmap`) that carefully turn your function into a pure function and "
        "then call the underlying JAX function.")


# NOTE: Split for monkey patching in random.py.
def next_rng_key_internal() -> PRNGKey:
  rng_seq = rng_seq_or_fail()
  return next(rng_seq)


def next_rng_keys(num: int) -> jnp.ndarray:
  """Returns one or more JAX random keys split from the current global key.

  >>> k1, k2 = hk.next_rng_keys(2)
  >>> assert (k1 != k2).all()
  >>> a = jax.random.uniform(k1, [])
  >>> b = jax.random.uniform(k2, [])
  >>> assert a != b

  Args:
    num: The number of keys to split.

  Returns:
    An array of shape ``[num, 2]`` unique (within a transformed function) JAX
    rng keys that can be used with APIs such as :func:`jax.random.uniform`.
  """
  assert_context("next_rng_keys")
  assert_jax_usage("next_rng_keys")
  rng_seq = rng_seq_or_fail()
  return jnp.vstack(rng_seq.take(num))


def maybe_next_rng_key() -> Optional[PRNGKey]:
  """:func:`next_rng_key` if random numbers are available, else ``None``."""
  assert_context("maybe_next_rng_key")
  rng_seq = current_frame().rng_stack.peek()
  if rng_seq is not None:
    assert_jax_usage("maybe_next_rng_key")
    return next(rng_seq)
  else:
    return None


def extract_state(state: MutableState, *, initial) -> State:
  state = {m: {k: (v.initial if initial else v.current) for k, v in p.items()}
           for m, p in state.items()}
  state = data_structures.to_haiku_dict(state)
  return state


def get_state(
    name: str,
    shape: Optional[Sequence[int]] = None,
    dtype: Any = jnp.float32,
    init: Optional[Initializer] = None,
) -> jnp.ndarray:
  """Gets the current value for state with an optional initializer.

  "State" can be used to represent mutable state in your network. The most
  common usage of state is to represent the moving averages used in batch
  normalization (see :class:`ExponentialMovingAverage`). If your network uses
  "state" then you are required to use :func:`transform_with_state` and pass
  state into and out of the apply function.

  >>> hk.get_state("counter", [], init=jnp.zeros)
  DeviceArray(0., dtype=float32)

  If the value for the given state is already defined (e.g. using
  :func:`set_state`) then you can call with just the name:

  >>> hk.get_state("counter")
  DeviceArray(0., dtype=float32)

  MOTE: state within the same :func:`transform` and/or :class:`Module` with the
  same name have the same value:

  >>> c1 = hk.get_state("counter")
  >>> c2 = hk.get_state("counter")
  >>> assert c1 is c2

  Args:
    name: A name for the state.
    shape: The shape of the state.
    dtype: The dtype of the state.
    init: A callable ``f(shape, dtype)`` that returns an initial value for the
      state.

  Returns:
    A jnp.ndarray with the state of the given shape.
  """
  assert_context("get_state")
  assert_jax_usage("get_state")
  bundle_name = current_bundle_name()
  state = current_frame().state[bundle_name]
  fq_name = f"{bundle_name}/{name}"
  context = GetterContext(fq_name, current_module(), dtype, shape, init)

  value = state.get(name, None)
  if value is None:
    if init is None:
      raise ValueError(f"No value for {name!r} in {bundle_name!r}, perhaps "
                       "set an init function?")
    if shape is None or dtype is None:
      raise ValueError(f"Must provide shape and dtype to initialize {name!r} "
                       f"in {bundle_name!r}.")

    if state_creator_stack:
      value = run_creators(state_creator_stack, context, shape, dtype, init)
    else:
      value = init(shape, dtype)

    if value is not DO_NOT_STORE:
      state[name] = StatePair(value, value)
  else:
    value = value.current

  # Custom getters allow a hook for users to customize the value returned by
  # get_state. For example casting values to some dtype.
  if state_getter_stack:
    value = run_getters(state_getter_stack, context, value)

  return value

maybe_shape = lambda x: getattr(x, "shape", None)
maybe_dtype = lambda x: getattr(x, "dtype", None)


def set_state(name: str, value):
  """Sets the current value for some state.

  See :func:`get_state`.

  "State" can be used to represent mutable state in your network. The most
  common usage of state is to represent the moving averages used in batch
  normalization (see :class:`ExponentialMovingAverage`). If your network uses
  "state" then you are required to use :func:`transform_with_state` and pass
  state into and out of the apply function.

  >>> hk.set_state("counter", jnp.zeros([]))
  >>> hk.get_state("counter")
  DeviceArray(0., dtype=float32)

  NOTE: state within the same :func:`transform` and/or :class:`Module` with the
  same name have the same value:

  >>> w1 = hk.get_state("counter")
  >>> w2 = hk.get_state("counter")
  >>> assert w1 is w2

  Args:
    name: A name for the state.
    value: A value to set.
  """
  assert_context("set_state")
  assert_jax_usage("set_state")
  frame = current_frame()
  bundle_name = current_bundle_name()
  state = frame.state[bundle_name]

  if state_setter_stack:
    shape = jax.tree_map(maybe_shape, value)
    dtype = jax.tree_map(maybe_dtype, value)
    fq_name = bundle_name + "/" + name
    context = SetterContext(full_name=fq_name, module=current_module(),
                            original_dtype=dtype, original_shape=shape)
    value = run_setters(state_setter_stack, context, value)

    if value is DO_NOT_STORE:
      remove_if_empty(frame.state, bundle_name)
      return

  if name in state:
    initial, _ = state[name]
    current = value
  else:
    initial = current = value
  state[name] = StatePair(initial, current)


def with_rng(key: PRNGKey):
  """Provides a new sequence for :func:`next_rng_key` to draw from.

  When :func:`next_rng_key` is called, it draws a new key from the
  :class:`PRNGSequence` defined by the input key to the transformed function.
  This context manager overrides the sequence for the duration of the scope.

  >>> with hk.with_rng(jax.random.PRNGKey(428)):
  ...   s = jax.random.uniform(hk.next_rng_key(), ())
  >>> print("{:.1f}".format(s))
  0.5

  Args:
    key: The key to seed the sequence with.

  Returns:
    Context manager under which the given sequence is active.
  """
  assert_context("with_rng")
  assert_jax_usage("with_rng")
  return current_frame().rng_stack(PRNGSequence(key))


def param_names() -> FrozenSet[Tuple[str, str]]:
  """Returns all module and parameter names as a set of pairs."""
  out = []
  params = current_frame().params
  for mod_name, bundle in params.items():
    if not isinstance(bundle, Mapping):
      # TODO(tomhennigan) Fix broken user code and remove this warning.
      warnings.warn(f"Invalid entry {mod_name!r} in params {params}")
      continue

    for name in bundle:
      out.append((mod_name, name))
  return frozenset(out)


@contextlib.contextmanager
def assert_no_new_parameters():
  before = param_names()
  yield
  diff = param_names() - before
  if diff:
    raise AssertionError(f"New parameters were created: {list(sorted(diff))}")


def _get_ids(collection_name: str) -> FrozenSet[int]:
  """Returns the identity for all state in the current context."""
  out = []
  collection = getattr(current_frame(), collection_name)
  for mod_name, bundle in collection.items():
    if not isinstance(bundle, Mapping):
      # TODO(tomhennigan) Fix broken user code and remove this warning.
      warnings.warn(f"Invalid entry {mod_name!r} in collection {collection}")
      continue

    out.extend(map(id, bundle.values()))

  return frozenset(out)


def _all_state():
  params = _get_ids("params")
  state = _get_ids("state")
  rng = current_frame().rng_stack.peek()
  if rng is not None:
    key, subkeys = rng.internal_state
    rng = frozenset(map(id, [key] + list(subkeys)))
  else:
    rng = frozenset()
  return params, state, rng


@contextlib.contextmanager
def assert_state_unchanged():
  """Asserts that in the given block params, state and rng are unchanged."""
  params_before, state_before, rng_before = _all_state()
  yield
  params_after, state_after, rng_after = _all_state()

  params_diff = params_after - params_before
  state_diff = state_after - state_before
  rng_diff = rng_after - rng_before
  if params_diff or state_diff or rng_diff:
    raise StateChangedError("Within this code block you are not able to modify "
                            "Haiku managed state (e.g. via `next_rng_key` or "
                            "`set_state`).")


class StateChangedError(AssertionError):
  pass
