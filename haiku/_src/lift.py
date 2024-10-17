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
"""Lifting parameters in Haiku."""

from collections.abc import Callable, Mapping, MutableMapping
import functools
from typing import Any, TypeVar

from haiku._src import base
from haiku._src import data_structures
from haiku._src import module
from haiku._src import transform
from haiku._src.typing import LiftingModuleType


# If you are forking replace this with `import haiku as hk`.
# pylint: disable=invalid-name
class hk:
  Params = base.Params
  State = base.State
  Module = module.Module
  running_init = transform.running_init
  data_structures = data_structures
# pylint: enable=invalid-name
del module, data_structures, transform

T = TypeVar("T")
MutableBundle = MutableMapping[str, MutableMapping[str, Any]]


def pack_into_dict(src: hk.Params,
                   dst: MutableMapping[str, Any],
                   prefix: str,
                   state: bool = False,
                   check_param_reuse: bool = True):
  """Puts items from src into dst, with an added prefix."""
  for key, value in src.items():
    new_key = f"{prefix}/{key}" if prefix else key
    if check_param_reuse and new_key in dst:
      raise ValueError(
          f"Key '{new_key}' already exists in the destination params. To "
          "prevent accidental parameter re-use during lift, you can't re-use a "
          "parameter already defined in the outer scope.")
    value = dict(value)
    if state:
      value = {k: base.StatePair(v, v) for k, v in value.items()}
    dst.setdefault(new_key, {})
    dst[new_key].update(value)


def unpack_from_dict(src: hk.Params, prefix: str) -> MutableBundle:
  """Returns pairs from src where key begins with prefix, cutting off prefix."""
  if not prefix:
    return {k: dict(v) for k, v in src.items()}

  l = len(prefix)
  return {k[l:]: dict(v) for k, v in src.items() if k.startswith(prefix)}


def add_state_to_init_fn(
    init_fn: Callable[..., hk.Params],
) -> Callable[..., tuple[hk.Params, hk.State]]:
  def wrapped_init_fn(*a, **k):
    params = init_fn(*a, **k)
    if not isinstance(params, Mapping):
      raise base.NonEmptyStateError("For stateful lifted functions use "
                                    "`hk.lift_with_state`")
    return params, {}
  return wrapped_init_fn


# TODO(tycai): Make sure transformed functions have better names.
class LiftingModule(hk.Module, LiftingModuleType):
  """See :func:`lift` or :func:`lift_with_state`."""

  def __init__(self,
               init_fn,
               transparent=False,
               allow_reuse=False,
               name="lifted"):
    super().__init__(name=name)
    self._init_fn = init_fn
    if transparent:
      self.prefix_name = "/".join(self.module_name.split("/")[:-1])
    else:
      self.prefix_name = self.module_name
    self._allow_reuse = allow_reuse

  def __call__(self, *args, **kwargs):
    frame = base.current_frame()
    outer_params = frame.params
    outer_state = frame.state
    if hk.running_init():
      inner_params, inner_state = self._init_fn(*args, **kwargs)
      # Lift parameters into this transform's params_dict.
      check_param_reuse = not self._allow_reuse
      pack_into_dict(
          inner_params,
          outer_params,
          self.prefix_name,
          check_param_reuse=check_param_reuse)
      pack_into_dict(
          inner_state,
          outer_state,
          self.prefix_name,
          state=True,
          check_param_reuse=check_param_reuse)
      return inner_params, inner_state
    else:
      if self.prefix_name:
        prefix = f"{self.prefix_name}/"
      else:
        prefix = ""
      inner_params = unpack_from_dict(outer_params, prefix)
      inner_state = unpack_from_dict(outer_state, prefix)
      inner_state = base.extract_state(inner_state, initial=False)
      inner_params = hk.data_structures.to_haiku_dict(inner_params)
      return inner_params, inner_state


def lift(
    init_fn: Callable[..., hk.Params],
    *,
    allow_reuse: bool = False,
    name: str = "lifted",
) -> Callable[..., hk.Params]:
  r"""Registers parameters from an inner init function in an outer transform.

  HINT: :func:`lift` is for when you want to make non-trivial use of JAX
  transforms (e.g. ``jax.vmap``) **inside** of a :func:`transform` or
  :func:`transform_with_state`. We generally recommend trying to use JAX
  transforms on the pure functions returned by :func:`transform`, in which case
  you do not need :func:`lift`.

  Use :func:`lift`\ when nesting Haiku transforms to register the parameters of
  the inner transform in any outer transform. This is mainly useful when using
  JAX functions inside of a Haiku module (eg. using ``jax.vmap`` on a layer).
  See
  https://dm-haiku.readthedocs.io/en/latest/notebooks/transforms.html#Using-hk.lift
  for more explanation of when to use :func:`lift`\ . (If you're not using JAX
  functions inside of a module or don't need access to your parameters inside of
  a transform, you probably don't need to use :func:`lift`\ )

  Must be called inside :func:`transform`\ , and be passed the ``init``
  member of a :class:`Transformed`\ .

  During init, the returned callable will run the given ``init_fn``, and include
  the resulting params in the outer transform's dictionaries.
  During ``apply``, the returned callable will instead pull the relevant
  parameters from the outer transform's dictionaries.

  By default, users must ensure that the given ``init`` does not accidentally
  catch modules from an outer :func:`transform` via functional
  closure. If this behavior is desirable, set ``allow_reuse`` to ``True``.

  Example:

  A common usage of :func:`lift` is to use JAX transformations like ``vmap`` in
  non-trivial ways, inside a :func:`transform`. For example, we can use
  :func:`lift` and ``jax.vmap`` to create an ensemble.

  First we'll create a helper function that uses :func:`lift` to apply ``vmap``
  to our model. As you can see from the comments, we are using ``vmap`` to
  change how parameters should be created (in this case we create a unique set
  of parameters for each member of the ensemble) and we change how apply works
  (we "map" the parameters meaning JAX will compute the forward pass separately,
  in parallel, for each member of the ensemble):

  >>> def create_ensemble(model, size: int):
  ...   init_rng = hk.next_rng_keys(size) if hk.running_init() else None
  ...   model = hk.transform(model)
  ...   # in_axes: rng is mapped, data is not.
  ...   init_model = jax.vmap(model.init, in_axes=(0, None))
  ...   # Use hk.lift to "lift" parameters created by `init_model` into the
  ...   # outer transform.
  ...   init_model = hk.lift(init_model, name="ensemble")
  ...   def ensemble(x):
  ...     params = init_model(init_rng, x)
  ...     # in_axes: params are mapped, rng/data are not.
  ...     return jax.vmap(model.apply, in_axes=(0, None, None))(params, None, x)
  ...   return ensemble

  We can now use this function to ensemble any Haiku module(s), inside of a
  transform. First we define a function for each member of the ensemble:

  >>> def member_fn(x):
  ...   return hk.nets.MLP([300, 100, 10])(x)

  Secondly we can combine our two functions, inside a :func:`transform` to
  create an ensemble:

  >>> def f(x):
  ...   ensemble = create_ensemble(member_fn, size=4)
  ...   x = ensemble(x)
  ...   # You could create other modules here which were not ensembled.
  ...   return x
  >>> f = hk.transform(f)

  When we initialize the network, our ensemble member's parameters have a
  leading dimension the size of the ensemble:

  >>> rng = jax.random.PRNGKey(777)
  >>> x = jnp.ones([32, 128])
  >>> params = f.init(rng, x)
  >>> jax.tree.map(lambda x: x.shape, params)
  {'ensemble/mlp/~/linear_0': {'b': (4, 300), 'w': (4, 128, 300)},
   'ensemble/mlp/~/linear_1': {'b': (4, 100), 'w': (4, 300, 100)},
   'ensemble/mlp/~/linear_2': {'b': (4, 10), 'w': (4, 100, 10)}}

  When we apply the network, we get an output for each member of the ensemble
  for the entire batch:

  >>> y = f.apply(params, None, x)
  >>> y.shape
  (4, 32, 10)

  Args:
    init_fn: The ``init`` function from an :class:`Transformed`\ .
    allow_reuse: Allows lifted parameters and state to be reused from the outer
      :func:`transform`. This can be desirable when using ``lift`` within
      control flow (e.g. ``hk.scan``).
    name: A string name to prefix parameters with.

  Returns:
    A callable that during ``init`` injects parameter values into the outer
    context and during ``apply`` retrieves parameters from the outer context. In
    both cases returns parameter values to be used with an ``apply`` function.

  See also:
    - :func:`lift_with_state`: Register params and state with an outer
      transform.
    - :func:`transparent_lift`: Register params with an outer transform
      without a namespace.
    - :func:`transparent_lift_with_state`: Register params and state with
      an outer transform without a namespace.
  """
  base.assert_context("lift")
  init_fn = add_state_to_init_fn(init_fn)
  params_and_state_fn, updater = lift_with_state(
      init_fn, allow_reuse=allow_reuse, name=name)
  updater.ignore_update()
  return lambda *a, **k: params_and_state_fn(*a, **k)[0]


def transparent_lift(
    init_fn: Callable[..., hk.Params],
    *,
    allow_reuse: bool = False) -> Callable[..., hk.Params]:
  r"""Registers parameters in an outer transform without adding a name scope.

  Functionally this is equivalent to :func:`lift`\ but without automatically
  adding an additional variable scoping. Note that closing over a module from
  an outer scope is disallowed.

  See :func:`lift`\ for more context on when to use ``lift``.

  Args:
    init_fn: The ``init`` function from an :class:`Transformed`\ .
    allow_reuse: Allows lifted parameters to be reused from the outer
      :func:`transform_with_state`\ . This can be desirable when e.g. within
      control flow (e.g. ``hk.scan``).

  Returns:
    A callable that during ``init`` injects parameter values into the outer
    context and during ``apply`` reuses parameters from the outer context. In
    both cases returns parameter values to be used with an ``apply`` function.

  See also:
    - :func:`lift`: Register params with an outer transform.
    - :func:`lift_with_state`: Register params and state with an outer
      transform.
    - :func:`transparent_lift_with_state`: Register params and state with an
      outer transform without a namespace.
  """

  base.assert_context("transparent_lift")
  init_fn = add_state_to_init_fn(init_fn)
  lifted = LiftingModule(
      init_fn, transparent=True, allow_reuse=allow_reuse)
  def fn(*a, **k):
    with base.closure_boundary_stack(base.current_frame().frame_id+1):
      return lifted(*a, **k)[0]

  return fn


def with_assert_not_used(f):
  """Validates that an updater method is called correctly."""
  # NOTE defined outside LiftWithStateUpdater to avoid adding this to the public
  # API and letting users call directly.
  @functools.wraps(f)
  def wrapper(self, *a, **k):
    if self._used:  # pylint: disable=protected-access
      raise ValueError("State updater must only be used once.")

    if not base.inside_transform():
      raise ValueError(
          "State updater must be used inside hk.transform_with_state.")

    if self._context_id != id(base.current_context()):  # pylint: disable=protected-access
      raise ValueError(
          "State updater must be used within the same call to init/apply.")

    self._used = True  # pylint: disable=protected-access

    return f(self, *a, **k)

  return wrapper


class LiftWithStateUpdater:
  """Handles updating the state for a `lift_with_state` computation."""

  __slots__ = ("_used", "_name", "_context_id")

  def __init__(self, name: str | None):
    self._used = False
    self._name = name
    ctx = base.current_context()
    # Note: using ID is safe here because we know the lifetime of the context
    # instance will outlive the updater thanks to the callback.
    self._context_id = id(ctx)
    ctx.add_teardown_callback(self.assert_used)

  def assert_used(self):
    if not self._used:
      raise ValueError("LiftWithStateUpdater (from `lift_with_state`) must be "
                       "used, call `.update(..)` or `.ignore_update()` before "
                       "it goes out of scope.")

  @with_assert_not_used
  def ignore_update(self):
    """Notifies the updater that state does not need to be updated."""
    pass

  @with_assert_not_used
  def update(self, state: hk.State):
    """Updates Haiku's internal state to the given state."""
    frame = base.current_frame()
    for mod_name, bundle in state.items():
      if self._name is not None:
        mod_name = f"{self._name}/{mod_name}"
      for name, value in bundle.items():
        initial_pair = base.StatePair(value, value)
        initial = frame.state[mod_name].get(name, initial_pair).initial
        frame.state[mod_name][name] = base.StatePair(initial, value)


def _to_callable(f: Callable[..., T]) -> Callable[..., T]:
  """Enapsulates the given callable inside a lambda."""
  # Prevents us from leaking methods other than __call__ on `f`.
  return lambda *a, **k: f(*a, **k)  # pylint: disable=unnecessary-lambda


def lift_with_state(
    init_fn: Callable[..., tuple[hk.Params, hk.State]],
    *,
    allow_reuse: bool = False,
    name: str = "lifted",
) -> tuple[Callable[..., tuple[hk.Params, hk.State]], LiftWithStateUpdater]:
  r"""Registers params and state from an init function in an outer transform.

  See :func:`lift`\ for more context on when to use ``lift``.

  This function returns two objects. The first is a callable that runs your init
  function with slightly different behaviour based on if it's run during init
  vs. apply time. The second is an updater that can be used to pass updated
  state values that result from running your apply function. See later in the
  docs for a worked example.

  During init, the returned callable will run the given ``init_fn``, and include
  the resulting params/state in the outer transform's dictionaries. During
  ``apply``, the returned callable will instead pull the relevant params/state
  from the outer transform's dictionaries.

  Must be called inside :func:`transform_with_state`\ , and be passed the
  ``init`` member of a :class:`TransformedWithState`\ .

  By default, users must ensure that the given ``init`` does not accidentally
  catch modules from an outer :func:`transform_with_state` via functional
  closure. If this behavior is desirable, set ``allow_reuse`` to ``True``.

  Example:

    >>> def g(x):
    ...   return hk.nets.ResNet50(1)(x, True)
    >>> g = hk.transform_with_state(g)
    >>> params_and_state_fn, updater = (
    ...   hk.lift_with_state(g.init, name='f_lift'))
    >>> init_rng = hk.next_rng_key() if hk.running_init() else None
    >>> x = jnp.ones([1, 224, 224, 3])
    >>> params, state = params_and_state_fn(init_rng, x)
    >>> out, state = g.apply(params, state, None, x)
    >>> updater.update(state)

  Args:
    init_fn: The ``init`` function from an :class:`TransformedWithState`\ .
    allow_reuse: Allows lifted parameters and state to be reused from the
      outer :func:`transform_with_state`. This can be desirable when using
      ``lift_with_state`` within control flow (e.g. ``hk.scan``).
    name: A string name to prefix parameters with.

  Returns:
    A callable that during ``init`` injects parameter values into the outer
    context and during ``apply`` reuses parameters from the outer context. In
    both cases returns parameter values to be used with an ``apply`` function.
    The ``init`` function additionally returns an object used to update the
    outer context with new state after ``apply`` is called.

  See also:
    - :func:`lift`: Register parameters with an outer transform.
    - :func:`transparent_lift`: Register parameters with an outer transform
      without a namespace.
    - :func:`transparent_lift_with_state`: Register parameters and state with an
      outer transform without a namespace.
  """
  base.assert_context("lift_with_state")
  params_and_state_fn = _to_callable(
      LiftingModule(init_fn, allow_reuse=allow_reuse, name=name))
  if base.current_module():
    name = f"{base.current_name()}/{name}"
  updater = LiftWithStateUpdater(name)
  return params_and_state_fn, updater


def transparent_lift_with_state(
    init_fn: Callable[..., tuple[hk.Params, hk.State]],
    *,
    allow_reuse: bool = False,
) -> tuple[Callable[..., tuple[hk.Params, hk.State]], LiftWithStateUpdater]:
  r"""Registers params and state in an outer transform without adding scope.

  Functionally this is equivalent to :func:`lift_with_state`\ but without
  automatically adding an additional variable scoping.

  See :func:`lift_with_state`\ for more context on when to use
  ``lift_with_state``.

  Args:
    init_fn: The ``init`` function from an :class:`TransformedWithState`\.
    allow_reuse: Allows lifted parameters and state to be reused from the outer
      :func:`transform_with_state`\ . This can be desirable when e.g. within
      control flow (e.g. ``hk.scan``).

  Returns:
    A callable that during ``init`` injects parameter values into the outer
    context and during ``apply`` reuses parameters from the outer context. In
    both cases returns parameter values to be used with an ``apply`` function.
    The ``init`` function additionally returns an object used to update the
    outer context with new state after ``apply`` is called.

  See also:
    - :func:`lift`: Register params with an outer transform.
    - :func:`lift_with_state`: Register params and state with an outer
      transform.
    - :func:`transparent_lift`: Register params with an outer transform
      without a namespace.
  """

  base.assert_context("lift_with_state")
  params_and_state_fn = _to_callable(
      LiftingModule(init_fn, transparent=True, allow_reuse=allow_reuse))
  name = base.current_name() if base.current_module() else None
  updater = LiftWithStateUpdater(name)
  return params_and_state_fn, updater
