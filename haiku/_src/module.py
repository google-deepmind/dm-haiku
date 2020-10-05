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

import abc
import contextlib
import functools
import inspect
import re
from typing import (Any, Callable, ContextManager, Dict, Mapping, NamedTuple,
                    Optional, Tuple, Type, TypeVar)

from haiku._src import base
from haiku._src import data_structures
from haiku._src import named_call
from haiku._src import utils
import jax.numpy as jnp

ThreadLocalStack = data_structures.ThreadLocalStack
T = TypeVar("T")
_APPLY_NAME_SCOPE = "__haiku_name_scope"
modules_with_named_call = False


def profiler_name_scopes(enabled=True):
  """Enable/disable profiler name_scopes on all haiku module methods.

  Note: currently only enables for ``__call__``. See: :function:`named_call` if
  you want to annotate other methods explicitly.

  Args:
    enabled: Whether to enable name scopes or not.
  Returns:
    The previous value of the name_scopes setting.
  """
  global modules_with_named_call
  previously_enabled = modules_with_named_call
  modules_with_named_call = enabled
  return previously_enabled


class ModuleMetaclass(abc.ABCMeta):
  """Metaclass for `Module`."""

  def __new__(
      mcs: Type[Type[T]],
      name: str,
      bases: Tuple[Type[Any], ...],
      clsdict: Dict[str, Any],
  ) -> Type[T]:
    method_names = []

    for key, value in clsdict.items():
      if key == "module_name":
        # Don't patch `module_name` in case the user implemented it as a
        # computed property.
        continue

      elif key.startswith("__") and key != "__call__":
        # Don't patch methods like `__getattr__` or `__del__`.
        continue

      elif isinstance(value, property):
        # TODO(tomhennigan) Preserve the type of property subclasses.
        clsdict[key] = property(
            value.fget if not value.fget else wrap_method(key, value.fget),
            value.fset if not value.fset else wrap_method(key, value.fset),
            value.fdel if not value.fdel else wrap_method(key, value.fdel),
            doc=value.__doc__)

      elif inspect.isfunction(value):
        # We defer patching methods until after the type is created such that we
        # can trigger the descriptor binding them to the class.
        method_names.append(key)

    clsdict.setdefault(
        "__repr__",
        lambda module: module._auto_repr)  # pylint: disable=protected-access

    cls = super(ModuleMetaclass, mcs).__new__(mcs, name, bases, clsdict)

    for method_name in method_names:
      # Note: the code below is subtle, we need to ensure that we're wrapping
      # the method bound to the class. In some cases (e.g. `wrapt`) this is
      # important since the method can trigger different behavior when it is
      # bound (e.g. in wrapt `FunctionWrapper.__get__(None, cls)` produces a
      # `BoundFunctionWrapper` which in turn populates the `instance` argument
      # to decorator functions using args[0]).
      # Equivalent to: `cls.__dict__[method_name].__get__(None, cls)`
      method = getattr(cls, method_name)
      method = wrap_method(method_name, method)
      setattr(cls, method_name, method)

    return cls

  def __call__(cls: Type[T], *args, **kwargs) -> T:
    # Call new such that we have an un-initialized module instance that we can
    # still reference even if there is an exception during __init__. This is
    # needed such that we can make sure the name_scope constructed in __init__
    # is closed even if there is an exception.

    # NOTE: We disable pytype since (somewhat surprisingly) this method is bound
    # with the new class and not the metaclass.
    module = cls.__new__(cls, *args, **kwargs)  # pytype: disable=wrong-arg-types

    # Now attempt to initialize the object.
    init = wrap_method("__init__", cls.__init__)
    init(module, *args, **kwargs)

    module._auto_repr = utils.auto_repr(cls, *args, **kwargs)  # pylint: disable=protected-access

    ran_super_ctor = hasattr(module, "module_name")
    if not ran_super_ctor:
      raise ValueError(
          "Constructing an hk.Module without calling the super constructor "
          "is not supported. Add the following as the first line in your "
          "__init__ method:\n\nsuper(%s, self).__init__()" % cls.__name__)

    return module


class MethodContext(NamedTuple):
  r"""Read only state showing the calling context for a method.

  For example lets define two interceptors and print the values in the context.
  Additionally we will make the first interceptor conditionally short circuit,
  since interceptors stack and are run in order, an earlier interceptor can
  decide to call the next interecptor, or short circuit and call the underlying
  method directly:

  >>> module = hk.Linear(1, name="method_context_example")
  >>> short_circuit = False

  >>> def my_interceptor_1(next_fun, args, kwargs, context):
  ...   print('running my_interceptor_1')
  ...   print('- module.name: ', context.module.name)
  ...   print('- method_name: ', context.method_name)
  ...   if short_circuit:
  ...     return context.orig_method(*args, **kwargs)
  ...   else:
  ...     return next_fun(*args, **kwargs)
  >>> def my_interceptor_2(next_fun, args, kwargs, context):
  ...   print('running my_interceptor_2')
  ...   print('- module.name: ', context.module.name)
  ...   print('- method_name: ', context.method_name)
  ...   return next_fun(*args, **kwargs)

  When ``short_circuit=False`` the two interceptors will run in order:

  >>> with hk.experimental.intercept_methods(my_interceptor_1), \
  ...      hk.experimental.intercept_methods(my_interceptor_2):
  ...   _ = module(jnp.ones([1, 1]))
  running my_interceptor_1
  - module.name:  method_context_example
  - method_name:  __call__
  running my_interceptor_2
  - module.name:  method_context_example
  - method_name:  __call__

  Setting ``short_circuit=True`` will cause the first interecptor to call the
  original method (rather than ``next_fun`` which will trigger the next
  interceptor):

  >>> short_circuit = True
  >>> with hk.experimental.intercept_methods(my_interceptor_1), \
  ...      hk.experimental.intercept_methods(my_interceptor_2):
  ...   _ = module(jnp.ones([1, 1]))
  running my_interceptor_1
  - module.name:  method_context_example
  - method_name:  __call__

  Attributes:
    module: A :class:`~haiku.Module` instance whose method is being called.
    method_name: The name of the method being called on the module.
    orig_method: The underlying method on the module which when called will
      *not* trigger interceptors. You should only call this if you want to
      short circuit all the other interceptors, in general you should prefer to
      call the ``next_fun`` passed to your interceptor which will run
      ``orig_method`` after running all other interceptors.
  """

  module: "Module"
  method_name: str
  orig_method: Callable[..., Any]


Args = Tuple[Any]
Kwargs = Dict[str, Any]
NextGetter = Callable[..., Any]
MethodGetter = Callable[[NextGetter, Args, Kwargs, MethodContext], Any]
interceptor_stack = ThreadLocalStack()  # type: ThreadLocalStack[MethodGetter]


def intercept_methods(interceptor: MethodGetter):
  """Register a new method interceptor.

  Method interceptors allow you to (at a distance) intercept method calls to
  modules and modify args/kwargs before calling the underlying method. After the
  underlying method is called you can modify its result before it is passed back
  to the user.

  For example you could intercept method calls to :class:`~haiku.BatchNorm` and
  ensure it is always computed in full precision:

  >>> def my_interceptor(next_f, args, kwargs, context):
  ...   if (type(context.module) is not hk.BatchNorm
  ...       or context.method_name != "__call__"):
  ...     # We ignore methods other than BatchNorm.__call__.
  ...     return next_f(*args, **kwargs)
  ...
  ...   def cast_if_array(x):
  ...     if isinstance(x, jnp.ndarray):
  ...       x = x.astype(jnp.float32)
  ...     return x
  ...
  ...   args, kwargs = jax.tree_map(cast_if_array, (args, kwargs))
  ...   out = next_f(*args, **kwargs)
  ...   return out

  We can create and use our module in the usual way, we just need to wrap any
  method calls we want to intercept in the context manager:

  >>> mod = hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True)
  >>> x = jnp.ones([], jnp.bfloat16)
  >>> with hk.experimental.intercept_methods(my_interceptor):
  ...   out = mod(x, is_training=True)
  >>> assert out.dtype == jnp.float32

  Without the interceptor BatchNorm would compute in bf16, however since we
  cast `x` before the underlying method is called we compute in f32.

  Args:
    interceptor: A method interceptor.

  Returns:
    Context manager under which the interceptor is active.
  """
  base.assert_context("experimental.intercept_methods")
  return interceptor_stack(interceptor)


def run_interceptors(  # pylint: disable=invalid-name
    bound_method: Callable[..., Any],
    method_name: str,
    self: "Module",
    *args: Args,
    **kwargs: Kwargs,
) -> Any:
  """Runs any method interceptors or the original method."""
  if not interceptor_stack:
    return bound_method(*args, **kwargs)

  ctx = MethodContext(module=self,
                      method_name=method_name,
                      orig_method=bound_method)
  interceptor_stack_copy = interceptor_stack.clone()

  def next_fun(*args, **kwargs):
    if interceptor_stack_copy:
      # NOTE: The `interceptor_fun` may call `next_fun` to trigger the next
      # interceptor (and so on) allowing interceptors to be run in turn.
      interceptor_fun = interceptor_stack_copy.popleft()
      return interceptor_fun(next_fun, args, kwargs, ctx)
    else:
      return bound_method(*args, **kwargs)

  return next_fun(*args, **kwargs)


def wrap_method(method_name, unbound_method):
  """Wraps `method` such that it enters name stack and runs method interceptors.

  Args:
    method_name: The name of the method (e.g. "__call__").
    unbound_method: An unbound method to wrap.

  Returns:
    A function that runs the original method but in a context where parameters
    are reused and modules can be created.
  """
  if not getattr(unbound_method, _APPLY_NAME_SCOPE, True):
    return unbound_method

  @functools.wraps(unbound_method)
  def wrapped(self, *args, **kwargs):
    """Calls the original method with a group name set before and after."""
    if not base.frame_stack:
      raise ValueError(
          "All `hk.Module`s must be initialized inside an `hk.transform`.")

    frame = base.current_frame()
    state = base.ModuleState(module=self, method_name=method_name)
    with frame.module(state), _module_method_call(self, method_name):
      # hk.Module enters the module name scope for all methods.
      module_name = getattr(self, "module_name", None)
      f = functools.partial(unbound_method, self)
      f = functools.partial(run_interceptors, f, method_name, self)
      # TODO(tomhennigan): With omnistaging primitives (like named call) will
      # stage out return values eagerly. For functions that produce non-Array
      # values (e.g. `def is_batched(self, x) -> bool`) a tracer will be
      # returned that might result in a concretization error. For now we only
      # enable named call on __call__ (covering 99% of the interesting usages)
      # with an assumption that __call__ is `f(*) -> Tree[Array]`. Longer term
      # we may want to split static and dynamic results in named call to support
      # other methods.
      if modules_with_named_call and module_name and method_name == "__call__":
        local_name = module_name.split("/")[-1]
        f = named_call.stateful_named_call(f, name=local_name)

      out = f(*args, **kwargs)

      # Notify parent modules about our existence.
      if module_name is not None:
        for module_state in frame.module_stack:
          module_state.module._submodules.add(module_name)  # pylint: disable=protected-access
    return out

  return wrapped


_VALID_IDENTIFIER_R = re.compile(r"^[a-zA-Z_]([a-zA-Z0-9_])*$")
valid_identifier = lambda name: bool(_VALID_IDENTIFIER_R.match(name))


class Module(object, metaclass=ModuleMetaclass):
  """Base class for Haiku modules.

  A Haiku module is a lightweight container for variables and other modules.
  Modules typically define one or more "forward" methods (e.g. ``__call__``)
  which apply operations combining user input and module parameters.

  Modules must be initialized inside a :func:`transform` call.

  For example:

  >>> class AddModule(hk.Module):
  ...   def __call__(self, x):
  ...     w = hk.get_parameter("w", [], init=jnp.ones)
  ...     return x + w

  >>> def forward_fn(x):
  ...   mod = AddModule()
  ...   return mod(x)

  >>> forward = hk.transform(forward_fn)
  >>> x = 1.
  >>> rng = None
  >>> params = forward.init(rng, x)
  >>> forward.apply(params, None, x)
  DeviceArray(2., dtype=float32)
  """

  def __init__(self, name: Optional[str] = None):
    """Initializes the current module with the given name.

    Subclasses should call this constructor before creating other modules or
    variables such that those modules are named correctly.

    Args:
      name: An optional string name for the class. Must be a valid Python
        identifier. If ``name`` is not provided then the class name for the
        current instance is converted to ``lower_snake_case`` and used instead.
    """
    if name is None:
      if hasattr(self, "name") and self.name is not None:
        # Attribute assigned by @dataclass constructor.
        name = self.name
      else:
        name = utils.camel_to_snake(type(self).__name__)
    if not valid_identifier(name):
      raise ValueError(
          "'{}' is not a valid module name (must be a valid Python identifier)"
          .format(name))
    self._submodules = set()
    self.module_name = unique_and_canonical_name(name)
    self.name = self.module_name.split("/")[-1]

  # Support @dataclass annotated modules.
  __post_init__ = __init__

  def params_dict(self) -> Mapping[str, jnp.array]:
    """Returns parameters keyed by name for this module and submodules."""
    if not base.frame_stack:
      raise ValueError(
          "`module.params_dict()` must be used as part of an `hk.transform`.")

    params = {}
    curr_name = self.module_name
    for mod_name, mod_params in base.current_frame().params.items():
      if (mod_name == curr_name
          or mod_name.startswith(curr_name + "/")
          or mod_name in self._submodules):
        for param_name, param in mod_params.items():
          fq_name = mod_name + "/" + param_name
          params[fq_name] = param

    return params


def transparent(method: T) -> T:
  """Decorator to wrap a method, preventing automatic variable scope wrapping.

  By default, all variables and modules created in a method are scoped by the
  module and method names. This is undesirable in some cases. Any method
  decorated with :func:`transparent` will create variables and modules in the
  scope in which it was called.

  Args:
    method: the method to wrap.

  Returns:
    The method, with a flag indicating no name scope wrapping should occur.
  """
  setattr(method, _APPLY_NAME_SCOPE, False)
  return method


def unique_and_canonical_name(name: str) -> str:
  """Returns a canonical name for the given name."""
  frame = base.current_frame()

  # If we are outside init/call then prefix the name with the method name.
  if len(frame.module_stack) > 1:
    # -2 since we are inside the ctor and want to look at the caller state.
    module_state = frame.module_stack.peek(-2)

    # Make sure to include the method name if appropriate.
    method_name = module_state.method_name
    if method_name == "__init__":
      name = "~/" + name
    elif method_name != "__call__":
      name = "~" + method_name + "/" + name

    # Include the parent name.
    parent_module = module_state.module
    parent_name = base.safe_get_module_name(parent_module)
    name = parent_name + "/" + name

  # Test if the user has explicitly numbered this module.
  splits = re.split(r"_(\d+)$", name, 3)
  if len(splits) > 1:
    name, n = splits[0], int(splits[1])
    explicit_n = True
  else:
    n = None
    explicit_n = False

  # Determine a unique name for this module within the current context.
  counters = frame.counter_stack.peek(-2)
  if n is not None:
    counters[name] = max(counters[name], n + 1)
  else:
    n = counters[name]
    counters[name] += 1
  qualified_name = f"{name}_{n}" if explicit_n or n else name

  # Final sanity check that this name has not been used before.
  used_names = frame.used_names_stack.peek(-2)
  if qualified_name in used_names:
    raise ValueError(f"Module name '{qualified_name}' is not unique.")
  used_names.add(qualified_name)

  return qualified_name

MethodHook = Callable[[Module, str], ContextManager[None]]
method_hook_stack = ThreadLocalStack()  # type: ThreadLocalStack[MethodHook]


def hook_methods(method_hook: MethodHook) -> ContextManager[None]:
  """Context manager that registers a given module method_hook."""
  return method_hook_stack(method_hook)


@contextlib.contextmanager
def _module_method_call(module: Module, method_name: str):
  """Context manager that wraps a method being called on a module."""
  with contextlib.ExitStack() as stack:
    for method_hook in method_hook_stack:
      stack.enter_context(method_hook(module, method_name))
    yield
