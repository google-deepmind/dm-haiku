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

import abc
import contextlib
import functools
import inspect
import re
from typing import (Any, Callable, ContextManager, Dict, Mapping, Optional,
                    Text, Tuple, Type, TypeVar)

from haiku._src import base
from haiku._src import data_structures
from haiku._src import utils
import jax.numpy as jnp

ThreadLocalStack = data_structures.ThreadLocalStack
T = TypeVar("T")
_APPLY_NAME_SCOPE = "__haiku_name_scope"


class ModuleMetaclass(abc.ABCMeta):
  """Metaclass for `Module`."""

  def __new__(
      mcs: Type[Type[T]],
      name: Text,
      bases: Tuple[Type[Any], ...],
      clsdict: Dict[Text, Any],
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
            value.fget if not value.fget else with_name_scope(key, value.fget),
            value.fset if not value.fset else with_name_scope(key, value.fset),
            value.fdel if not value.fdel else with_name_scope(key, value.fdel),
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
      method = with_name_scope(method_name, method)
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
    init = with_name_scope("__init__", cls.__init__)
    init(module, *args, **kwargs)

    module._auto_repr = utils.auto_repr(cls, *args, **kwargs)  # pylint: disable=protected-access

    ran_super_ctor = hasattr(module, "module_name")
    if not ran_super_ctor:
      raise ValueError(
          "Constructing an hk.Module without calling the super constructor "
          "is not supported. Add the following as the first line in your "
          "__init__ method:\n\nsuper(%s, self).__init__()" % cls.__name__)

    return module


def with_name_scope(method_name, unbound_method):
  """Wraps `method` such that it enters the module stacks.

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
  def wrapped(module, *args, **kwargs):
    """Calls the original method with a group name set before and after."""
    if not base.frame_stack:
      raise ValueError(
          "All `hk.Module`s must be initialized inside an `hk.transform`.")

    frame = base.current_frame()
    state = base.ModuleState(module=module, method_name=method_name)
    with frame.module(state), _module_method_call(module, method_name):
      # hk.Module enters the module name scope for all methods.
      out = unbound_method(module, *args, **kwargs)

      # Notify parent modules about our existence.
      module_name = getattr(module, "module_name", None)
      if module_name is not None:
        for module_state in frame.module_stack:
          module_state.module._submodules.add(module_name)  # pylint: disable=protected-access
    return out

  return wrapped


_VALID_IDENTIFIER_R = re.compile(r"^[a-zA-Z_]([a-zA-Z0-9_])*$")
valid_identifier = lambda name: bool(_VALID_IDENTIFIER_R.match(name))

_CAMEL_TO_SNAKE_R = re.compile(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")
camel_to_snake = lambda value: _CAMEL_TO_SNAKE_R.sub(r"_\1", value).lower()


class Module(object, metaclass=ModuleMetaclass):
  """Base class for Haiku modules.

  A Haiku module is a lightweight container for variables and other modules.
  Modules typically define one or more "forward" methods (e.g. ``__call__``)
  which apply operations combining user input and module parameters.

  Modules must be initialized inside a `hk.transform` call.

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
  >>> forward.apply(params, x)
  DeviceArray(2., dtype=float32)
  """

  def __init__(self, name: Optional[Text] = None):
    """Initializes the current module with the given name.

    Subclasses should call this constructor before creating other modules or
    variables such that those modules are named correctly.

    Args:
      name: An optional string name for the class. Must be a valid Python
        identifier. If ``name`` is not provided then the class name for the
        current instance is converted to ``lower_snake_case`` and used instead.
    """
    if name is None:
      name = camel_to_snake(type(self).__name__)
    if not valid_identifier(name):
      raise ValueError(
          "'{}' is not a valid module name (must be a valid Python identifier)"
          .format(name))
    self._submodules = set()
    self.module_name = unique_and_canonical_name(name)
    self.name = self.module_name.split("/")[-1]

  def params_dict(self) -> Mapping[base.ParamName, jnp.array]:
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
  decorated with ``transparent`` will create variables and modules in the scope
  in which it was called.

  Args:
    method: the method to wrap.
  Returns:
    The method, with a flag indicating no name scope wrapping should occur.
  """
  setattr(method, _APPLY_NAME_SCOPE, False)
  return method


def unique_and_canonical_name(name: Text) -> Text:
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
