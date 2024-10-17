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

from collections.abc import Callable, Mapping
import contextlib
import functools
import inspect
import re
from typing import Any, ContextManager, NamedTuple, Protocol, TypeVar

from haiku._src import base
from haiku._src import config
from haiku._src import data_structures
from haiku._src import utils
import jax
import jax.numpy as jnp


T = TypeVar("T")

ThreadLocalStack = data_structures.ThreadLocalStack[T]

_APPLY_NAME_SCOPE = "__haiku_name_scope"
_CUSTOM_NAME = "__haiku_custom_name"


class Future:
  """Represents a value that will be produced eventually."""

  def __init__(self):
    self._result_set = False
    self._result = None

  def set_result(self, result: type["Module"]):
    if self._result_set:
      raise ValueError("Result already set.")
    self._result_set = True
    self._result = result

  def result(self) -> type["Module"]:
    if not self._result_set:
      raise ValueError("Result not set.")
    return self._result


# We subclass `type(Protocol)` in order to avoid metaclass conflicts when
# defining modules that also inherit from `Protocol`. Note that `type(Protocol)`
# already inherits from `abc.ABCMeta`.
class ModuleMetaclass(type(Protocol)):
  """Metaclass for `Module`."""

  def __new__(  # pylint: disable=bad-classmethod-argument
      mcs: type[type[T]],
      name: str,
      bases: tuple[type[Any], ...],
      clsdict: dict[str, Any],
  ) -> type[T]:
    method_names = []
    cls_fut = Future()

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
        p = value
        clsdict[key] = property(
            p.fget if not p.fget else wrap_method(key, p.fget, cls_fut.result),
            p.fset if not p.fset else wrap_method(key, p.fset, cls_fut.result),
            p.fdel if not p.fdel else wrap_method(key, p.fdel, cls_fut.result),
            doc=value.__doc__)

      elif inspect.isfunction(value):
        # We defer patching methods until after the type is created such that we
        # can trigger the descriptor binding them to the class.
        method_names.append(key)

    # Without this any classes with @abc.abstract* elements in their dict fail
    # isinstance checks.
    # TODO(b/177339347): Remove this workaround once the underlying bug in
    #                    `_ProtocolMeta.__instancecheck__` has been fixed.
    clsdict.setdefault("_is_protocol", False)

    clsdict.setdefault(
        "__repr__",
        lambda module: module._auto_repr)  # pylint: disable=protected-access

    cls = super().__new__(mcs, name, bases, clsdict)

    # Provides access to the class object to method interceptors.
    cls_fut.set_result(cls)

    for method_name in method_names:
      # Note: the code below is subtle, we need to ensure that we're wrapping
      # the method bound to the class. In some cases (e.g. `wrapt`) this is
      # important since the method can trigger different behavior when it is
      # bound (e.g. in wrapt `FunctionWrapper.__get__(None, cls)` produces a
      # `BoundFunctionWrapper` which in turn populates the `instance` argument
      # to decorator functions using args[0]).
      # Equivalent to: `cls.__dict__[method_name].__get__(None, cls)`
      method = getattr(cls, method_name)
      method = wrap_method(method_name, method, cls_fut.result)
      setattr(cls, method_name, method)

    return cls

  def __call__(cls, *args, **kwargs) -> Any:  # pylint: disable=no-self-argument
    # Call new such that we have an un-initialized module instance that we can
    # still reference even if there is an exception during __init__. This is
    # needed such that we can make sure the name_scope constructed in __init__
    # is closed even if there is an exception.

    # NOTE: We disable pytype since (somewhat surprisingly) this method is bound
    # with the new class and not the metaclass.
    module = cls.__new__(cls, *args, **kwargs)  # pytype: disable=wrong-arg-types

    # We populate _auto_repr before `__init__` to allow `repr(self)` during the
    # constructor of the module.
    if (config.get_config().module_auto_repr and
        getattr(module, "AUTO_REPR", True)):
      module_repr = utils.auto_repr(cls, *args, **kwargs)  # pylint: disable=protected-access
    else:
      module_repr = object.__repr__(module)

    # Avoid triggering user defined __setattr__ overrides since we have not yet
    # run their constructor.
    object.__setattr__(module, "_auto_repr", module_repr)

    # Now attempt to initialize the object.
    init = wrap_method("__init__", cls.__init__, lambda: cls)
    init(module, *args, **kwargs)

    ran_super_ctor = hasattr(module, "module_name")
    if not ran_super_ctor:
      raise ValueError(
          "Constructing an hk.Module without calling the super constructor "
          "is not supported. Add the following as the first line in your "
          "__init__ method:\n\nsuper(%s, self).__init__()" % cls.__name__)

    return module

  @property
  def __signature__(cls):  # pylint: disable=no-self-argument
    signature = inspect.signature(cls.__init__)
    params = tuple(signature.parameters.values())
    return signature.replace(parameters=params[1:])


class MethodContext(NamedTuple):
  r"""Read only state showing the calling context for a method.

  For example, let's define two interceptors and print the values in the
  context. Additionally, we will make the first interceptor conditionally short
  circuit, since interceptors stack and are run in order, an earlier interceptor
  can decide to call the next interceptor, or short circuit and call the
  underlying method directly:

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

  >>> with hk.intercept_methods(my_interceptor_1), \
  ...      hk.intercept_methods(my_interceptor_2):
  ...   _ = module(jnp.ones([1, 1]))
  running my_interceptor_1
  - module.name:  method_context_example
  - method_name:  __call__
  running my_interceptor_2
  - module.name:  method_context_example
  - method_name:  __call__

  Setting ``short_circuit=True`` will cause the first interceptor to call the
  original method (rather than ``next_fun`` which will trigger the next
  interceptor):

  >>> short_circuit = True
  >>> with hk.intercept_methods(my_interceptor_1), \
  ...      hk.intercept_methods(my_interceptor_2):
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
    orig_class: The class which defined `orig_method`. Note that when
      using inheritance this is not necessarily the same as `type(module)`.
  """

  module: "Module"
  method_name: str
  orig_method: Callable[..., Any]
  orig_class: type["Module"]


Args = tuple[Any]
Kwargs = dict[str, Any]
NextGetter = Callable[..., Any]
MethodGetter = Callable[[NextGetter, Args, Kwargs, MethodContext], Any]
interceptor_stack: ThreadLocalStack[MethodGetter] = ThreadLocalStack()


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
  ...     if isinstance(x, jax.Array):
  ...       x = x.astype(jnp.float32)
  ...     return x
  ...
  ...   args, kwargs = jax.tree.map(cast_if_array, (args, kwargs))
  ...   out = next_f(*args, **kwargs)
  ...   return out

  We can create and use our module in the usual way, we just need to wrap any
  method calls we want to intercept in the context manager:

  >>> mod = hk.BatchNorm(decay_rate=0.9, create_scale=True, create_offset=True)
  >>> x = jnp.ones([], jnp.bfloat16)
  >>> with hk.intercept_methods(my_interceptor):
  ...   out = mod(x, is_training=True)
  >>> assert out.dtype == jnp.float32

  Without the interceptor BatchNorm would compute in bf16, however since we
  cast `x` before the underlying method is called we compute in f32.

  Args:
    interceptor: A method interceptor.

  Returns:
    Context manager under which the interceptor is active.
  """
  return interceptor_stack(interceptor)


def intercept_methods_global(interceptor: MethodGetter):
  interceptor_stack.pushleft(interceptor)


def run_interceptors(  # pylint: disable=invalid-name
    bound_method: Callable[..., Any],
    method_name: str,
    self: "Module",
    orig_class: type["Module"],
    *args: Args,
    **kwargs: Kwargs,
) -> Any:
  """Runs any method interceptors or the original method."""
  if not interceptor_stack:
    return bound_method(*args, **kwargs)

  ctx = MethodContext(module=self,
                      method_name=method_name,
                      orig_method=bound_method,
                      orig_class=orig_class)
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


def simulate_module_method(module, method_name):
  frame = base.current_frame()
  state = base.ModuleState(module=module, method_name=method_name)
  return frame.module(state)


class NameScope:
  """Context manager that when active adds a new name in the hierarcy."""

  def __init__(self, name: str, method_name: str):
    if not name or name[0] == "/":
      raise ValueError("Name scopes must not start with /")

    parts = [name] if name.startswith(OVERRIDE_PREFIX) else name.split("/")
    module = None
    with contextlib.ExitStack() as stack:
      for subname in parts:
        module = NameScopeModule(name=subname)
        stack.enter_context(simulate_module_method(module, method_name))

    self.__entered = False
    self.__module = module
    self.__method = method_name
    self.__stack = contextlib.ExitStack()

  def __enter__(self):
    if self.__stack is None:
      raise ValueError("name_scope is not reusable")
    if self.__entered:
      raise ValueError("name_scope is not reentrant")
    self.__entered = True
    self.__stack.enter_context(simulate_module_method(self.__module,
                                                      self.__method))

  def __exit__(self, exc_type, exc_value, traceback):
    try:
      return self.__stack.__exit__(exc_type, exc_value, traceback)
    finally:
      self.__entered = False
      self.__stack = None


def name_scope(
    name: str,
    *,
    method_name: str = "__call__",
) -> ContextManager[None]:
  """Context manager which adds a prefix to all new modules, params or state.

  >>> with hk.name_scope("my_name_scope"):
  ...   net = hk.Linear(1, name="my_linear")
  >>> net.module_name
  'my_name_scope/my_linear'

  When used inside a module, any submodules, parameters or state created inside
  the name scope will have a prefix added to their names:

  >>> class MyModule(hk.Module):
  ...   def __call__(self, x):
  ...     with hk.name_scope("my_name_scope"):
  ...       submodule = hk.Linear(1, name="submodule")
  ...       w = hk.get_parameter("w", [], init=jnp.ones)
  ...     return submodule(x) + w

  >>> f = hk.transform(lambda x: MyModule()(x))
  >>> params = f.init(jax.random.PRNGKey(42), jnp.ones([1, 1]))
  >>> jax.tree.map(jnp.shape, params)
  {'my_module/my_name_scope': {'w': ()},
   'my_module/my_name_scope/submodule': {'b': (1,), 'w': (1, 1)}}

  Name scopes are very similar to putting all of the code inside the context
  manager inside a method on a :class:`Module` with the name you provide. Behind
  the scenes this is precisely how name scopes are implemented.

  If you are familiar with TensorFlow then Haiku's :func:`name_scope` is similar
  to ``tf.variable_scope(..)`` in TensorFlow 1 and ``tf.name_scope(..)`` in
  TensorFlow 1 and 2 in that it changes the names associated with modules,
  parameters and state.

  Args:
    name: The name scope to use (e.g. ``"foo"`` or ``"foo/bar"``).
    method_name: (Advanced uses only). Since name scopes are equivalent to
      calling methods on modules the method name attribute allows you to specify
      which method name you want to simulate. Most users should leave this as
      the default value (`"__call__"`).

  Returns:
    A single use context manager that when active prefixes new modules,
    parameters or state with the given name.
  """
  base.assert_context("name_scope")
  return NameScope(name, method_name)


def wrap_method(method_name, unbound_method, cls_resolver):
  """Wraps `method` such that it enters name stack and runs method interceptors.

  Args:
    method_name: The name of the method (e.g. "__call__").
    unbound_method: An unbound method to wrap.
    cls_resolver: A callable the returns the Module subclass which defined this
      method.

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

    # Submodules are associated with this method. We allow users to associate
    # submodules with a different method than the one being called via
    # `@name_like("other_method")`. Interceptors and custom getters are still
    # provided the actual method name (e.g. "submodule_method_name" is only used
    # for naming submodules).
    submodule_method_name = getattr(unbound_method, _CUSTOM_NAME, method_name)

    frame = base.current_frame()
    state = base.ModuleState(module=self, method_name=submodule_method_name)
    with frame.module(state), _module_method_call(self, method_name):
      # hk.Module enters the module name scope for all methods.
      module_name = getattr(self, "module_name", None)
      orig_class = cls_resolver()
      f = functools.partial(unbound_method, self)
      f = functools.partial(run_interceptors, f, method_name, self,
                            orig_class)
      if module_name:
        local_module_name = module_name.split("/")[-1]
        f = jax.named_call(f, name=local_module_name)
        if method_name != "__call__":
          f = jax.named_call(f, name=method_name)

      out = f(*args, **kwargs)

      # Module names are set in the constructor. If `f` is the constructor then
      # its name will only be set **after** `f` has run. For methods other
      # than `__init__` we need the name before running in order to wrap their
      # execution with `named_call`.
      if module_name is None:
        module_name = getattr(self, "module_name", None)

      # Notify parent modules about our existence.
      if module_name is not None:
        for module_state in frame.module_stack:
          if module_state.module is not self:
            module_state.module._submodules.add(module_name)  # pylint: disable=protected-access
    return out

  return wrapped


_VALID_IDENTIFIER_R = re.compile(r"^[a-zA-Z_]([a-zA-Z0-9_])*$")
valid_identifier = lambda name: bool(_VALID_IDENTIFIER_R.match(name))


def name_and_number(name: str) -> tuple[str, int | None]:
  splits = re.split(r"_(0|[1-9]\d*)$", name, 3)
  if len(splits) > 1:
    return splits[0], int(splits[1])
  else:
    return name, None


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
  name, n = name_and_number(name)
  explicit_n = n is not None

  # Determine a unique name for this module within the current context.
  if n is None:
    n = next_module_number(name)
  name = f"{name}_{n}" if explicit_n or n else name

  # Final sanity check that this name has not been used before.
  reserve_module_name(name, check_unique=True)

  return name


def reserve_module_name(name: str, *, check_unique: bool):
  """Reserves the given module name."""
  frame = base.current_frame()
  used_names = frame.used_names_stack.peek(-2)
  if check_unique and name in used_names:
    raise ValueError(f"Module name '{name}' is not unique.")
  used_names.add(name)

  name, number = name_and_number(name)
  if number is None:
    number = 0
  counters = frame.counter_stack.peek(-2)
  counters[name] = max(counters[name], number + 1)


def next_module_number(name: str) -> int:
  frame = base.current_frame()
  counters = frame.counter_stack.peek(-2)
  return counters[name]


# NOTE: Since `:` is not a valid symbol in a module name (it has been rejected
# by check_name since the first version of Haiku) we know that no existing users
# have this name so it is a safe token.
OVERRIDE_PREFIX = "FORCE:"


def force_name(name: str) -> str:
  """Forces Haiku to use this name, ignoring all context information.

  NOTE: This method is intended for advanced use cases only and should be
  avoided whenever possible as it effectively enforces a singleton pattern when
  setting absolute names.

  Haiku names modules according to where they are created (e.g. the stack of
  modules that created them, or the current :func:`~haiku.name_scope`). This
  function allows you to create modules that ignore all of this and have
  precisely the name you provide.

  This might be useful in the case that you have two modules and you want to
  force them to share parameters:

  >>> mod0 = hk.Linear(1)
  >>> some_hyperparameter = True
  >>> if some_hyperparameter:
  ...   # Force mod1 and mod0 to have shared weights.
  ...   mod1 = hk.Linear(1, name=hk.force_name(mod0.module_name))
  ... else:
  ...   # mod0 and mod1 are independent.
  ...   mod1 = hk.Linear(1)

  (A simpler version of this snippet would do `mod1 = mod0` instead of using
  force_name, however in real examples it can be simpler to use force_name,
  especially in cases where you may not have access to the module instance
  without lots of plumbing, but getting the module name is easy [e.g. it is a
  hyperparameter]).

  Args:
    name: String name for the module. For example ``"foo"`` or ``"foo/bar"``.

  Returns:
    A value suitable to pass into the ``name`` argument of any Haiku module
    constructor.
  """
  return f"{OVERRIDE_PREFIX}{name}"


def check_name(component: str, name: str, allow_leading_tilde: bool = False):
  if allow_leading_tilde and component.startswith("~"):
    component = component[1:]
    if not component:
      # "~" is a valid component name (e.g. "foo/~/bar" is a valid name).
      return

  if not valid_identifier(component):
    raise ValueError(f"'{name}' is not a valid module name (must be a "
                     "valid Python identifier)")


class Module(metaclass=ModuleMetaclass):
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
  >>> print(forward.apply(params, None, x))
  2.0
  """

  def __init__(self, name: str | None = None):
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

    if name.startswith(OVERRIDE_PREFIX):
      name = name[len(OVERRIDE_PREFIX):]
      for component in name.split("/"):
        check_name(component, name, allow_leading_tilde=True)
      reserve_module_name(name, check_unique=False)
    else:
      check_name(name, name)
      name = unique_and_canonical_name(name)

    self._submodules: set[str] = set()
    self.module_name = name
    self.name = self.module_name.split("/")[-1]
    self._creation_frame_id = base.current_frame().frame_id

  # Support @dataclass annotated modules.
  __post_init__ = __init__

  def params_dict(self) -> Mapping[str, jnp.ndarray]:
    """Returns parameters keyed by name for this module and submodules."""
    if not base.frame_stack:
      raise ValueError(
          "`module.params_dict()` must be used as part of an `hk.transform`.")

    return params_or_state_dict(self.module_name, self._submodules, "params")

  def state_dict(self) -> Mapping[str, jnp.ndarray]:
    """Returns state keyed by name for this module and submodules."""
    if not base.frame_stack:
      raise ValueError(
          "`module.state_dict()` must be used as part of an `hk.transform`.")

    return params_or_state_dict(self.module_name, self._submodules, "state")


def params_or_state_dict(
    module_name: str,
    submodules: set[str],
    which: str,
) -> Mapping[str, jnp.ndarray]:
  """Returns module parameters or state for the given module or submodules."""
  assert which in ("params", "state")
  out = {}
  frame = base.current_frame()
  for their_module_name, bundle in getattr(frame, which).items():
    if (their_module_name == module_name
        or their_module_name.startswith(module_name + "/")
        or their_module_name in submodules):
      for name, value in bundle.items():
        fq_name = their_module_name + "/" + name
        out[fq_name] = value.current if which == "state" else value
  return out


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


def name_like(method_name: str) -> Callable[[T], T]:
  """Allows a method to be named like some other method.

  In Haiku submodules are named based on the name of their parent module and the
  method in which they are created. When refactoring code it may be desirable to
  maintain previous names in order to keep checkpoint compatibility, this can be
  achieved using :func:`name_like`.

  As an example, consider the following toy autoencoder:

  >>> class Autoencoder(hk.Module):
  ...   def __call__(self, x):
  ...     z = hk.Linear(10, name="enc")(x)  # name: autoencoder/enc
  ...     y = hk.Linear(10, name="dec")(z)  # name: autoencoder/dec
  ...     return y

  If we want to refactor this such that users can encode or decode, we would
  create two methods (encode, decode) which would create and apply our modules.
  In order to retain checkpoint compatibility with the original module we can
  use :func:`name_like` to name those submodules as if they were created inside
  ``__call__``:

  >>> class Autoencoder(hk.Module):
  ...   @hk.name_like("__call__")
  ...   def encode(self, x):
  ...     return hk.Linear(10, name="enc")(x)  # name: autoencoder/enc
  ...
  ...   @hk.name_like("__call__")
  ...   def decode(self, z):
  ...     return hk.Linear(10, name="dec")(z)  # name: autoencoder/dec
  ...
  ...   def __call__(self, x):
  ...     return self.decode(self.encode(x))

  One sharp edge is if users rely on Haiku's numbering to take care of giving
  unique names and refactor using :func:`name_like`. For example when
  refactoring the following:

  >>> class Autoencoder(hk.Module):
  ...   def __call__(self, x):
  ...     y = hk.Linear(10)(z)  # name: autoencoder/linear_1
  ...     z = hk.Linear(10)(x)  # name: autoencoder/linear
  ...     return y

  To use :func:`name_like`, the unnamed linear modules in encode/decode will end
  up with the same name (both: ``autoencoder/linear``) because module numbering
  is only applied within a method:

  >>> class Autoencoder(hk.Module):
  ...   @hk.name_like("__call__")
  ...   def encode(self, x):
  ...     return hk.Linear(10)(x)  # name: autoencoder/linear
  ...
  ...   @hk.name_like("__call__")
  ...   def decode(self, z):
  ...     return hk.Linear(10)(z)  # name: autoencoder/linear  <-- NOT INTENDED

  To fix this case you need to explicitly name the modules within the method
  with their former name:

  >>> class Autoencoder(hk.Module):
  ...   @hk.name_like("__call__")
  ...   def encode(self, x):
  ...     return hk.Linear(10, name="linear")(x)    # name: autoencoder/linear
  ...
  ...   @hk.name_like("__call__")
  ...   def decode(self, z):
  ...     return hk.Linear(10, name="linear_1")(z)  # name: autoencoder/linear_1

  Args:
    method_name: The name of a method whose name we should adopt. This method
      does not actually have to be defined on the class.

  Returns:
    A decorator that when applied to a method marks it as having a different
    name.
  """
  def decorator(method: T) -> T:
    setattr(method, _CUSTOM_NAME, method_name)
    return method
  return decorator

MethodHook = Callable[[Module, str], ContextManager[None]]
method_hook_stack: ThreadLocalStack[MethodHook] = ThreadLocalStack()


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


class NameScopeModule(Module):
  pass
