# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
"""Automatic Mixed Precision (AMP) utilities."""

import collections
import contextlib
import threading
from typing import TypeVar, Union

from haiku._src import base
from haiku._src import data_structures
from haiku._src import module
import jmp

T = TypeVar('T')


# If you are forking replace this block with `import haiku as hk`.
# pylint: disable=invalid-name
class hk:
  custom_getter = base.custom_getter
  custom_creator = base.custom_creator
  MethodContext = module.MethodContext
  Module = module.Module
# pylint: enable=invalid-name
# TODO(slebedev): This make the module non-forkable.
Stack = data_structures.Stack[T]
del data_structures

ClassInfo = collections.namedtuple('ClassInfo', 'module,qualname')
ClassInfoOrType = Union[ClassInfo, type[hk.Module]]


def key_for_module(cls: type[hk.Module]) -> ClassInfoOrType:
  """Returns a suitable key for the given module class."""
  if '<locals>' in cls.__qualname__:
    # Some APIs (e.g. `hk.to_module`) are factory functions that create modules.
    # It is not desirable for us to use the qualname in this case since that
    # would associate all class instances created by the factory with a single
    # policy. Instead we use the class object itself, with the assumption that
    # these types are less likely to be created as a side effect of force
    # reloading modules.
    return cls
  else:
    return ClassInfo(cls.__module__, cls.__qualname__)


class _ThreadState(threading.local):
  """Holds per-thread state on mixed precision policies."""

  def __init__(self):
    super().__init__()
    self._installed_interceptor = False
    self._cls_policy: dict[ClassInfoOrType, jmp.Policy] = {}
    self._current_policy = Stack[jmp.Policy]()

  def push_current_policy(self, policy: jmp.Policy):
    return self._current_policy(policy)

  @property
  def has_current_policy(self):
    return bool(self._current_policy)

  @property
  def current_policy(self) -> jmp.Policy:
    return self._current_policy.peek()

  def clear_policy(self, cls: type[hk.Module]):
    key = key_for_module(cls)
    if key in self._cls_policy:
      del self._cls_policy[key]

  def set_policy(self, cls: type[hk.Module], policy: jmp.Policy):
    if not self._installed_interceptor:
      module.intercept_methods_global(_mixed_precision_interceptor)
      self._installed_interceptor = True
    key = key_for_module(cls)
    self._cls_policy[key] = policy

  def get_policy(self, cls: type[hk.Module]) -> jmp.Policy | None:
    key = key_for_module(cls)
    return self._cls_policy.get(key)


_thread_local_state = _ThreadState()


def reset_thread_local_state_for_test():
  global _thread_local_state
  _thread_local_state = _ThreadState()


def current_policy() -> jmp.Policy | None:
  """Retrieves the currently active policy in the current context.

  Returns:
    The currently active mixed precision policy, or ``None``.

  See also:
    - :func:`clear_policy`: Clears any policies associated with a class.
    - :func:`get_policy`: Gets the policy for a given class.
    - :func:`set_policy`: Sets a policy for a given class.
    - :func:`push_policy`: Context manager for setting policies.
  """
  tls = _thread_local_state
  return tls.current_policy if tls.has_current_policy else None


def get_policy(cls: type[hk.Module]) -> jmp.Policy | None:
  """Retrieves the currently active policy for the given class.

  Note that policies applied explicitly to a top level class (e.g. ``ResNet``)
  will be applied implicitly to all child modules (e.g. ``ConvND``) called from
  the parent. This function only returns policies that have been applied
  explicitly (e.g. via :func:`set_policy`).

  Args:
    cls: A Haiku module class.

  Returns:
    A JMP policy that is used for the given class, or ``None`` if one is not
    active.

  See also:
    - :func:`current_policy`: Retrieves the currently active policy (if any).
    - :func:`clear_policy`: Clears any policies associated with a class.
    - :func:`set_policy`: Sets a policy for a given class.
    - :func:`push_policy`: Context manager for setting policies.
  """
  return _thread_local_state.get_policy(cls)


def set_policy(cls: type[hk.Module], policy: jmp.Policy):
  """Uses the given policy for all instances of the module class.

  NOTE: Policies are only applied to modules created in the current thread.

  A mixed precision policy describes how inputs, module parameters and module
  outputs should be cast at runtime. By applying a policy to a given type of
  module, you can control how all instances of that module behave in your
  program.

  For example, you might want to try running a ResNet50 model in a mixture of
  ``float16`` and ``float32`` on GPU to get higher throughput. To do so you can
  apply a mixed precision policy to the ResNet50 type that will create
  parameters in ``float32``, but cast them to ``float16`` before use, along with
  all module inputs:

  >>> policy = jmp.get_policy('params=float32,compute=float16,output=float32')
  >>> hk.mixed_precision.set_policy(hk.nets.ResNet50, policy)
  >>> net = hk.nets.ResNet50(4)
  >>> x = jnp.ones([4, 224, 224, 3])
  >>> print(net(x, is_training=True))
  [[0. 0. 0. 0.]
   [0. 0. 0. 0.]
   [0. 0. 0. 0.]
   [0. 0. 0. 0.]]

  For a fully worked mixed precision example see the imagenet example in Haiku's
  examples directory. This example shows mixed precision on GPU offering a 2x
  speedup in training time with only a small impact on final top-1 accuracy.

  >>> hk.mixed_precision.clear_policy(hk.nets.ResNet50)

  Args:
    cls: A Haiku module class.
    policy: A JMP policy to apply to the module.

  See also:
    - :func:`push_policy`: Context manager for setting policies.
    - :func:`current_policy`: Retrieves the currently active policy (if any).
    - :func:`clear_policy`: Clears any policies associated with a class.
    - :func:`get_policy`: Gets the policy for a given class.
  """
  assert policy is not None, 'To unset policies use clear_policy.'
  _thread_local_state.set_policy(cls, policy)


@contextlib.contextmanager
def push_policy(cls: type[hk.Module], policy: jmp.Policy):
  """Sets the given policy for the given class while the context is active.

  Args:
    cls: A Haiku module class.
    policy: A JMP policy to apply to the module.

  Yields:
    ``None``.

  See also:
    - :func:`clear_policy`: Clears any policies associated with a class.
    - :func:`get_policy`: Gets the policy for a given class.
    - :func:`set_policy`: Sets a policy for a given class.
    - :func:`current_policy`: Retrieves the currently active policy (if any).
  """
  assert policy is not None, 'To unset policies use clear_policy.'

  # Check for trying to push a new policy inside a module method. In theory it
  # is safe to do this when varying the parameter dtype, but we are defensive
  # and ask users to set policies before calling module methods to avoid
  # confusion.
  current_module = base.inside_transform() and base.current_module()
  if (current_module and
      key_for_module(type(current_module)) == key_for_module(cls)):
    raise ValueError(
        'Pushing a policy inside a method on the same class is not supported.')

  old_policy = get_policy(cls)
  set_policy(cls, policy)
  try:
    yield
  finally:
    if old_policy is not None:
      set_policy(cls, old_policy)
    else:
      clear_policy(cls)


def clear_policy(cls: type[hk.Module]):
  """Clears any policy assocated with the given class.

  Args:
    cls: A Haiku module class.

  See also:
    - :func:`current_policy`: Retrieves the currently active policy (if any).
    - :func:`get_policy`: Gets the policy for a given class.
    - :func:`set_policy`: Sets a policy for a given class.
    - :func:`push_policy`: Context manager for setting policies.
  """
  _thread_local_state.clear_policy(cls)


def _mixed_precision_creator(next_creator, shape, dtype, init, context):
  del context
  dtype = _thread_local_state.current_policy.param_dtype
  return next_creator(shape, dtype, init)


def _mixed_precision_getter(next_getter, value, context):
  del context
  value = _thread_local_state.current_policy.cast_to_compute(value)
  return next_getter(value)


def _mixed_precision_interceptor(next_f, args, kwargs,
                                 context: hk.MethodContext):
  """Method interceptor used to apply mixed precision policies to classes."""
  policy = get_policy(type(context.module))
  if policy is None:
    return next_f(*args, **kwargs)

  ctx = contextlib.ExitStack()
  with ctx:
    if not _thread_local_state.has_current_policy:
      ctx.enter_context(hk.custom_creator(_mixed_precision_creator))
      ctx.enter_context(hk.custom_getter(_mixed_precision_getter, state=True))
    ctx.enter_context(_thread_local_state.push_current_policy(policy))

    args, kwargs = policy.cast_to_compute((args, kwargs))
    out = next_f(*args, **kwargs)
    return policy.cast_to_output(out)
