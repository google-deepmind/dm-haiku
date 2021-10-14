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

import contextlib
import sys
import threading
import types
from typing import Dict, Type, Optional

from haiku._src import base
from haiku._src import data_structures
from haiku._src import module
import jmp

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType('haiku')
hk.custom_getter = base.custom_getter
hk.custom_creator = base.custom_creator
hk.intercept_methods = module.intercept_methods
hk.MethodContext = module.MethodContext
hk.Module = module.Module
Stack = data_structures.Stack
del base, data_structures, module


class _ThreadState(threading.local):
  """Holds per-thread state on mixed precision policies."""

  def __init__(self):
    super().__init__()
    self._interceptor = None
    self._cls_policy = {}  # type: Dict[Type[hk.Module], jmp.Policy]
    self._current_policy = Stack()  # type: Stack[jmp.Policy]

  def push_current_policy(self, policy: jmp.Policy):
    return self._current_policy(policy)

  @property
  def has_current_policy(self):
    return bool(self._current_policy)

  @property
  def current_policy(self) -> jmp.Policy:
    return self._current_policy.peek()

  def clear_policy(self, cls: Type[hk.Module]):
    if cls in self._cls_policy:
      del self._cls_policy[cls]

  def set_policy(self, cls: Type[hk.Module], policy: jmp.Policy):
    if self._interceptor is None:
      self._interceptor = hk.intercept_methods(_mixed_precision_interceptor)
      self._interceptor.__enter__()
    self._cls_policy[cls] = policy

  def get_policy(self, cls: Type[hk.Module]) -> Optional[jmp.Policy]:
    return self._cls_policy.get(cls)

  def __del__(self):
    if self._interceptor is not None:
      self._interceptor.__exit__(*sys.exc_info())
      del self._interceptor

_thread_local_state = _ThreadState()


def set_policy(cls: Type[hk.Module], policy: jmp.Policy):
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
  >>> net(x, is_training=True)
  DeviceArray([[nan, nan, nan, nan],
               [nan, nan, nan, nan],
               [nan, nan, nan, nan],
               [nan, nan, nan, nan]], dtype=float32)

  Oh no, nan! This is because modules like batch norm are not numerically stable
  in ``float16``. To address this, we apply a second policy to our batch norm
  modules to keep them in full precision. We are careful to return a ``float16``
  output from the module such that subsequent modules receive ``float16`` input:

  >>> policy = jmp.get_policy('params=float32,compute=float32,output=float16')
  >>> hk.mixed_precision.set_policy(hk.BatchNorm, policy)
  >>> net(x, is_training=True)
  DeviceArray([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]], dtype=float32)

  For a fully worked mixed precision example see the imagenet example in Haiku's
  examples directory. This example shows mixed precision on GPU offering a 2x
  speedup in training time with only a small impact on final top-1 accuracy.

  >>> hk.mixed_precision.clear_policy(hk.nets.ResNet50)
  >>> hk.mixed_precision.clear_policy(hk.BatchNorm)

  Args:
    cls: A Haiku module class.
    policy: A JMP policy to apply to the module.

  See Also:
    :func:`clear_policy`: Clears any policies associated with a class.
  """
  assert policy is not None, 'To unset policies use clear_policy.'
  _thread_local_state.set_policy(cls, policy)


def clear_policy(cls: Type[hk.Module]):
  """Clears any policy assocated with the given class.

  Args:
    cls: A Haiku module class.

  See Also:
    :func:`set_policy`: Sets a policy for a given class.
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
  policy = _thread_local_state.get_policy(type(context.module))
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
