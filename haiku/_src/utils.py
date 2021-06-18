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
"""Misc utility functions."""

import collections
import decimal
import inspect
import pprint
import re
from typing import Any, Sequence, Tuple, Type, TypeVar, Union

import jax
import jax.numpy as jnp

T = TypeVar("T")


def compare_or_false(a, b) -> bool:
  try:
    return bool(a == b)
  except:  # pylint: disable=bare-except
    # Some equality checks might be buggy (e.g. `tf.Tensor == None`), in those
    # cases be defensive and assume `a != b`. Note that an exception is also
    # thrown when a and b are ndarrays of >1 element.
    # TODO(tomhennigan) We could be smarter about comparing ndarrays.
    return False


def auto_repr(cls: Type[Any], *args, **kwargs) -> str:
  """Derives a `__repr__` from constructor arguments of a given class.

      >>> class Foo:
      ...   def __init__(self, x=None, y=42):
      ...      pass
      ...

      >>> auto_repr(Foo, "x")
      "Foo(x='x')"

      >>> auto_repr(Foo, "x", y=21)
      "Foo(x='x', y=21)"

      >>> auto_repr(Foo, None, 42)
      Foo()

  Args:
    cls: a class to derive `__repr__` for.
    *args: positional arguments.
    **kwargs: keyword arguments.

  Returns:
    A string representing a call equivalent to `cls(*args, **kwargs)`.
  """
  argspec = inspect.getfullargspec(cls.__init__)
  arg_names = argspec.args
  # Keep used positionals minus self.
  arg_names = arg_names[1:(len(args) + 1)]
  # Keep used kwargs in the order they appear in argspec.
  arg_names.extend(n for n in argspec.args if n in kwargs)
  arg_values = inspect.getcallargs(cls.__init__, None, *args, **kwargs)  # pylint: disable=deprecated-method

  # Extract default parameter values.
  defaults = argspec.defaults or ()
  defaults = dict(zip(argspec.args[-len(defaults):], defaults))
  is_default = lambda n, v: (n in defaults and compare_or_false(v, defaults[n]))

  names_and_values = [(name + "=", arg_values[name]) for name in arg_names
                      if not is_default(name, arg_values[name])]
  # Add varargs.
  names_and_values.extend(("", arg) for arg in args[len(argspec.args) - 1:])
  # Add varkwargs.
  names_and_values.extend(
      (name + "=", kwargs[name]) for name in kwargs if name not in argspec.args)

  single_line = cls.__name__ + "({})".format(", ".join(
      name + repr(value) for name, value in names_and_values))
  if len(single_line) <= 80:
    return single_line
  else:
    return "{}(\n{},\n)".format(
        cls.__name__,
        indent(4, ",\n".join(fancy_repr(n, v) for n, v in names_and_values)))


def fancy_repr(name: str, value: Any) -> str:
  try:
    repr_value = pprint.pformat(value)
  # C++ obejcts by way of pybind11 may not pprint correctly, but do have repr.
  except TypeError:
    repr_value = repr(value)

  if name:
    repr_value = indent(len(name), repr_value).strip()
  return name + repr_value


def indent(amount: int, s: str) -> str:
  """Indents `s` with `amount` spaces."""
  prefix = amount * " "
  return "\n".join(prefix + line for line in s.splitlines())


def replicate(
    element: Union[T, Sequence[T]],
    num_times: int,
    name: str,
) -> Tuple[T]:
  """Replicates entry in `element` `num_times` if needed."""
  if (isinstance(element, (str, bytes)) or
      not isinstance(element, collections.abc.Sequence)):
    return (element,) * num_times
  elif len(element) == 1:
    return tuple(element * num_times)
  elif len(element) == num_times:
    return tuple(element)
  raise TypeError(
      "{} must be a scalar or sequence of length 1 or sequence of length {}."
      .format(name, num_times))


_SPATIAL_CHANNELS_FIRST = re.compile("^NC[^C]*$")
_SPATIAL_CHANNELS_LAST = re.compile("^N[^C]*C$")
_SEQUENTIAL = re.compile("^((BT)|(TB))[^D]*D$")


def get_channel_index(data_format: str) -> int:
  """Returns the channel index when given a valid data format.

  Args:
    data_format: String, the data format to get the channel index from. Valid
      data formats are spatial (e.g.`NCHW`), sequential (e.g. `BTHWD`),
      `channels_first` and `channels_last`).

  Returns:
    The channel index as an int - either 1 or -1.

  Raises:
    ValueError: If the data format is unrecognised.
  """
  if data_format == "channels_first":
    return 1
  if data_format == "channels_last":
    return -1
  if _SPATIAL_CHANNELS_FIRST.match(data_format):
    return 1
  if _SPATIAL_CHANNELS_LAST.match(data_format):
    return -1
  if _SEQUENTIAL.match(data_format):
    return -1
  raise ValueError(
      "Unable to extract channel information from '{}'. Valid data formats are "
      "spatial (e.g.`NCHW`), sequential (e.g. `BTHWD`), `channels_first` and "
      "`channels_last`).".format(data_format))


def assert_minimum_rank(inputs, rank: int):
  """Asserts the rank of the input is at least `rank`."""
  if inputs.ndim < rank:
    raise ValueError("Input %r must have rank >= %d" % (inputs, rank))


def tree_size(tree) -> int:
  """Sums the sizes of all arrays in a pytree.

  For example given a ResNet50 model:

  >>> f = hk.transform_with_state(lambda x: hk.nets.ResNet50(1000)(x, True))
  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.ones([128, 224, 224, 3])
  >>> params, state = f.init(rng, x)

  We can count the number of parameters and their size at f32:

  >>> num_params = hk.data_structures.tree_size(params)
  >>> byte_size = hk.data_structures.tree_bytes(params)
  >>> print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')
  25557032 params, size: 102.23MB

  And compare that with casting our parameters to bf16:

  >>> params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
  >>> num_params = hk.data_structures.tree_size(params)
  >>> byte_size = hk.data_structures.tree_bytes(params)
  >>> print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')
  25557032 params, size: 51.11MB

  Args:
    tree: A tree of jnp.ndarrays.

  Returns:
    The total size (number of elements) of the array(s) in the input.
  """
  return sum(x.size for x in jax.tree_leaves(tree))


def tree_bytes(tree) -> int:
  """Sums the size in bytes of all arrays in a pytree.

  Note that this is the minimum size of the array (e.g. for a float32 we need
  at least 4 bytes) however on some accelerators buffers may occupy more memory
  due to padding/alignment constraints.

  For example given a ResNet50 model:

  >>> f = hk.transform_with_state(lambda x: hk.nets.ResNet50(1000)(x, True))
  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.ones([128, 224, 224, 3])
  >>> params, state = f.init(rng, x)

  We can count the number of parameters and their size at f32:

  >>> num_params = hk.data_structures.tree_size(params)
  >>> byte_size = hk.data_structures.tree_bytes(params)
  >>> print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')
  25557032 params, size: 102.23MB

  And compare that with casting our parameters to bf16:

  >>> params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
  >>> num_params = hk.data_structures.tree_size(params)
  >>> byte_size = hk.data_structures.tree_bytes(params)
  >>> print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')
  25557032 params, size: 51.11MB

  Args:
    tree: A tree of jnp.ndarrays.

  Returns:
    The total size in bytes of the array(s) in the input.
  """
  return sum(x.size * x.dtype.itemsize for x in jax.tree_leaves(tree))

_CAMEL_TO_SNAKE_R = re.compile(r"((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))")
camel_to_snake = lambda value: _CAMEL_TO_SNAKE_R.sub(r"_\1", value).lower()


def simple_dtype(dtype) -> str:
  if isinstance(dtype, type):
    dtype = dtype(0).dtype
  dtype = dtype.name
  dtype = dtype.replace("complex", "c")
  dtype = dtype.replace("double", "d")
  dtype = dtype.replace("float", "f")
  dtype = dtype.replace("uint", "u")
  dtype = dtype.replace("int", "s")
  return dtype


def format_array(x: jnp.ndarray) -> str:
  """Formats the given array showing dtype and shape info."""
  return simple_dtype(x.dtype) + "[" + ",".join(map(str, x.shape)) + "]"


def format_bytes(num_bytes) -> str:
  suffix = "B"
  suffixes = ["KB", "MB", "GB", "TB"]
  num_bytes = decimal.Decimal(num_bytes)
  one_thousand = decimal.Decimal(1000)
  while suffixes and num_bytes >= one_thousand:
    num_bytes /= one_thousand
    suffix = suffixes.pop(0)
  return f"{num_bytes:.2f} {suffix}"
