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
"""Misc utility functions."""

import collections
import inspect
import pprint
import re
from typing import Any, Sequence, Text, Tuple, Type, TypeVar, Union

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


def auto_repr(cls: Type[Any], *args, **kwargs) -> Text:
  """Derives a `__repr__` from constructor arguments of a given class.

      >>> class Foo(object):
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


def fancy_repr(name: Text, value: Any) -> Text:
  try:
    repr_value = pprint.pformat(value)
  # C++ obejcts by way of pybind11 may not pprint correctly, but do have repr.
  except TypeError:
    repr_value = repr(value)

  if name:
    repr_value = indent(len(name), repr_value).strip()
  return name + repr_value


def indent(amount: int, s: Text) -> Text:
  """Indents `s` with `amount` spaces."""
  prefix = amount * " "
  return "\n".join(prefix + line for line in s.splitlines())


def replicate(
    element: Union[T, Sequence[T]],
    num_times: int,
    name: Text,
) -> Tuple[T]:
  """Replicates entry in `element` `num_times` if needed."""
  if (isinstance(element, (str, bytes)) or
      not isinstance(element, collections.Sequence)):
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


def get_channel_index(data_format: Text) -> int:
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
