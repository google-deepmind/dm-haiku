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
"""Padding module for Haiku."""

from collections import abc
from collections.abc import Callable, Sequence
import typing
from typing import Any

from haiku._src import utils

PadFn = Callable[[int], tuple[int, int]]


# If you are forking replace this block with `import haiku as hk`.
# pylint: disable=invalid-name
class hk:
  class pad:
    PadFn = PadFn
# pylint: enable=invalid-name


def valid(effective_kernel_size: int) -> tuple[int, int]:
  """No padding."""
  del effective_kernel_size
  return (0, 0)


def same(effective_kernel_size: int) -> tuple[int, int]:
  """Pads such that the output size matches input size for stride=1."""
  return ((effective_kernel_size - 1) // 2, effective_kernel_size // 2)


def full(effective_kernel_size: int) -> tuple[int, int]:
  """Maximal padding whilst not convolving over just padded elements."""
  return (effective_kernel_size - 1, effective_kernel_size - 1)


def causal(effective_kernel_size: int) -> tuple[int, int]:
  """Pre-padding such that output has no dependence on the future."""
  return (effective_kernel_size - 1, 0)


def reverse_causal(effective_kernel_size: int) -> tuple[int, int]:
  """Post-padding such that output has no dependence on the past."""
  return (0, effective_kernel_size - 1)


def create_from_padfn(
    padding: hk.pad.PadFn | Sequence[hk.pad.PadFn],  # pylint: disable=g-bare-generic
    kernel: int | Sequence[int],
    rate: int | Sequence[int],
    n: int,
) -> Sequence[tuple[int, int]]:
  """Generates the padding required for a given padding algorithm.

  Args:
    padding: callable/tuple or a sequence of callables/tuples. The callables
      take an integer representing the effective kernel size (kernel size when
      the rate is 1) and return a sequence of two integers representing the
      padding before and padding after for that dimension. The tuples are
      defined with two elements, padding before and after. If `padding` is a
      sequence it must be of length 1 or `n`.
    kernel: int or sequence of ints of length ``n``. The size of the kernel for
      each dimension. If it is an int it will be replicated for the non channel
      and batch dimensions.
    rate: int or sequence of ints of length ``n``. The dilation rate for each
      dimension. If it is an int it will be replicated for the non channel and
      batch dimensions.
    n: the number of spatial dimensions.

  Returns:
    A sequence of length n containing the padding for each element. These are of
    the form ``[pad_before, pad_after]``.
  """
  # The effective kernel size includes any holes/gaps introduced by the
  # dilation rate. It's equal to kernel_size when rate == 1.
  effective_kernel_size = map(
      lambda kernel, rate: (kernel - 1) * rate + 1,
      utils.replicate(kernel, n, "kernel"), utils.replicate(rate, n, "rate"))
  paddings = map(
      lambda x, y: x(y), utils.replicate(padding, n, "padding"),
      effective_kernel_size)
  return tuple(paddings)


def create_from_tuple(
    padding: tuple[int, int] | Sequence[tuple[int, int]],
    n: int,
) -> Sequence[tuple[int, int]]:
  """Create a padding tuple using partially specified padding tuple."""
  assert padding, "Padding must not be empty."
  if isinstance(padding[0], int):
    padding = (padding,) * n
  elif len(padding) == 1:
    padding = tuple(padding) * n
  elif len(padding) != n:
    raise TypeError(
        f"Padding {padding} must be a Tuple[int, int] or sequence of length 1"
        f" or sequence of length {n}.")
  padding = typing.cast(Sequence[tuple[int, int]], tuple(padding))
  return padding


def is_padfn(padding: hk.pad.PadFn | Sequence[hk.pad.PadFn] | Any) -> bool:  # pylint: disable=g-bare-generic
  """Tests whether the given argument is a single or sequence of PadFns."""
  if isinstance(padding, abc.Sequence):
    padding = padding[0]
  return callable(padding)
