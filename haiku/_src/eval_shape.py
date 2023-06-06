# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Functions for evaluating shapes."""

from unittest import mock

from haiku._src import base
from haiku._src import basic
from haiku._src import stateful
import jax
import jax.numpy as jnp


def zeros_creator(next_creator, shape, dtype, init, context):
  del context
  init = jnp.zeros
  return next_creator(shape, dtype, init)


def noop_dropout(rng, rate, x, broadcast_dims=()):
  del rng, rate, broadcast_dims
  return x


def fast_eval_shape(fun, *args, **kwargs):
  """Equivalent to ``eval_shape`` in JAX.

  This utility is equivalent to ``eval_shape`` in JAX except that it avoids
  running Haiku functions whose shapes are trivially known. This can avoid some
  Python overheads in JAX which can accumulate for very large models.

  Optimizations:

  * All parameter/state initialisers replaced with zeros.
  * ``hk.dropout`` replaced with identity.
  * ``jax.random.fold_in`` replaced with identity.

  Args:
    fun: The function to trace.
    *args: Positional arguments to ``fun``.
    **kwargs: Keyword arguments to ``fun``.

  Returns:
    The shape produced by ``fun`` for the given args/kwargs.
  """
  with base.custom_creator_unsafe(zeros_creator), \
       mock.patch.object(basic, 'dropout_impl', noop_dropout), \
       mock.patch.object(jax.random, 'fold_in', lambda key, data: key):
    if base.inside_transform():
      return stateful.eval_shape(fun, *args, **kwargs)
    else:
      return jax.eval_shape(fun, *args, **kwargs)
