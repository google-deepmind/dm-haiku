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
"""Utilities for working with random numbers."""

import contextlib
import functools

from haiku._src import base
from haiku._src import data_structures
import jax


@contextlib.contextmanager
def count_hk_rngs_requested():
  """Context manager counting calls to next_rng_key."""
  # TODO(tomhennigan): Catch keys from `next_rng_keys`, `maybe_next_rng_key`, ..
  # TODO(tomhennigan): Don't include keys returned under a `with_rng` context.
  # TODO(tomhennigan): Optimize use of keys within a `with_rng` heading as well.
  count = [0]
  orig_next_rng_key = base.next_rng_key_internal
  def counting_next_rng_key(*a, **k):
    count[0] += 1
    return orig_next_rng_key(*a, **k)
  try:
    base.next_rng_key_internal = counting_next_rng_key
    yield lambda: count[0]
  finally:
    base.next_rng_key_internal = orig_next_rng_key


def optimize_rng_use(fun):
  """Optimizes a RNG key splitting in ``fun``.

  Our strategy here is to use abstract interpretation to run your function
  twice, the first time we use :func:`jax.eval_shape` to avoid spending any
  flops and simply observe how many times you call :func:`~haiku.next_rng_key`.
  We then run your function again, but this time we reserve enough RNG keys
  ahead of time such that we only need to call :func:`jax.random.split` once.

  In the following example, we need three random samples for our weight
  matrices in our 3-layer MLP. To draw these samples we use
  :func:`~haiku.next_rng_key` which will split a new key for each sample. By
  using :func:`optimize_rng_use` Haiku will pre-allocate exactly enough RNGs for
  ``f`` to be evaluated by splitting the input key once and only once. For large
  models (unlike this example) this can lead to a significant reduction in
  compilation time for ``init``:

  >>> def f(x):
  ...   net = hk.nets.MLP([300, 100, 10])
  ...   return net(x)
  >>> f = hk.experimental.optimize_rng_use(f)
  >>> f = hk.transform(f)
  >>> params = f.init(jax.random.PRNGKey(42), jnp.ones([1, 1]))

  Args:
    fun: A function to wrap.

  Returns:
    A function that applies ``fun`` but only requires one call to
    :func:`jax.random.split` by Haiku.
  """

  @functools.wraps(fun)
  def wrapper(*args, **kwargs):
    base.assert_context("optimize_rng_use")

    # Extract all current state.
    frame = base.current_frame()
    params = frame.params or None
    if params is not None:
      params = data_structures.to_haiku_dict(params)
    state = frame.state or None
    if state is not None:
      state = base.extract_state(state, initial=True)
    rng = frame.rng_stack.peek()
    if rng is not None:
      rng = rng.internal_state

    def pure_fun(params, state, rng, *args, **kwargs):
      with base.new_context(params=params, state=state, rng=rng):
        return fun(*args, **kwargs)

    with count_hk_rngs_requested() as rng_count_f:
      jax.eval_shape(pure_fun, params, state, rng, *args, **kwargs)
    rng_count = rng_count_f()

    if rng_count:
      base.current_frame().rng_stack.peek().reserve(rng_count)
    return fun(*args, **kwargs)

  return wrapper
