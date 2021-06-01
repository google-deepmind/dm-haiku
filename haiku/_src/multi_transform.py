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
"""Transform a set of Haiku-using functions which use overlapping params."""

# pylint: disable=unnecessary-lambda

import types
from typing import Any, Callable, NamedTuple, Tuple

import dataclasses
from haiku._src import analytics
from haiku._src import transform
from haiku._src import typing
import jax

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType('haiku')
hk.transform_with_state = transform.transform_with_state
hk.Params = typing.Params
hk.State = typing.State
del transform, typing

PyTreeDef = Any
TemplateFn = Callable[..., Any]
TreeOfApplyFns = Any


class MultiTransformed(NamedTuple):
  """Holds a collection of pure functions.

  Attributes:
    init: A pure function: ``params = init(rng, *a, **k)``
    apply: A JAX tree of pure functions each with the signature:
      ``out = apply(params, rng, *a, **k)``.

  See also:
    :class:`Transformed`: Single apply variant of multi-transform.
    :class:`MultiTransformedWithState`: Multi apply with state variant.
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., hk.Params]

  # PyTree[Callable[[hk.Params, Optional[PRNGKey], ...], Any]]
  apply: Any


class MultiTransformedWithState(NamedTuple):
  """Holds a collection of pure functions.

  Attributes:
    init: A pure function: ``params, state = init(rng, *a, **k)``
    apply: A JAX tree of pure functions each with the signature:
      ``out, state = apply(params, state, rng, *a, **k)``.

  See also:
    :class:`TransformedWithState`: Single apply variant of multi-transform.
    :class:`MultiTransformed`: Multi apply with state variant.
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., Tuple[hk.Params, hk.State]]

  # PyTree[Callable[[hk.Params, hk.State, Optional[PRNGKey], ...],
  #                 Tuple[Any, hk.State]]]
  apply: Any


@dataclasses.dataclass
class Box:
  """Holds a Python value and has no leaves."""
  python_value: Any

jax.tree_util.register_pytree_node(
    Box, lambda b: ([], b.python_value), lambda v, _: Box(v))


def multi_transform_with_state(
    f: Callable[[], Tuple[TemplateFn, TreeOfApplyFns]],
) -> MultiTransformedWithState:
  """Transforms a collection of functions using Haiku into pure functions.

  See :func:`multi_transform` for more details.

  Example:

  >>> def f():
  ...   encoder = hk.Linear(1, name="encoder")
  ...   decoder = hk.Linear(1, name="decoder")
  ...
  ...   def init(x):
  ...     z = encoder(x)
  ...     return decoder(z)
  ...
  ...   return init, (encoder, decoder)

  >>> f = hk.multi_transform_with_state(f)
  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.ones([1, 1])
  >>> params, state = f.init(rng, x)
  >>> jax.tree_map(jnp.shape, params)
  {'decoder': {'b': (1,), 'w': (1, 1)},
   'encoder': {'b': (1,), 'w': (1, 1)}}

  >>> encode, decode = f.apply
  >>> z, state = encode(params, state, None, x)
  >>> y, state = decode(params, state, None, z)

  Args:
    f: Function returning a "template" function and an arbitrary
      tree of functions using modules connected in the template function.

  Returns:
    An ``init`` function and a tree of pure ``apply`` functions.

  See also:
    :func:`transform_with_state`: Transform a single apply function.
    :func:`multi_transform`: Transform multiple apply functions without state.
  """
  analytics.log_once('multi_transform_with_state')

  def init_fn(*args, **kwargs):
    """Returns initial state for the transformed functions."""
    return f()[0](*args, **kwargs)

  init_fn = hk.transform_with_state(init_fn).init

  def apply_fn_i(i):
    def apply_fn(*args, **kwargs):
      """Applies the transformed function at the given inputs."""
      return jax.tree_leaves(f()[1])[i](*args, **kwargs)
    return apply_fn

  # We need to find out the structure of f()[1], including how many
  # functions there are, so that we can transform them individually and repack
  # into the same tree structure. It's valid for modules to declare parameters
  # in their constructor, so we need to create something that looks like
  # hk.Params in order to do this. `jax.eval_shape` interprets the function
  # abstractly, ie no real params are created, and we don't need to touch the
  # accelerator. This means hardcoding the RNG below is fine.
  def get_output_treedef() -> Box:
    rng = jax.random.PRNGKey(42)  # This is fine, see above
    fns = hk.transform_with_state(lambda: f()[1])
    apply_fns, _ = fns.apply(*fns.init(rng), rng)
    return Box(jax.tree_structure(apply_fns))

  output_treedef = jax.eval_shape(get_output_treedef).python_value
  apply_fns = make_tree(lambda i: hk.transform_with_state(apply_fn_i(i)).apply,
                        output_treedef)

  return MultiTransformedWithState(init_fn, apply_fns)


def multi_transform(
    f: Callable[[], Tuple[TemplateFn, TreeOfApplyFns]],
) -> MultiTransformed:
  """Transforms a collection of functions using Haiku into pure functions.

  In many scenarios we have several modules which are used either as primitives
  for several Haiku modules/functions, or whose pure versions are to be reused
  in downstream code. This utility enables this by applying
  :func:`transform` to an arbitrary tree of Haiku functions which share modules
  and have a common ``init`` function.

  ``f`` is expected to return a tuple of two elements. First is a ``template``
  Haiku function which provides an example of how all internal Haiku modules are
  connected. This function is used to create a common ``init`` function (with
  your parameters).

  The second object is an arbitrary tree of Haiku functions all of which reuse
  the modules connected in the ``template`` function. These functions are
  transformed to pure ``apply`` functions.

  Example:

  >>> def f():
  ...   encoder = hk.Linear(1, name="encoder")
  ...   decoder = hk.Linear(1, name="decoder")
  ...
  ...   def init(x):
  ...     z = encoder(x)
  ...     return decoder(z)
  ...
  ...   return init, (encoder, decoder)

  >>> f = hk.multi_transform(f)
  >>> rng = jax.random.PRNGKey(42)
  >>> x = jnp.ones([1, 1])
  >>> params = f.init(rng, x)
  >>> jax.tree_map(jnp.shape, params)
  {'decoder': {'b': (1,), 'w': (1, 1)},
   'encoder': {'b': (1,), 'w': (1, 1)}}

  >>> encode, decode = f.apply
  >>> z = encode(params, None, x)
  >>> y = decode(params, None, z)

  Args:
    f: A factory function that returns two functions, firstly a common init
      function that creates all modules, and secondly a pytree of apply
      functions which make use of those modules.

  Returns:
    A :class:`MultiTransformed` instance which contains a pure init function
      that creates all parameters, and a pytree of pure apply functions that
      given the params apply the given function.

  See also:
    :func:`multi_transform_with_state`: Equivalent for modules using state.
  """
  analytics.log_once('multi_transform')

  f = multi_transform_with_state(f)
  f = without_state(f)
  return f


def without_state(f: MultiTransformedWithState) -> MultiTransformed:
  """Converts ``MultiTransformedWithState`` to ``MultiTransformed``."""
  def init_fn(rng, *args, **kwargs) -> hk.Params:
    params, state = f.init(rng, *args, **kwargs)
    if state:
      raise ValueError(
          'If your transformed function uses `hk.{get,set}_state` then use '
          '`hk.multi_transform_with_state`.')
    return params

  def apply_without_state(orig_apply_fn) -> Any:
    def apply_fn(params: hk.Params, rng, *args, **kwargs):
      out, state = orig_apply_fn(params, {}, rng, *args, **kwargs)
      if state:
        raise ValueError(
            'If your transformed function uses `hk.{get,set}_state` then use '
            '`hk.multi_transform_with_state`.')
      return out
    return apply_fn

  apply_fns = jax.tree_map(apply_without_state, f.apply)

  return MultiTransformed(init_fn, apply_fns)


def make_tree(f: Callable[[int], Any], treedef: PyTreeDef):
  leaves = list(map(f, range(treedef.num_leaves)))
  return jax.tree_unflatten(treedef, leaves)
