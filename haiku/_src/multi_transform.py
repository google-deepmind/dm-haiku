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

from collections.abc import Callable
import dataclasses
import functools
import inspect
from typing import Any, NamedTuple, Optional, TypeVar

from haiku._src import analytics
from haiku._src import transform
from haiku._src import typing
import jax


# If you are forking replace this block with `import haiku as hk`.
# pylint: disable=invalid-name
class hk:
  transform_with_state = transform.transform_with_state
  Transformed = transform.Transformed
  TransformedWithState = transform.TransformedWithState
  Params = typing.Params
  State = typing.State
  MutableParams = typing.MutableParams
  MutableState = typing.MutableState
# pylint: enable=invalid-name
# TODO(slebedev): This makes the module non-forkable.
_transform = transform
del transform, typing

TemplateFn = Callable[..., Any]
TreeOfApplyFns = Any


class MultiTransformed(NamedTuple):
  """Holds a collection of pure functions.

  Attributes:
    init: A pure function: ``params = init(rng, *a, **k)``
    apply: A JAX tree of pure functions each with the signature:
      ``out = apply(params, rng, *a, **k)``.

  See also:
    - :class:`Transformed`: Single apply variant of multi-transform.
    - :class:`MultiTransformedWithState`: Multi apply with state variant.
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., hk.MutableParams]

  # PyTree[Callable[[hk.Params, Optional[PRNGKey], ...], Any]]
  apply: Any


class MultiTransformedWithState(NamedTuple):
  """Holds a collection of pure functions.

  Attributes:
    init: A pure function: ``params, state = init(rng, *a, **k)``
    apply: A JAX tree of pure functions each with the signature:
      ``out, state = apply(params, state, rng, *a, **k)``.

  See also:
    - :class:`TransformedWithState`: Single apply variant of multi-transform.
    - :class:`MultiTransformed`: Multi apply with state variant.
  """

  # Args: [Optional[PRNGKey], ...]
  init: Callable[..., tuple[hk.MutableParams, hk.MutableState]]

  # PyTree[Callable[[hk.Params, hk.State, Optional[PRNGKey], ...],
  #                 Tuple[Any, hk.MutableState]]]
  apply: Any


@dataclasses.dataclass
class Box:
  """Holds a Python value and has no leaves."""
  python_value: Any

jax.tree_util.register_pytree_node(
    Box, lambda b: ([], b.python_value), lambda v, _: Box(v))


def multi_transform_with_state(
    f: Callable[[], tuple[TemplateFn, TreeOfApplyFns]],
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
  >>> jax.tree.map(jnp.shape, params)
  {'decoder': {'b': (1,), 'w': (1, 1)},
   'encoder': {'b': (1,), 'w': (1, 1)}}

  >>> encode, decode = f.apply
  >>> z, state = encode(params, state, None, x)
  >>> y, state = decode(params, state, None, z)

  Args:
    f: Function returning a "template" function and an arbitrary tree of
      functions using modules connected in the template function.

  Returns:
    An ``init`` function and a tree of pure ``apply`` functions.

  See also:
    - :func:`transform_with_state`: Transform a single apply function.
    - :func:`multi_transform`: Transform multiple apply functions without state.
  """
  analytics.log_once('multi_transform_with_state')

  def init_fn(*args, **kwargs) -> tuple[hk.MutableParams, hk.MutableState]:
    """Returns initial state for the transformed functions."""
    return f()[0](*args, **kwargs)

  init_fn = hk.transform_with_state(init_fn).init

  def apply_fn_i(i):
    def apply_fn(*args, **kwargs):
      """Applies the transformed function at the given inputs."""
      return jax.tree.leaves(f()[1])[i](*args, **kwargs)
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
    return Box(jax.tree.structure(apply_fns))

  output_treedef = jax.eval_shape(get_output_treedef).python_value
  apply_fns = make_tree(lambda i: hk.transform_with_state(apply_fn_i(i)).apply,
                        output_treedef)

  return MultiTransformedWithState(init_fn, apply_fns)


def multi_transform(
    f: Callable[[], tuple[TemplateFn, TreeOfApplyFns]],
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
  >>> jax.tree.map(jnp.shape, params)
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
  def init_fn(rng, *args, **kwargs) -> hk.MutableParams:
    params, state = f.init(rng, *args, **kwargs)
    if state:
      raise ValueError(
          'If your transformed function uses `hk.{get,set}_state` then use '
          '`hk.multi_transform_with_state`.')
    return params

  def apply_without_state(orig_apply_fn):
    def apply_fn(params: hk.Params, rng, *args, **kwargs):
      out, state = orig_apply_fn(params, None, rng, *args, **kwargs)
      if state:
        raise ValueError(
            'If your transformed function uses `hk.{get,set}_state` then use '
            '`hk.multi_transform_with_state`.')
      return out
    return apply_fn

  apply_fns = jax.tree.map(apply_without_state, f.apply)

  return MultiTransformed(init_fn, apply_fns)

TransformedT = TypeVar('TransformedT',
                       hk.Transformed,
                       hk.TransformedWithState,
                       MultiTransformed,
                       MultiTransformedWithState)


def without_apply_rng(f: TransformedT) -> TransformedT:
  """Removes the rng argument from the apply function.

  This is a convenience wrapper that makes the ``rng`` argument to
  ``f.apply`` default to ``None``. This is useful when ``f`` doesn't actually
  use random numbers as part of its computation, such that the ``rng`` argument
  wouldn't be used. Note that if ``f`` `does` use random numbers, this will
  cause an error to be thrown complaining that ``f`` needs a non-None PRNGKey.

  Args:
    f: A transformed function.

  Returns:
    The same transformed function, with a modified ``apply``.
  """
  def check_rng_kwarg(kwargs):
    if 'rng' in kwargs:
      raise TypeError(
          'Haiku transform adds three arguments (params, state, rng) to apply. '
          'If the functions you are transforming use the same names you must '
          'pass them positionally (e.g. `f.apply(.., my_rng)` and not by '
          'name (e.g. `f.apply(.., rng=my_rng)`)')

  if isinstance(f, hk.TransformedWithState):
    def apply_fn(params, state, *args, **kwargs):
      check_rng_kwarg(kwargs)
      return f.apply(params, state, None, *args, **kwargs)
    apply_fn.__signature__ = _transform.sig_replace_leading_parameters(
        inspect.signature(f.apply), 3, [
            inspect.Parameter(
                'params',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[hk.Params]),
            inspect.Parameter(
                'state',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[hk.State]),
        ])
    f_new = hk.TransformedWithState(init=f.init, apply=apply_fn)
    _transform.tie_in_original_fn(f, f_new.init, f_new.apply)

  elif isinstance(f, hk.Transformed):
    def apply_fn(params, *args, **kwargs):
      check_rng_kwarg(kwargs)
      return f.apply(params, None, *args, **kwargs)
    apply_fn.__signature__ = _transform.sig_replace_leading_parameters(
        inspect.signature(f.apply), 2, [
            inspect.Parameter(
                'params',
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Optional[hk.Params]),
        ])
    f_new = hk.Transformed(init=f.init, apply=apply_fn)
    _transform.tie_in_original_fn(f, f_new.init, f_new.apply)

  elif isinstance(f, MultiTransformedWithState):
    def make_new_apply_fn(apply_fn, params, state, *args, **kwargs):
      check_rng_kwarg(kwargs)
      return apply_fn(params, state, None, *args, **kwargs)
    apply_fn = jax.tree.map(
        lambda fn: functools.partial(make_new_apply_fn, fn), f.apply
    )
    f_new = MultiTransformedWithState(init=f.init, apply=apply_fn)

  elif isinstance(f, MultiTransformed):
    def make_new_apply_fn(apply_fn, params, *args, **kwargs):
      check_rng_kwarg(kwargs)
      return apply_fn(params, None, *args, **kwargs)
    apply_fn = jax.tree.map(
        lambda fn: functools.partial(make_new_apply_fn, fn), f.apply
    )
    f_new = MultiTransformed(init=f.init, apply=apply_fn)

  else:
    raise ValueError('Must be called with the result of `hk.transformed`, '
                     '`hk.multi_transform`, `hk.transformed_with_state` or '
                     '`hk.multi_transform_with_state`, '
                     f'actually called with {type(f)}')

  return f_new


def make_tree(f: Callable[[int], Any], treedef: jax.tree_util.PyTreeDef):
  leaves = list(map(f, range(treedef.num_leaves)))
  return jax.tree.unflatten(treedef, leaves)
