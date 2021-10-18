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
"""Lifting parameters in Haiku."""

import types
from typing import Any, Callable, MutableMapping

from haiku._src import base
from haiku._src import data_structures
from haiku._src import module
from haiku._src import transform

# If you are forking replace this with `import haiku as hk`.
hk = types.ModuleType("haiku")
hk.Params = base.Params
hk.Module = module.Module
hk.running_init = transform.running_init
hk.data_structures = data_structures
del module, data_structures, transform


def pack_into_dict(src: hk.Params, dst: MutableMapping[str, Any], prefix: str):
  """Puts items from src into dst, with an added prefix."""
  for key, value in src.items():
    new_key = f"{prefix}/{key}"
    assert new_key not in dst
    dst[new_key] = value


def unpack_from_dict(src: hk.Params, prefix: str) -> hk.Params:
  """Returns pairs from src where key begins with prefix, cutting off prefix."""
  l = len(prefix)
  out = {k[l:]: v for k, v in src.items() if k.startswith(prefix)}
  return hk.data_structures.to_haiku_dict(out)


# TODO(tycai): Accept state=True.
# TODO(tycai): Make sure transformed functions have better names.
class LiftingModule(hk.Module):
  """See :func:`lift`."""

  def __init__(self, init_fn, name="lifted"):
    super().__init__(name=name)
    self._init_fn = init_fn

  def __call__(self, *args, **kwargs):
    outer_params = base.current_frame().params
    if hk.running_init():
      inner_params = self._init_fn(*args, **kwargs)
      # Lift parameters into this transform's params_dict.
      pack_into_dict(inner_params, outer_params, self.module_name)
      return inner_params
    else:
      return unpack_from_dict(outer_params, f"{self.module_name}/")


def lift(
    init_fn: Callable[..., hk.Params],
    name: str = "lifted",
) -> Callable[..., hk.Params]:
  r"""Lifts the given init fn to a function in the current Haiku namespace.

  During init, the returned callable will run the given ``init_fn``, and include
  the resulting params in the outer transform's dictionaries.
  During ``apply``, the returned callable will instead pull the relevant
  parameters from the outer transform's dictionaries.

  Must be called inside :func:`transform`\ , and be passed the ``init``
  member of a :class:`Transformed`\ .

  The user must ensure that the given ``init`` does not accidentally catch
  modules from an outer :func:`transform` via functional closure.

  Example:

    >>> def g(x):
    ...   return hk.Linear(1, name='g_linear')(x)
    >>> g = hk.transform(g)
    >>> init_rng = hk.next_rng_key() if hk.running_init() else None
    >>> x = jnp.ones([1, 1])
    >>> params = hk.lift(g.init, name='f_lift')(init_rng, x)
    >>> out = g.apply(params, None, x)

  Args:
    init_fn: The ``init`` function from an :class:`Transformed`\ .
    name: A string name to prefix parameters with.

  Returns:
    A callable that during ``init`` injects parameter values into the outer
    context and during ``apply`` reuses parameters from the outer context. In
    both cases returns parameter values to be used with an ``apply`` function.
  """
  base.assert_context("lift")
  lifted = LiftingModule(init_fn, name=name)
  # NOTE: Using lambda to avoid exposing module object.
  return lambda *a, **k: lifted(*a, **k)  # pylint: disable=unnecessary-lambda
