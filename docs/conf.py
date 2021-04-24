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
"""Configuration file for the Sphinx documentation builder."""

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
import doctest
import inspect
import os
import sys
import typing


def _add_annotations_import(path):
  """Appends a future annotations import to the file at the given path."""
  with open(path) as f:
    contents = f.read()
  if contents.startswith('from __future__ import annotations'):
    # If we run sphinx multiple times then we will append the future import
    # multiple times too.
    return

  assert contents.startswith('#'), (path, contents.split('\n')[0])
  with open(path, 'w') as f:
    # NOTE: This is subtle and not unit tested, we're prefixing the first line
    # in each Python file with this future import. It is important to prefix
    # not insert a newline such that source code locations are accurate (we link
    # to GitHub). The assertion above ensures that the first line in the file is
    # a comment so it is safe to prefix it.
    f.write('from __future__ import annotations  ')
    f.write(contents)


def _recursive_add_annotations_import():
  for path, _, files in os.walk('../haiku/'):
    for file in files:
      if file.endswith('.py'):
        _add_annotations_import(os.path.abspath(os.path.join(path, file)))

if 'READTHEDOCS' in os.environ:
  _recursive_add_annotations_import()

typing.get_type_hints = lambda obj, *unused: obj.__annotations__
sys.path.insert(0, os.path.abspath('../'))
sys.path.append(os.path.abspath('ext'))

import haiku as hk
import sphinxcontrib.katex as katex

# -- Project information -----------------------------------------------------

project = 'Haiku'
copyright = '2019, DeepMind'  # pylint: disable=redefined-builtin
author = 'Haiku Contributors'

# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.katex',
    'sphinx_autodoc_typehints',
    'coverage_check',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': True,
    'exclude-members': '__repr__, __str__, __weakref__',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
# html_favicon = '_static/favicon.ico'

# -- Options for doctest -----------------------------------------------------

doctest_test_doctest_blocks = 'true'
doctest_global_setup = """
import collections
import itertools
import os
import unittest

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jmp

# Use dict instead of FlatMapping (soon to be the default).
os.environ["HAIKU_FLATMAPPING"] = "0"

# Equivalent to wrapping the whole files doctests in `hk.transform_with_state`.
from haiku._src import base
base.new_context(rng=jax.random.PRNGKey(42)).__enter__()
"""
doctest_default_flags = (
    doctest.ELLIPSIS
    | doctest.IGNORE_EXCEPTION_DETAIL
    | doctest.DONT_ACCEPT_TRUE_FOR_1
    | doctest.NORMALIZE_WHITESPACE)

# -- Options for katex ------------------------------------------------------

# See: https://sphinxcontrib-katex.readthedocs.io/en/0.4.1/macros.html
latex_macros = r"""
    \def \d              #1{\operatorname{#1}}
"""

# Translate LaTeX macros to KaTeX and add to options for HTML builder
katex_macros = katex.latex_defs_to_katex_macros(latex_macros)
katex_options = 'macros: {' + katex_macros + '}'

# Add LaTeX macros for LATEX builder
latex_elements = {'preamble': latex_macros}

# -- Source code links -------------------------------------------------------


def linkcode_resolve(domain, info):
  """Resolve a GitHub URL corresponding to Python object."""
  if domain != 'py':
    return None

  try:
    mod = sys.modules[info['module']]
  except ImportError:
    return None

  obj = mod
  try:
    for attr in info['fullname'].split('.'):
      obj = getattr(obj, attr)
  except AttributeError:
    return None
  else:
    obj = inspect.unwrap(obj)

  try:
    filename = inspect.getsourcefile(obj)
  except TypeError:
    return None

  try:
    source, lineno = inspect.getsourcelines(obj)
  except OSError:
    return None

  # TODO(slebedev): support tags after we release an initial version.
  return 'https://github.com/deepmind/dm-haiku/blob/master/haiku/%s#L%d#L%d' % (
      os.path.relpath(filename, start=os.path.dirname(
          hk.__file__)), lineno, lineno + len(source) - 1)


# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# -- nbsphinx configuration --------------------------------------------------

# TODO(tomhennigan): Consider auto/always here.
nbsphinx_execute = 'never'
nbsphinx_codecell_lexer = 'ipython'
nbsphinx_kernel_name = 'python'
nbsphinx_timeout = 180
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::

        Interactive online version:
        :raw-html:`<a href="https://colab.research.google.com/github/deepmind/dm-haiku/blob/master/{{ docname }}"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>`

    __ https://github.com/deepmind/dm-haiku/blob/
        {{ env.config.release }}/{{ docname }}
"""
