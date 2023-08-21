###################
Haiku API reference
###################

Haiku Fundamentals
==================

.. currentmodule:: haiku

Haiku Transforms
----------------

.. autosummary::

    transform
    transform_with_state
    multi_transform
    multi_transform_with_state
    without_apply_rng
    without_state

transform
~~~~~~~~~

.. autofunction:: transform

transform_with_state
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transform_with_state

multi_transform
~~~~~~~~~~~~~~~

.. autofunction:: multi_transform

multi_transform_with_state
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: multi_transform_with_state

without_apply_rng
~~~~~~~~~~~~~~~~~

.. autofunction:: without_apply_rng

without_state
~~~~~~~~~~~~~

.. autofunction:: without_state

with_empty_state
~~~~~~~~~~~~~~~~

.. autofunction:: with_empty_state

Modules, Parameters and State
-----------------------------

.. autosummary::

    Module
    to_module
    get_parameter
    get_state
    set_state

Module
~~~~~~

.. autoclass:: Module
   :members:

to_module
~~~~~~~~~

.. autofunction:: to_module

get_parameter
~~~~~~~~~~~~~

.. autofunction:: get_parameter

get_state
~~~~~~~~~

.. autofunction:: get_state

set_state
~~~~~~~~~

.. autofunction:: set_state

Getters and Interceptors
------------------------

.. autosummary::

    custom_creator
    custom_getter
    custom_setter
    GetterContext
    SetterContext
    intercept_methods
    MethodContext

custom_creator
~~~~~~~~~~~~~~

.. autofunction:: custom_creator

custom_getter
~~~~~~~~~~~~~

.. autofunction:: custom_getter

custom_setter
~~~~~~~~~~~~~

.. autofunction:: custom_setter

GetterContext
~~~~~~~~~~~~~

.. autoclass:: GetterContext

SetterContext
~~~~~~~~~~~~~

.. autoclass:: SetterContext

intercept_methods
~~~~~~~~~~~~~~~~~

.. autofunction:: intercept_methods

MethodContext
~~~~~~~~~~~~~

.. autoclass:: MethodContext

Random Numbers
--------------

.. autosummary::

    PRNGSequence
    next_rng_key
    next_rng_keys
    maybe_next_rng_key
    reserve_rng_keys
    with_rng
    maybe_get_rng_sequence_state
    replace_rng_sequence_state

PRNGSequence
~~~~~~~~~~~~

.. autoclass:: PRNGSequence
   :members:

next_rng_key
~~~~~~~~~~~~

.. autofunction:: next_rng_key

next_rng_keys
~~~~~~~~~~~~~

.. autofunction:: next_rng_keys

maybe_next_rng_key
~~~~~~~~~~~~~~~~~~

.. autofunction:: maybe_next_rng_key

reserve_rng_keys
~~~~~~~~~~~~~~~~

.. autofunction:: reserve_rng_keys

with_rng
~~~~~~~~

.. autofunction:: with_rng

maybe_get_rng_sequence_state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: maybe_get_rng_sequence_state

replace_rng_sequence_state
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: replace_rng_sequence_state

Type Hints
----------

.. autosummary::

    LSTMState
    Params
    MutableParams
    State
    MutableState
    Transformed
    TransformedWithState
    MultiTransformed
    MultiTransformedWithState
    ModuleProtocol
    SupportsCall

LSTMState
~~~~~~~~~

.. autoclass:: LSTMState

Params
~~~~~~

.. autoclass:: Params

MutableParams
~~~~~~~~~~~~~

.. autoclass:: MutableParams

State
~~~~~

.. autoclass:: State

MutableState
~~~~~~~~~~~~

.. autoclass:: MutableState


Transformed
~~~~~~~~~~~

.. autoclass:: Transformed

TransformedWithState
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TransformedWithState

MultiTransformed
~~~~~~~~~~~~~~~~

.. autoclass:: MultiTransformed

MultiTransformedWithState
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiTransformedWithState

ModuleProtocol
~~~~~~~~~~~~~~

.. autoclass:: ModuleProtocol

SupportsCall
~~~~~~~~~~~~

.. autoclass:: SupportsCall

Flax Interop
============

.. automodule:: haiku.experimental.flax

Haiku inside Flax
-----------------

Module
~~~~~~

.. autoclass:: Module

flatten_flax_to_haiku
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flatten_flax_to_haiku

Flax inside Haiku
-----------------

lift
~~~~

.. autofunction:: lift

Advanced State Management
=========================

.. automodule:: haiku

Lifting
-------

.. autosummary::

    lift
    lift_with_state
    transparent_lift
    transparent_lift_with_state
    LiftWithStateUpdater

lift
~~~~

.. autofunction:: lift

lift_with_state
~~~~~~~~~~~~~~~

.. autofunction:: lift_with_state

transparent_lift
~~~~~~~~~~~~~~~~

.. autofunction:: transparent_lift

transparent_lift_with_state
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transparent_lift_with_state

LiftWithStateUpdater
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LiftWithStateUpdater

Layer Stack
-----------

.. autosummary::

    layer_stack
    LayerStackTransparencyMapping

layer_stack
~~~~~~~~~~~

.. autoclass:: layer_stack

LayerStackTransparencyMapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LayerStackTransparencyMapping

Naming
------

.. autosummary::

    name_scope
    current_name
    DO_NOT_STORE
    get_params
    get_current_state
    get_initial_state
    force_name
    name_like
    transparent

name_scope
~~~~~~~~~~

.. autofunction:: name_scope

current_name
~~~~~~~~~~~~

.. autofunction:: current_name

DO_NOT_STORE
~~~~~~~~~~~~

.. autodata:: DO_NOT_STORE

get_params
~~~~~~~~~~

.. autofunction:: get_params

get_current_state
~~~~~~~~~~~~~~~~~

.. autofunction:: get_current_state

get_initial_state
~~~~~~~~~~~~~~~~~

.. autofunction:: get_initial_state

force_name
~~~~~~~~~~

.. autofunction:: force_name

name_like
~~~~~~~~~

.. autofunction:: name_like

transparent
~~~~~~~~~~~~

.. autofunction:: transparent

Visualisation
-------------

.. autosummary::

    to_dot

to_dot
~~~~~~~~~~~~~~~

.. autofunction:: to_dot

Common Modules
==============

Linear
------

.. autosummary::

    Linear
    Bias

Linear
~~~~~~

.. autoclass:: Linear
   :members:

Bias
~~~~

.. autoclass:: Bias
   :members:

Pooling
-------

.. currentmodule:: haiku

.. autosummary::

    avg_pool
    AvgPool
    max_pool
    MaxPool

Average Pool
~~~~~~~~~~~~

.. autofunction:: avg_pool

.. autoclass:: AvgPool
   :members:

Max Pool
~~~~~~~~

.. autofunction:: max_pool

.. autoclass:: MaxPool
   :members:

Dropout
-------

.. autosummary::

    dropout

dropout
~~~~~~~

.. autofunction:: dropout

Combinator
----------

.. autosummary::

    Sequential

Sequential
~~~~~~~~~~

.. autoclass:: Sequential
  :members:

Convolutional
-------------

.. currentmodule:: haiku

.. autosummary::

    ConvND
    Conv1D
    Conv2D
    Conv3D
    ConvNDTranspose
    Conv1DTranspose
    Conv2DTranspose
    Conv3DTranspose
    DepthwiseConv1D
    DepthwiseConv2D
    DepthwiseConv3D
    get_channel_index

ConvND
~~~~~~

.. autoclass:: ConvND
   :members:

Conv1D
~~~~~~

.. autoclass:: Conv1D
   :members:

Conv2D
~~~~~~

.. autoclass:: Conv2D
   :members:

Conv3D
~~~~~~

.. autoclass:: Conv3D
   :members:

ConvNDTranspose
~~~~~~~~~~~~~~~

.. autoclass:: ConvNDTranspose
   :members:

Conv1DTranspose
~~~~~~~~~~~~~~~

.. autoclass:: Conv1DTranspose
   :members:

Conv2DTranspose
~~~~~~~~~~~~~~~

.. autoclass:: Conv2DTranspose
   :members:

Conv3DTranspose
~~~~~~~~~~~~~~~

.. autoclass:: Conv3DTranspose
   :members:

DepthwiseConv1D
~~~~~~~~~~~~~~~

.. autoclass:: DepthwiseConv1D
   :members:

DepthwiseConv2D
~~~~~~~~~~~~~~~

.. autoclass:: DepthwiseConv2D
   :members:

DepthwiseConv3D
~~~~~~~~~~~~~~~

.. autoclass:: DepthwiseConv3D
   :members:

SeparableDepthwiseConv2D
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SeparableDepthwiseConv2D
   :members:

get_channel_index
~~~~~~~~~~~~~~~~~

.. autofunction:: get_channel_index


Normalization
-------------

.. currentmodule:: haiku

.. autosummary::

    BatchNorm
    GroupNorm
    InstanceNorm
    LayerNorm
    RMSNorm
    SpectralNorm
    ExponentialMovingAverage
    SNParamsTree
    EMAParamsTree

BatchNorm
~~~~~~~~~

.. autoclass:: BatchNorm
   :members:

GroupNorm
~~~~~~~~~

.. autoclass:: GroupNorm
   :members:

InstanceNorm
~~~~~~~~~~~~

.. autoclass:: InstanceNorm
   :members:

LayerNorm
~~~~~~~~~

.. autoclass:: LayerNorm
   :members:

RMSNorm
~~~~~~~~~

.. autoclass:: RMSNorm
   :members:

SpectralNorm
~~~~~~~~~~~~

.. autoclass:: SpectralNorm
   :members:

ExponentialMovingAverage
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ExponentialMovingAverage
   :members:

SNParamsTree
~~~~~~~~~~~~

.. autoclass:: SNParamsTree
   :members:

EMAParamsTree
~~~~~~~~~~~~~

.. autoclass:: EMAParamsTree
   :members:

Recurrent
---------

.. currentmodule:: haiku

.. autosummary::

    RNNCore
    dynamic_unroll
    static_unroll
    expand_apply
    VanillaRNN
    LSTM
    GRU
    DeepRNN
    deep_rnn_with_skip_connections
    ResetCore
    IdentityCore
    Conv1DLSTM
    Conv2DLSTM
    Conv3DLSTM

RNNCore
~~~~~~~

.. autoclass:: RNNCore
   :members:

dynamic_unroll
~~~~~~~~~~~~~~

.. autofunction:: dynamic_unroll

static_unroll
~~~~~~~~~~~~~~

.. autofunction:: static_unroll

expand_apply
~~~~~~~~~~~~

.. autofunction:: expand_apply

VanillaRNN
~~~~~~~~~~

.. autoclass:: VanillaRNN
   :members:
   :special-members:

LSTM
~~~~

.. autoclass:: LSTM
   :members:

GRU
~~~

.. autoclass:: GRU
   :members:

DeepRNN
~~~~~~~

.. autoclass:: DeepRNN
   :members:

.. autofunction:: deep_rnn_with_skip_connections

ResetCore
~~~~~~~~~

.. autoclass:: ResetCore
   :members:

IdentityCore
~~~~~~~~~~~~

.. autoclass:: IdentityCore
   :members:

Conv1DLSTM
~~~~~~~~~~~~~~~

.. autoclass:: Conv1DLSTM
   :members:

Conv2DLSTM
~~~~~~~~~~~~~~~

.. autoclass:: Conv2DLSTM
   :members:

Conv3DLSTM
~~~~~~~~~~~~~~~

.. autoclass:: Conv3DLSTM
   :members:

Attention
-----------------

.. currentmodule:: haiku

MultiHeadAttention
~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiHeadAttention
   :members:

Batch
-----

.. currentmodule:: haiku

.. autosummary::

    Reshape
    Flatten
    BatchApply

Reshape
~~~~~~~

.. autoclass:: Reshape
   :members:

Flatten
~~~~~~~

.. autoclass:: Flatten
   :members:

BatchApply
~~~~~~~~~~

.. autoclass:: BatchApply
   :members:

Embedding
---------

.. currentmodule:: haiku

.. autosummary::

    Embed
    EmbedLookupStyle

Embed
~~~~~

.. autoclass:: Embed
   :members:

EmbedLookupStyle
~~~~~~~~~~~~~~~~

.. autoclass:: EmbedLookupStyle
   :members:

Utilities
---------

.. currentmodule:: haiku

.. autosummary::

    Deferred

Deferred
~~~~~~~~

.. autoclass:: Deferred
   :members:


Initializers
------------

.. automodule:: haiku.initializers

.. autosummary::

    Initializer
    Constant
    Identity
    Orthogonal
    RandomNormal
    RandomUniform
    TruncatedNormal
    VarianceScaling
    UniformScaling

Initializer
~~~~~~~~~~~

.. autoclass:: Initializer
   :members:

Constant
~~~~~~~~

.. autoclass:: Constant
   :members:

Identity
~~~~~~~~

.. autoclass:: Identity
   :members:

Orthogonal
~~~~~~~~~~

.. autoclass:: Orthogonal
   :members:

RandomNormal
~~~~~~~~~~~~

.. autoclass:: RandomNormal
   :members:

RandomUniform
~~~~~~~~~~~~~

.. autoclass:: RandomUniform
   :members:

TruncatedNormal
~~~~~~~~~~~~~~~

.. autoclass:: TruncatedNormal
   :members:

VarianceScaling
~~~~~~~~~~~~~~~

.. autoclass:: VarianceScaling
   :members:

UniformScaling
~~~~~~~~~~~~~~

.. autoclass:: UniformScaling
   :members:

Paddings
--------

.. automodule:: haiku.pad

.. autosummary::

    PadFn
    is_padfn
    create
    create_from_padfn
    create_from_tuple
    causal
    full
    reverse_causal
    same
    valid

PadFn
~~~~~

.. autoclass:: PadFn
   :members:

is_padfn
~~~~~~~~

.. autofunction:: is_padfn

create
~~~~~~

.. autofunction:: create

create_from_padfn
~~~~~~~~~~~~~~~~~

.. autofunction:: create_from_padfn

create_from_tuple
~~~~~~~~~~~~~~~~~

.. autofunction:: create_from_tuple

causal
~~~~~~

.. autofunction:: causal

full
~~~~

.. autofunction:: full

reverse_causal
~~~~~~~~~~~~~~

.. autofunction:: reverse_causal

same
~~~~

.. autofunction:: same

valid
~~~~~

.. autofunction:: valid

Full Networks
=============

.. automodule:: haiku.nets

MLP
---

.. autoclass:: MLP
   :members:

MobileNet
---------

MobileNetV1
~~~~~~~~~~~

.. autoclass:: MobileNetV1
   :members:

ResNet
------

.. autosummary::

    ResNet
    ResNet.BlockGroup
    ResNet.BlockV1
    ResNet.BlockV2
    ResNet18
    ResNet34
    ResNet50
    ResNet101
    ResNet152
    ResNet200

ResNet
~~~~~~

.. autoclass:: ResNet
   :members:

ResNet18
~~~~~~~~

.. autoclass:: ResNet18
   :members:

ResNet34
~~~~~~~~

.. autoclass:: ResNet34
   :members:

ResNet50
~~~~~~~~

.. autoclass:: ResNet50
   :members:

ResNet101
~~~~~~~~~

.. autoclass:: ResNet101
   :members:

ResNet152
~~~~~~~~~

.. autoclass:: ResNet152
   :members:

ResNet200
~~~~~~~~~

.. autoclass:: ResNet200
   :members:

VectorQuantizer
---------------

.. autosummary::

    VectorQuantizer
    VectorQuantizerEMA

VectorQuantizer
~~~~~~~~~~~~~~~

.. autoclass:: VectorQuantizer
   :members:

VectorQuantizerEMA
~~~~~~~~~~~~~~~~~~

.. autoclass:: VectorQuantizerEMA
   :members:

JAX Fundamentals
================

.. currentmodule:: haiku

Control Flow
------------

.. autosummary::

    cond
    fori_loop
    map
    scan
    switch
    while_loop

cond
~~~~

.. autofunction:: cond

fori_loop
~~~~~~~~~

.. autofunction:: fori_loop

map
~~~~~~~~~

.. autofunction:: map

scan
~~~~

.. autofunction:: scan

switch
~~~~~~

.. autofunction:: switch

while_loop
~~~~~~~~~~

.. autofunction:: while_loop

JAX Transforms
--------------

.. autosummary::

    eval_shape
    grad
    remat
    value_and_grad
    vmap

eval_shape
~~~~~~~~~~

.. autofunction:: eval_shape

grad
~~~~

.. autofunction:: grad

remat
~~~~~

.. autofunction:: remat

value_and_grad
~~~~~~~~~~~~~~

.. autofunction:: value_and_grad

vmap
~~~~

.. autofunction:: vmap

Mixed Precision
===============

.. automodule:: haiku.mixed_precision

Automatic Mixed Precision
-------------------------

.. autosummary::

    set_policy
    current_policy
    get_policy
    clear_policy
    push_policy

set_policy
~~~~~~~~~~

.. autofunction:: set_policy

current_policy
~~~~~~~~~~~~~~

.. autofunction:: current_policy

get_policy
~~~~~~~~~~

.. autofunction:: get_policy

clear_policy
~~~~~~~~~~~~

.. autofunction:: clear_policy

push_policy
~~~~~~~~~~~~

.. autofunction:: push_policy

🚧 Experimental
===============

.. automodule:: haiku.experimental


Graphviz Visualisation
----------------------

.. autosummary::

    abstract_to_dot

abstract_to_dot
~~~~~~~~~~~~~~~

.. autofunction:: abstract_to_dot

Summarisation
-------------

.. autosummary::

    tabulate
    eval_summary
    ArraySpec
    MethodInvocation
    ModuleDetails

tabulate
~~~~~~~~

.. autofunction:: tabulate

eval_summary
~~~~~~~~~~~~

.. autofunction:: eval_summary

ArraySpec
~~~~~~~~~

.. autoclass:: ArraySpec
  :members:

MethodInvocation
~~~~~~~~~~~~~~~~

.. autoclass:: MethodInvocation
  :members:

ModuleDetails
~~~~~~~~~~~~~

.. autoclass:: ModuleDetails
  :members:

Managing State
--------------

.. autosummary::

    check_jax_usage

check_jax_usage
~~~~~~~~~~~~~~~

.. autofunction:: check_jax_usage

Optimizations
-------------

.. autosummary::

    optimize_rng_use
    module_auto_repr
    fast_eval_shape
    rng_reserve_size

optimize_rng_use
~~~~~~~~~~~~~~~~

.. autofunction:: optimize_rng_use

module_auto_repr
~~~~~~~~~~~~~~~~

.. autofunction:: module_auto_repr

fast_eval_shape
~~~~~~~~~~~~~~~

.. autofunction:: fast_eval_shape

rng_reserve_size
~~~~~~~~~~~~~~~~

.. autofunction:: rng_reserve_size

jaxpr_info
----------

.. automodule:: haiku.experimental.jaxpr_info

.. autosummary::

    make_model_info
    as_html
    as_html_page
    css
    format_module
    js
    Expression
    Module

make_model_info
~~~~~~~~~~~~~~~

.. autofunction:: make_model_info

as_html
~~~~~~~

.. autofunction:: as_html

as_html_page
~~~~~~~~~~~~

.. autofunction:: as_html_page

css
~~~

.. autofunction:: css

format_module
~~~~~~~~~~~~~

.. autofunction:: format_module

js
~~

.. autofunction:: js

Expression
~~~~~~~~~~

.. autoclass:: Expression

Module
~~~~~~

.. autoclass:: Module

Configuration
=============

.. automodule:: haiku.config

.. autosummary::

    context
    set

context
-------

.. autofunction:: context

set
---

.. autofunction:: set

Utilities
=========

.. currentmodule:: haiku

Data Structures
---------------

.. automodule:: haiku.data_structures

.. autosummary::

    filter
    is_subset
    map
    merge
    partition
    partition_n
    to_haiku_dict
    to_immutable_dict
    to_mutable_dict
    traverse
    tree_bytes
    tree_size

filter
~~~~~~

.. autofunction:: filter

is_subset
~~~~~~~~~

.. autofunction:: is_subset

map
~~~~~~

.. autofunction:: map

merge
~~~~~

.. autofunction:: merge

partition
~~~~~~~~~

.. autofunction:: partition

partition_n
~~~~~~~~~~~

.. autofunction:: partition_n

to_haiku_dict
~~~~~~~~~~~~~

.. autofunction:: to_haiku_dict

to_immutable_dict
~~~~~~~~~~~~~~~~~

.. autofunction:: to_immutable_dict

to_mutable_dict
~~~~~~~~~~~~~~~

.. autofunction:: to_mutable_dict

traverse
~~~~~~~~

.. autofunction:: traverse

tree_bytes
~~~~~~~~~~

.. autofunction:: tree_bytes

tree_size
~~~~~~~~~

.. autofunction:: tree_size

Testing
-------

.. automodule:: haiku.testing

.. autosummary::

    transform_and_run

transform_and_run
~~~~~~~~~~~~~~~~~

.. autofunction:: transform_and_run

Conditional Computation
-----------------------

.. automodule:: haiku

.. autosummary::

    running_init

running_init
~~~~~~~~~~~~

.. autofunction:: running_init

Functions
---------

.. automodule:: haiku

.. autosummary::

    multinomial
    one_hot

multinomial
~~~~~~~~~~~

.. autofunction:: multinomial

one_hot
~~~~~~~

.. autofunction:: one_hot

References
==========

.. bibliography:: references.bib
   :style: unsrt
