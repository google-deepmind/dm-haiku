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
    transparent
    lift

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

transparent
~~~~~~~~~~~~

.. autofunction:: transparent

lift
~~~~

.. autofunction:: lift

Getters and Interceptors
------------------------

.. autosummary::

    custom_creator
    custom_getter
    GetterContext
    intercept_methods
    MethodContext

custom_creator
~~~~~~~~~~~~~~

.. autofunction:: custom_creator

custom_getter
~~~~~~~~~~~~~

.. autofunction:: custom_getter

GetterContext
~~~~~~~~~~~~~

.. autoclass:: GetterContext

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

Type Hints
----------

.. autosummary::

    LSTMState
    Params
    State
    Transformed
    TransformedWithState
    MultiTransformed
    MultiTransformedWithState

LSTMState
~~~~~~~~~

.. autoclass:: LSTMState

Params
~~~~~~

.. autoclass:: Params

State
~~~~~

.. autoclass:: State

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
    scan
    switch
    while_loop

cond
~~~~

.. autofunction:: cond

fori_loop
~~~~~~~~~

.. autofunction:: fori_loop

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

🚧 Experimental
===============

.. automodule:: haiku.experimental


TensorFlow Profiler
-------------------

.. autosummary::

    named_call
    profiler_name_scopes

named_call
~~~~~~~~~~

.. autofunction:: named_call

profiler_name_scopes
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: profiler_name_scopes

Graphviz Visualisation
----------------------

.. autosummary::

    to_dot
    abstract_to_dot

abstract_to_dot
~~~~~~~~~~~~~~~

.. autofunction:: abstract_to_dot

to_dot
~~~~~~

.. autofunction:: to_dot

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

    name_scope
    name_like
    lift
    lift_with_state
    LiftWithStateUpdater
    check_jax_usage

name_scope
~~~~~~~~~~

.. autofunction:: name_scope

name_like
~~~~~~~~~

.. autofunction:: name_like

lift_with_state
~~~~~~~~~~~~~~~

.. autofunction:: lift_with_state

LiftWithStateUpdater
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LiftWithStateUpdater

check_jax_usage
~~~~~~~~~~~~~~~

.. autofunction:: check_jax_usage

Optimizations
-------------

.. autosummary::

    optimize_rng_use
    layer_stack
    module_auto_repr

optimize_rng_use
~~~~~~~~~~~~~~~~

.. autofunction:: optimize_rng_use

layer_stack
~~~~~~~~~~~~~~~~

.. autofunction:: layer_stack

module_auto_repr
~~~~~~~~~~~~~~~~

.. autofunction:: module_auto_repr

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
