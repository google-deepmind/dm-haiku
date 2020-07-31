.. TODO(slebedev): add a title, e.g. "API docs"?

Base
----

.. currentmodule:: haiku

Transforming Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transform

.. autofunction:: transform_with_state

.. autofunction:: without_apply_rng

.. autofunction:: without_state

Type Hints
~~~~~~~~~~

.. autoclass:: LSTMState

.. autoclass:: Params

.. autoclass:: State

.. autoclass:: Transformed

.. autoclass:: TransformedWithState

Parameters and State
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Module
   :members:

.. autofunction:: to_module

.. autofunction:: get_parameter

.. autofunction:: get_state

.. autofunction:: set_state

.. autofunction:: transparent

Random Number Generators
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PRNGSequence
   :members:

.. autofunction:: next_rng_key

.. autofunction:: next_rng_keys

.. autofunction:: maybe_next_rng_key

.. autofunction:: reserve_rng_keys

.. autofunction:: with_rng

Linear modules
--------------

Linear
~~~~~~

.. autoclass:: Linear
   :members:

Bias
~~~~

.. autoclass:: Bias
   :members:

Pooling modules
---------------

.. currentmodule:: haiku

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

dropout
~~~~~~~

.. autofunction:: dropout

Combinator modules
------------------

Sequential
~~~~~~~~~~

.. autoclass:: Sequential
  :members:

Convolutional modules
---------------------

.. currentmodule:: haiku

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

DepthwiseConv2D
~~~~~~~~~~~~~~~

.. autoclass:: DepthwiseConv2D
   :members:

Normalization modules
---------------------

.. currentmodule:: haiku

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

Recurrent modules
-----------------

.. currentmodule:: haiku

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

Batch modules
-------------

.. currentmodule:: haiku

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

Embedding modules
-----------------

.. currentmodule:: haiku

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

PadFn
~~~~~

.. autoclass:: PadFn
   :members:

create
~~~~~~

.. autofunction:: create

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

.. TODO(tomhennigan): rename to something more appropriate.

Networks
--------

.. automodule:: haiku.nets

MLP
~~~

.. autoclass:: MLP
   :members:

MobileNetV1
~~~~~~~~~~~

.. autoclass:: MobileNetV1
   :members:

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
~~~~~~~~~~~~~~~

.. autoclass:: VectorQuantizer
   :members:

VectorQuantizerEMA
~~~~~~~~~~~~~~~~~~

.. autoclass:: VectorQuantizerEMA
   :members:

JAX Transforms
--------------

.. currentmodule:: haiku

cond
~~~~

.. autofunction:: cond

grad
~~~~

.. autofunction:: grad

jit
~~~

.. autofunction:: jit

remat
~~~~~

.. autofunction:: remat

scan
~~~~

.. autofunction:: scan

value_and_grad
~~~~~~~~~~~~~~

.. autofunction:: value_and_grad

Testing
-------

.. automodule:: haiku.testing

transform_and_run
~~~~~~~~~~~~~~~~~

.. autofunction:: transform_and_run

Data structures
---------------

.. automodule:: haiku.data_structures

filter
~~~~~~

.. autofunction:: filter

merge
~~~~~

.. autofunction:: merge

partition
~~~~~~~~~

.. autofunction:: partition

to_immutable_dict
~~~~~~~~~~~~~~~~~

.. autofunction:: to_immutable_dict

to_mutable_dict
~~~~~~~~~~~~~~~

.. autofunction:: to_mutable_dict

tree_bytes
~~~~~~~~~~

.. autofunction:: tree_bytes

tree_size
~~~~~~~~~

.. autofunction:: tree_size


Experimental
------------

.. automodule:: haiku.experimental

custom_creator
~~~~~~~~~~~~~~

.. autofunction:: custom_creator

custom_getter
~~~~~~~~~~~~~

.. autofunction:: custom_getter

ParamContext
~~~~~~~~~~~~

.. autoclass:: ParamContext

intercept_methods
~~~~~~~~~~~~~~~~~

.. autofunction:: intercept_methods

MethodContext
~~~~~~~~~~~~~

.. autoclass:: MethodContext

named_call
~~~~~~~~~~

.. autofunction:: named_call

optimize_rng_use
~~~~~~~~~~~~~~~~

.. autofunction:: optimize_rng_use

lift
~~~~

.. autofunction:: lift

profiler_name_scopes
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: profiler_name_scopes

to_dot
~~~~~~

.. autofunction:: to_dot

Utilities
---------

.. currentmodule:: haiku

multinomial
~~~~~~~~~~~

.. autofunction:: multinomial

one_hot
~~~~~~~

.. autofunction:: one_hot

References
----------

.. bibliography:: references.bib
   :style: unsrt
