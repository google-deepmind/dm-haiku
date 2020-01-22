.. TODO(slebedev): add a title, e.g. "API docs"?

Base
----

.. currentmodule:: haiku

Module
~~~~~~

.. autoclass:: Module
   :members:

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

Convolutional modules
---------------------

.. currentmodule:: haiku

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

Normalization modules
---------------------

.. currentmodule:: haiku

LayerNorm
~~~~~~~~~

.. autoclass:: LayerNorm
   :members:

InstanceNorm
~~~~~~~~~~~~

.. autoclass:: InstanceNorm
   :members:

BatchNorm
~~~~~~~~~

.. autoclass:: BatchNorm
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

VanillaRNN
~~~~~~~~~~

.. autoclass:: VanillaRNN
   :members:
   :special-members:

DeepRNN
~~~~~~~

.. autoclass:: DeepRNN
   :members:

.. autofunction:: deep_rnn_with_skip_connections

LSTM
~~~~

.. autoclass:: LSTM
   :members:

Batch
-----

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

Paddings
--------

.. automodule:: haiku.pad

causal
~~~~~~

.. autofunction:: causal

create
~~~~~~

.. autofunction:: create

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

Nets
----

.. automodule:: haiku.nets

MLP
~~~

.. autoclass:: MLP
   :members:

ResNet50
~~~~~~~~

.. autoclass:: ResNet50
   :members:

References
----------

.. bibliography:: references.bib
   :style: unsrt
