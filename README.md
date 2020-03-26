# Haiku: [Sonnet](https://github.com/deepmind/sonnet) for [JAX](https://github.com/google/jax)

[**Overview**](#overview)
| [**Why Haiku?**](#why-haiku)
| [**Quickstart**](#quickstart)
| [**Installation**](#installation)
| [**Examples**](https://github.com/deepmind/dm-haiku/tree/master/examples/)
| [**User manual**](#user-manual)
| [**Documentation**](https://dm-haiku.readthedocs.io/)
| [**Citing Haiku**](#citing-haiku)

![pytest](https://github.com/deepmind/dm-haiku/workflows/pytest/badge.svg)

## What is Haiku?

Haiku is a simple neural network library for
[JAX](https://github.com/google/jax) developed by some of the
authors of [Sonnet](https://github.com/deepmind/sonnet), a neural network
library for [TensorFlow](https://github.com/tensorflow/tensorflow).

**Disambiguation:** if you are looking for Haiku the operating system then
please see https://haiku-os.org/.

NOTE: Haiku is currently **beta**. A number of researchers have tested Haiku
for several months and have reproduced a number of experiments at scale. Please
feel free to use Haiku, and
[let us know](https://github.com/deepmind/dm-haiku/issues) if you have issues!

## Overview

[JAX](https://github.com/google/jax) is a numerical computing library that
combines NumPy, automatic differentiation, and first-class GPU/TPU support.

Haiku is a simple neural network library for JAX that enables users to use
familiar **object-oriented programming models** while allowing full access to
JAX's pure function transformations.

Haiku provides two core tools: a module abstraction, `hk.Module`, and a simple
function transformation, `hk.transform`.

`hk.Module`s are Python objects that hold references to their own parameters,
other modules, and methods that apply functions on user inputs.

`hk.transform` turns functions that use these object-oriented, functionally
"impure" modules into pure functions that can be used with `jax.jit`,
`jax.grad`, `jax.pmap`, etc.

## Why Haiku?

There are a number of neural network libraries for JAX. Why should you choose
Haiku?

### Haiku has been tested by researchers at DeepMind at scale.

- DeepMind has reproduced a number of experiments in Haiku and JAX with relative
  ease. These include large-scale results in image and language processing,
  generative models, and reinforcement learning.

### Haiku is a library, not a framework.

- Haiku is designed to make specific things simpler: managing model parameters
  and other model state.
- Haiku can be expected to compose with other libraries and work well with the
  rest of JAX.
- Haiku otherwise is designed to get out of your way - it does not define custom
  optimizers, checkpointing formats, or replication APIs.

### Haiku does not reinvent the wheel.

- Haiku builds on the programming model and APIs of Sonnet, a neural network
  library with near universal adoption at DeepMind. It preserves Sonnet's
  `Module`-based programming model for state management while retaining access
  to JAX's function transformations.
- Haiku APIs and abstractions are as close as reasonable to Sonnet. Many users
  have found Sonnet to be a productive programming model in TensorFlow; Haiku
  enables the same experience in JAX.

### Transitioning to Haiku is easy.

- By design, transitioning from TensorFlow and Sonnet to JAX and Haiku is easy.
- Outside of new features (e.g. `hk.transform`), Haiku aims to match the API of
  Sonnet 2. Modules, methods, argument names, defaults, and initialization
  schemes should match.

### Haiku makes other aspects of JAX simpler.

- Haiku offers a trivial model for working with random numbers. Within a
  transformed function, `hk.next_rng_key()` returns a unique rng key.
- These unique keys are deterministically derived from an initial random key
  passed into the top-level transformed function, and are thus safe to use with
  JAX program transformations.

## Quickstart

Let's take a look at an example neural network and loss function.

```python
import haiku as hk
import jax.numpy as jnp

def softmax_cross_entropy(logits, labels):
  one_hot = hk.one_hot(labels, logits.shape[-1])
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

def loss_fn(images, labels):
  mlp = hk.Sequential([
      hk.Linear(300), jax.nn.relu,
      hk.Linear(100), jax.nn.relu,
      hk.Linear(10),
  ])
  logits = mlp(images)
  return jnp.mean(softmax_cross_entropy(logits, labels))

loss_obj = hk.transform(loss_fn)
```

`hk.transform` allows us to turn this function into a pair of pure functions:
`init` and `apply`. All JAX transformations (e.g. `jax.grad`) require you to pass
in a pure function for correct behaviour. Haiku makes it easy to write them.

The `init` function returned by `hk.transform` allows you to **collect** the
initial value of any parameters in the network. Haiku does this by running your
function, keeping track of any parameters requested through `hk.get_parameter`
and returning them to you:

```python
# Initial parameter values are typically random. In JAX you need a key in order
# to generate random numbers and so Haiku requires you to pass one in.
rng = jax.random.PRNGKey(42)

# `init` runs your function, as such we need an example input. Typically you can
# pass "dummy" inputs (e.g. ones of the same shape and dtype) since initialization
# is not usually data dependent.
images, labels = next(input_dataset)

# The result of `init` is a nested data structure of all the parameters in your
# network. You can pass this into `apply`.
params = loss_obj.init(rng, images, labels)
```

The `params` object is designed for you to inspect and manipulate. It is a
mapping of module name to module parameters, where a module parameter is a mapping
of parameter name to parameter value. For example:

```
{'linear': {'b': ndarray(..., shape=(1000,), dtype=float32),
            'w': ndarray(..., shape=(28, 1000), dtype=float32)},
 'linear_1': {'b': ndarray(..., shape=(100,), dtype=float32),
              'w': ndarray(..., shape=(1000, 100), dtype=float32)},
 'linear_2': {'b': ndarray(..., shape=(10,), dtype=float32),
              'w': ndarray(..., shape=(100, 10), dtype=float32)}}
```

The `apply` function allows you to **inject** parameter values into your
function. Whenever `hk.get_parameter` is called the value returned will come
from the `params` you provide as input to `apply`:

```python
loss = loss_obj.apply(params, images, labels)
```

Since `apply` is a pure function we can pass it to `jax.grad` (or any of JAX's
other transforms):

```python
grads = jax.grad(loss_obj.apply)(params, images, labels)
```

Finally, we put this all together into a simple training loop:

```python
def sgd(param, update):
  return param - 0.01 * update

for images, labels in input_dataset:
  grads = jax.grad(loss_obj.apply)(params, images, labels)
  params = jax.tree_multimap(sgd, params, grads)
```

Here we used `jax.tree_multimap` to apply the `sgd` function across all matching
entries in `params` and `grads`. The result has the same structure as the previous
`params` and can again be used with `apply`.

For more, see our
[examples directory](https://github.com/deepmind/dm-haiku/tree/master/examples/).
The
[MNIST example](https://github.com/deepmind/dm-haiku/tree/master/examples/mnist.py)
is a good place to start.

## Installation

Haiku is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, Haiku does
not list JAX as a dependency in `requirements.txt`.

First, follow [these instructions](https://github.com/google/jax#installation)
to install JAX with the relevant accelerator support.

Then, install Haiku using pip:

```bash
$ pip install git+https://github.com/deepmind/dm-haiku
```

Our examples rely on additional libraries (e.g. [bsuite](https://github.com/deepmind/bsuite)). You can install the full set of additional requirements using pip:

```bash
$ pip install -r examples/requirements.txt
```

## User manual

### Writing your own modules

In Haiku, all modules are a subclass of `hk.Module`. You can implement any
method you like (nothing is special-cased), but typically modules implement
`__init__` and `__call__`.

Let's work through implementing a linear layer:

```python
class MyLinear(hk.Module):

  def __init__(self, output_size, name=None):
    super(MyLinear, self).__init__(name=name)
    self.output_size = output_size

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
    w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
    b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.zeros)
    return jnp.dot(x, w) + b
```

All modules have a name. When no `name` argument is passed to the module, its
name is inferred from the name of the Python class (for example `MyLinear`
becomes `my_linear`). Modules can have named parameters that are accessed
using `hk.get_parameter(param_name, ...)`. We use this API (rather than just
using object properties) so that we can convert your code into a pure function
using `hk.transform`.

When using modules you need to define functions and transform them into a pair
of pure functions using `hk.transform`. See our [quickstart](#quickstart) for
more details about the functions returned from `transform`:

```python
def forward_fn(x):
  model = MyLinear(10)
  return model(x)

# Turn `forward_fn` into an object with `init` and `apply` methods.
forward = hk.transform(forward_fn)

x = jnp.ones([1, 1])

# When we run `forward.init`, Haiku will run `forward(x)` and collect initial
# parameter values. Haiku requires you pass a RNG key to `init`, since parameters
# are typically initialized randomly:
key = hk.PRNGSequence(42)
params = forward.init(next(key), x)

# When we run `forward.apply`, Haiku will run `forward(x)` and inject parameter
# values from the `params` that are passed as the first argument. We do not require
# an RNG key by default since models are deterministic. You can (of course!) change
# this using `hk.transform(f, apply_rng=True)` if you prefer:
y = forward.apply(params, x)
```

### Working with stochastic models

Some models may require random sampling as part of the computation.
For example, in variational autoencoders with the reparametrization trick,
a random sample from the standard normal distribution is needed.
The main hurdle in making this work with JAX is in management of PRNG keys.

In Haiku we provide a simple API for maintaining a PRNG key sequence associated
with modules: `hk.next_rng_key()` (or `next_rng_keys()` for multiple keys).
In order to use this functionality you need to specify `apply_rng=True`
argument on the `hk.transform` call:

```python
class Dropout(hk.Module):
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    rng_key = hk.next_rng_key()
    p = jax.random.bernoulli(rng_key, 1.0 - self.rate, shape=x.shape)
    return x * p / (1.0 - self.rate)

forward = hk.transform(lambda x: VAE()(x), apply_rng=True)

rng_key1, rng_key2 = jax.random.split(jax.random.PRNGKey(42), 2)

params = forward.init(rng_key1, x)
prediction = forward.apply(params, rng_key2, x)
```

For a more complete look at working with stochastic models, please see our
[VAE example](https://github.com/deepmind/dm-haiku/tree/master/examples/vae.py).

**Note:** `hk.next_rng_key()` is not functionally pure which means you should
avoid using it alongside JAX transformations which are inside `hk.transform`.
For more information and possible workarounds, please consult the docs on
[Haiku transforms](https://dm-haiku.readthedocs.io/en/latest/transforms.html).

### Working with non-trainable state

Some models may want to maintain some internal, mutable state. For example, in
batch normalization a moving average of values encountered during training is
maintained.

In Haiku we provide a simple API for maintaining mutable state that is
associated with modules: `hk.set_state` and `hk.get_state`. When using these
functions you need to transform your function using `hk.transform_with_state`
since the signature of the returned pair of functions is different:

```python
def forward(x, is_training):
  net = hk.nets.ResNet50(1000)
  return net(x, is_training)

forward = hk.transform_with_state(forward)

# The `init` function now returns parameters **and** state. State contains
# anything that was created using `hk.set_state`. The structure is the same as
# params (e.g. it is a per-module mapping of named values).
params, state = forward.init(rng, x, is_training=True)

# The apply function now takes both params **and** state. Additionally it will
# return updated values for state. In the resnet example this will be the
# updated values for moving averages used in the batch norm layers.
logits, state = forward.apply(params, state, rng, x, is_training=True)
```

If you forget to use `hk.transform_with_state` don't worry, we will print a
clear error pointing you to `hk.transform_with_state` rather than silently
dropping your state.

### Distributed training with `jax.pmap`

The pure functions returned from `hk.transform` (or `hk.transform_with_state`)
are fully compatible with `jax.pmap`. For more details on SPMD programming with
`jax.pmap`,
[look here](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap).

One common use of `jax.pmap` with Haiku is for data-parallel training on many
accelerators, potentially across multiple hosts. With Haiku, that might look
like this:

```python
def loss_fn(inputs, labels):
  logits = hk.nets.MLP([8, 4, 2])(x)
  return jnp.mean(softmax_cross_entropy(logits, labels))

loss_obj = hk.transform(loss_fn)

# Initialize the model on a single device.
rng = jax.random.PRNGKey(428)
sample_image, sample_label = next(input_dataset)
params = loss_obj.init(rng, sample_image, sample_label)

# Replicate params onto all devices.
num_devices = jax.local_device_count()
params = jax.tree_util.tree_map(lambda x: np.stack([x] * num_devices), params)

def make_superbatch():
  """Constructs a superbatch, i.e. one batch of data per device."""
  # Get N batches, then split into list-of-images and list-of-labels.
  superbatch = [next(input_dataset) for _ in range(num_devices)]
  superbatch_images, superbatch_labels = zip(*superbatch)
  # Stack the superbatches to be one array with a leading dimension, rather than
  # a python list. This is what `jax.pmap` expects as input.
  superbatch_images = np.stack(superbatch_images)
  superbatch_labels = np.stack(superbatch_labels)
  return superbatch_images, superbatch_labels

def update(params, inputs, labels, axis_name='i'):
  """Updates params based on performance on inputs and labels."""
  grads = jax.grad(loss_obj.apply)(params, inputs, labels)
  # Take the mean of the gradients across all data-parallel replicas.
  grads = jax.lax.pmean(grads, axis_name)
  # Update parameters using SGD or Adam or ...
  new_params = my_update_rule(params, grads)
  return new_params

# Run several training updates.
for _ in range(10):
  superbatch_images, superbatch_labels = make_superbatch()
  params = jax.pmap(update, axis_name='i')(params, superbatch_images,
                                           superbatch_labels)
```

For a more complete look at distributed Haiku training, take a look at our
[ResNet-50 on ImageNet example](https://github.com/deepmind/dm-haiku/tree/master/examples/imagenet/).

## Citing Haiku

To cite this repository:

```
@software{haiku2020github,
  author = {Tom Hennigan and Trevor Cai and Tamara Norman and Igor Babuschkin},
  title = {{H}aiku: {S}onnet for {JAX}},
  url = {http://github.com/deepmind/dm-haiku},
  version = {0.0.1b0},
  year = {2020},
}
```

In this bibtex entry, the version number is intended to be from
[`haiku/__init__.py`](https://github.com/deepmind/dm-haiku/blob/master/haiku/__init__.py),
and the year corresponds to the project's open-source release.
