# Haiku: 基于 [JAX] 的 [Sonnet]

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[**概览**](#overview)
| [**为何选择 Haiku？**](#why-haiku)
| [**快速上手**](#quickstart)
| [**安装指南**](#installation)
| [**示例程序**](https://github.com/deepmind/dm-haiku/tree/main/examples/)
| [**用户手册**](#user-manual)
| [**官方文档**](https://dm-haiku.readthedocs.io/)
| [**引用 Haiku**](#citing-haiku)

![pytest](https://github.com/deepmind/dm-haiku/workflows/pytest/badge.svg)
![docs](https://readthedocs.org/projects/dm-haiku/badge/?version=latest)
![pypi](https://img.shields.io/pypi/v/dm-haiku)

> [!IMPORTANT]
> 📣 **截至 2023 年 7 月，[Google DeepMind] 建议新项目采用 [Flax] 代替 Haiku。[Flax] 是最初由 [Google Brain] 开发、现由 [Google DeepMind] 共同维护的神经网络库。** 📣
>
> 截至撰写本文时，[Flax] 拥有 Haiku 所具备特性的超集、一支[规模更大](https://github.com/google/flax/graphs/contributors)且[更活跃](https://github.com/google/flax/activity)的开发团队，并在 Alphabet 以外的用户群体中拥有更高的采纳度。[Flax] 拥有[更完善的文档](https://flax.readthedocs.io/)、[丰富的示例](https://github.com/huggingface/transformers/tree/main/examples/flax)以及积极构建端到端范例的[活跃社区](https://huggingface.co/flax-community)。
>
> Haiku 将继续保持“尽力而为（best-effort）”的技术支持，但该项目已进入[维护模式（maintenance mode）](https://en.wikipedia.org/wiki/Maintenance_mode)，这意味着后续开发重心将聚焦于问题修复以及与 JAX 新版本的兼容性维护。
>
> 后续将发布新版本以保持 Haiku 与较新版本的 Python 和 [JAX] 协同工作，但我们不会再添加（亦不再接受）关于新功能的 PR。
>
> [Google DeepMind] 内部仍有大量项目广泛使用 Haiku，目前计划在此模式下无限期为 Haiku 提供维护支持。

## 什么是 Haiku？

> 俳句乃小具<br>
> 用于构建神经网络<br>
> 犹如：基于 [JAX] 的 [Sonnet]

Haiku 是一个由 [Sonnet]（面向 [TensorFlow] 的神经网络库）的部分原作者专为 [JAX] 设计开发的简洁神经网络库。

Haiku 官方文档详见：https://dm-haiku.readthedocs.io/。

**消歧义说明：** 如果您正在寻找名为 Haiku 的操作系统，请参阅 https://haiku-os.org/。

## 概览<a id="overview"></a>

[JAX] 是一个结合了 NumPy 语法风格、自动微分机制以及一流 GPU/TPU 加速支持的数值计算库。

Haiku 是面向 JAX 的简洁神经网络库，允许开发者使用熟悉的**面向对象编程模型（OOP）**组织网络结构，同时完全保留并无缝调用 JAX 的纯函数式变换能力。

Haiku 提供了两项核心工具：模块抽象 `hk.Module` 以及简洁的函数变换器 `hk.transform`。

- `hk.Module` 是 Python 对象，用于持有对其自身参数、子模块以及对用户输入应用前向计算的方法的引用。
- `hk.transform` 将使用这些面向对象、在函数式视角下“不纯”的模块所编写的函数，转换为可无缝与 `jax.jit`、`jax.grad`、`jax.pmap` 等组合使用的纯函数（pure functions）。

## 为何选择 Haiku？<a id="why-haiku"></a>

目前社区中存在多个面向 JAX 的神经网络库。为何您应该选择 Haiku？

### Haiku 已经受 DeepMind 研究人员的大规模实战检验

- DeepMind 凭借 Haiku 与 JAX 相对轻松地复现了大量前沿实验。这些成果涵盖图像与自然语言处理、生成式模型以及强化学习领域的大规模前沿研究。

### Haiku 是一个库（Library），而非重型框架（Framework）

- Haiku 旨在简化特定核心诉求：管理模型参数以及其他模型状态。
- Haiku 能够自然地与其他第三方库协同组合，并与 JAX 生态系统的其余部分完美契合。
- Haiku 在其他设计上尽可能避免对开发者形成干扰——它不定义自定义优化器、检查点存储格式或分布式复制 API。

### Haiku 不重复造轮子

- Haiku 构建在 Sonnet 的编程模型与 API 之上（Sonnet 在 DeepMind 内部获得了近乎通用的采纳）。它保留了 Sonnet 基于 `Module` 的状态管理编程模型，同时完全解锁了 JAX 的函数式变换能力。
- Haiku 的 API 与抽象设计在合理范围内尽可能贴近 Sonnet。许多开发者已发现 Sonnet 是 TensorFlow 中非常高效的编程范式；Haiku 在 JAX 中提供了完全一致的开发体验。

### 向 Haiku 过渡极为顺畅

- 根据设计初衷，从 TensorFlow 和 Sonnet 迁移至 JAX 和 Haiku 相当自然顺畅。
- 除了新引入的特性（例如 `hk.transform`）之外，Haiku 力求匹配 Sonnet 2 的 API。模块名称、方法、参数名、默认值以及初始化策略均保持高度一致。

### Haiku 简化了 JAX 的其他复杂面向

- Haiku 为随机数管理提供了极其直观的模型。在经变换的函数内部，调用 `hk.next_rng_key()` 即可返回唯一的 PRNG 密钥。
- 这些唯一密钥是确定性地从传入顶层变换函数的初始随机密钥派生而来，因此在配合 JAX 程序变换时完全安全无副作用。

## 快速上手<a id="quickstart"></a>

让我们来看一个包含神经网络、损失函数以及训练循环的完整示例。（如需查看更多示例，请访问我们的 [examples 示例目录](https://github.com/deepmind/dm-haiku/tree/main/examples/)。其中 [MNIST 示例](https://github.com/deepmind/dm-haiku/tree/main/examples/mnist.py) 是极佳的起步参考。）

```python
import haiku as hk
import jax.numpy as jnp

def softmax_cross_entropy(logits, labels):
  one_hot = jax.nn.one_hot(labels, logits.shape[-1])
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

def loss_fn(images, labels):
  mlp = hk.Sequential([
      hk.Linear(300), jax.nn.relu,
      hk.Linear(100), jax.nn.relu,
      hk.Linear(10),
  ])
  logits = mlp(images)
  return jnp.mean(softmax_cross_entropy(logits, labels))

loss_fn_t = hk.transform(loss_fn)
loss_fn_t = hk.without_apply_rng(loss_fn_t)

rng = jax.random.PRNGKey(42)
dummy_images, dummy_labels = next(input_dataset)
params = loss_fn_t.init(rng, dummy_images, dummy_labels)

def update_rule(param, update):
  return param - 0.01 * update

for images, labels in input_dataset:
  grads = jax.grad(loss_fn_t.apply)(params, images, labels)
  params = jax.tree.map(update_rule, params, grads)
```

Haiku 的核心在于 `hk.transform`。`transform` 函数允许您编写依赖参数（此处为 `Linear` 层的权重）的神经网络函数，而无需显式编写用于初始化这些参数的冗长样板代码。`transform` 的实现方式是将该函数转换为符合 JAX 纯函数式规范的一对**纯函数**：`init` 和 `apply`。

### `init`

`init` 函数的调用签名为 `params = init(rng, ...)`（其中 `...` 为未变换函数的入参），允许您**收集**网络中所有参数的初始值。Haiku 通过执行您的函数，持续追踪通过 `hk.get_parameter` 请求的所有参数（由 `hk.Linear` 等内部调用），并将它们作为统一结构返回。

返回的 `params` 对象是网络中所有参数构成的嵌套数据结构，专门设计用于方便开发者检查和操作。具体而言，它是由模块名称映射到该模块参数的字典映射，其中每个模块参数又是从参数名称映射到对应参数张量值的字典。例如：

```
{'linear': {'b': ndarray(..., shape=(300,), dtype=float32),
            'w': ndarray(..., shape=(28, 300), dtype=float32)},
 'linear_1': {'b': ndarray(..., shape=(100,), dtype=float32),
              'w': ndarray(..., shape=(1000, 100), dtype=float32)},
 'linear_2': {'b': ndarray(..., shape=(10,), dtype=float32),
              'w': ndarray(..., shape=(100, 10), dtype=float32)}}
```

### `apply`

`apply` 函数的调用签名为 `result = apply(params, rng, ...)`，允许您将参数值**注入**到函数计算中。每当函数内部调用 `hk.get_parameter` 时，返回的数值都将取自您作为 `apply` 入参提供的 `params`：

```python
loss = loss_fn_t.apply(params, rng, images, labels)
```

注意：由于我们损失函数执行的实际计算并不依赖随机数，因此传递随机数生成器并非必需，我们也可以为 `rng` 参数传入 `None`。（请注意，如果您的计算确实需要使用随机数，传入 `None` 将导致抛出错误。）在上面的示例中，我们通过以下代码让 Haiku 自动省去该入参：

```python
loss_fn_t = hk.without_apply_rng(loss_fn_t)
```

由于 `apply` 是纯函数，我们可以直接将其传递给 `jax.grad`（或 JAX 的任何其他变换）：

```python
grads = jax.grad(loss_fn_t.apply)(params, images, labels)
```

### 训练循环 (Training)

本示例中的训练循环非常简单。值得注意的一个细节是使用 `jax.tree.map` 将更新函数应用到 `params` 和 `grads` 中所有相匹配的条目上。得到的结果与之前的 `params` 具有完全相同的结构，可以再次直接传递给 `apply` 使用。

## 安装指南<a id="installation"></a>

Haiku 使用纯 Python 编写，但通过 JAX 依赖 C++ 底层代码。

由于 JAX 的安装方式取决于您的 CUDA 版本，Haiku 在 `requirements.txt` 中并未将 JAX 列为硬性依赖。

首先，请遵循 [JAX 官方安装说明](https://github.com/jax-ml/jax#installation) 安装带有相应硬件加速器支持的 JAX。

然后，使用 pip 安装 Haiku：

```bash
$ pip install git+https://github.com/deepmind/dm-haiku
```

或者通过 PyPI 安装：

```bash
$ pip install -U dm-haiku
```

我们的示例程序依赖额外的第三方库（例如 [bsuite](https://github.com/deepmind/bsuite)）。您可以使用 pip 安装全套扩展依赖：

```bash
$ pip install -r examples/requirements.txt
```

## 用户手册<a id="user-manual"></a>

### 编写自定义模块

在 Haiku 中，所有模块都是 `hk.Module` 的子类。您可以按需实现任何方法（没有特殊限制），但通常模块会实现 `__init__` 和 `__call__`。

让我们一起实现一个自定义线性全连接层：

```python
class MyLinear(hk.Module):

  def __init__(self, output_size, name=None):
    super().__init__(name=name)
    self.output_size = output_size

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
    w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
    b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.zeros)
    return jnp.dot(x, w) + b
```

所有模块都拥有一个名称。若未显式向模块传递 `name` 参数，其名称将从 Python 类名推断得出（例如 `MyLinear` 变为 `my_linear`）。模块可以包含使用 `hk.get_parameter(param_name, ...)` 访问的具名参数。我们采用此 API（而非仅使用普通对象属性），是为了能够通过 `hk.transform` 将您的代码转换为纯函数。

使用模块时，您需要定义计算函数并使用 `hk.transform` 将其转换为一对纯函数。有关 `transform` 返回函数的更多细节请参阅[快速上手](#quickstart)：

```python
def forward_fn(x):
  model = MyLinear(10)
  return model(x)

# 将 `forward_fn` 转换为具有 `init` 和 `apply` 方法的对象。
# 默认情况下，`apply` 需要一个 rng 密钥（可以为 None），用于配合 `hk.next_rng_key`。
forward = hk.transform(forward_fn)

x = jnp.ones([1, 1])

# 当我们运行 `forward.init` 时，Haiku 将执行 `forward_fn(x)` 并收集初始参数值。
# Haiku 要求向 `init` 传递一个 RNG 密钥，因为参数通常是随机初始化的：
key = hk.PRNGSequence(42)
params = forward.init(next(key), x)

# 当我们运行 `forward.apply` 时，Haiku 将执行 `forward_fn(x)` 并从作为首个参数传入的 `params` 中注入参数值。
# 注意：使用 `hk.transform(f)` 变换的模型在调用时必须传入额外的 `rng` 参数：`forward.apply(params, rng, x)`。
# 若不需要此参数，可使用 `hk.without_apply_rng(hk.transform(f))`。
y = forward.apply(params, None, x)
```

### 处理随机性模型 (Working with stochastic models)

某些模型可能需要在前向计算中引入随机采样。例如，在包含重参数化技巧（reparametrization trick）的变分自编码器（VAE）中，需要从标准正态分布中随机采样；在 Dropout 中，我们需要随机掩码来丢弃输入单元。在 JAX 中实现这一机制的主要难点在于 PRNG 密钥的繁琐管理。

在 Haiku 中，我们提供了维护与模块关联的 PRNG 密钥序列的简洁 API：`hk.next_rng_key()`（或用于获取多个密钥的 `next_rng_keys()`）：

```python
class MyDropout(hk.Module):

  def __init__(self, rate=0.5, name=None):
    super().__init__(name=name)
    self.rate = rate

  def __call__(self, x):
    key = hk.next_rng_key()
    p = jax.random.bernoulli(key, 1.0 - self.rate, shape=x.shape)
    return x * p / (1.0 - self.rate)

forward = hk.transform(lambda x: MyDropout()(x))

key1, key2 = jax.random.split(jax.random.PRNGKey(42), 2)
params = forward.init(key1, x)
prediction = forward.apply(params, key2, x)
```

如需更全面地了解随机性模型的使用方法，请参阅我们的 [VAE 示例](https://github.com/deepmind/dm-haiku/tree/main/examples/vae.py)。

**注意**：`hk.next_rng_key()` 在函数式语义上是不纯的，这意味着您应当避免将其与置于 `hk.transform` 内部的 JAX 原生变换一同混用。如需了解更多信息及潜在的规避方案，请查阅关于 [Haiku 变换](https://dm-haiku.readthedocs.io/en/latest/notebooks/transforms.html) 的文档，以及网络内部可用的 [JAX 变换包装器](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku-transforms)。

### 处理非可训练状态 (Working with non-trainable state)

某些模型需要维护内部可变状态。例如，在批量归一化（Batch Normalization）中，需要维护训练期间统计量的滑动平均值。

在 Haiku 中，我们提供了用于维护与模块关联的可变状态的简洁 API：`hk.set_state` 和 `hk.get_state`。使用这些函数时，您需要通过 `hk.transform_with_state` 转换函数，因为返回的一对函数的调用签名有所不同：

```python
def forward(x, is_training):
  net = hk.nets.ResNet50(1000)
  return net(x, is_training)

forward = hk.transform_with_state(forward)

# `init` 函数现在返回参数（params）与状态（state）。
# 状态包含所有通过 `hk.set_state` 创建的内容。
# 其结构与 params 完全相同（即按模块组织的具名数值映射）。
params, state = forward.init(rng, x, is_training=True)

# `apply` 函数现在同时接收 params 与 state。
# 此外，它还会返回更新后的 state 值。
# 在 ResNet 示例中，这将是批量归一化层中使用的滑动平均值的更新值。
logits, state = forward.apply(params, state, rng, x, is_training=True)
```

如果您忘记使用 `hk.transform_with_state` 也不必担心，Haiku 会输出明确的错误提示引导您使用 `hk.transform_with_state`，而不会静默丢弃您的状态。

### 基于 `jax.pmap` 的分布式训练

从 `hk.transform`（或 `hk.transform_with_state`）返回的纯函数完全兼容 `jax.pmap`。有关使用 `jax.pmap` 进行 SPMD（单程序多数据）并行编程的更多细节，请[参阅此处](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)。

在 Haiku 中，`jax.pmap` 的常见用途是在多个加速器（可能跨越多个计算节点主机）上进行数据并行训练。在 Haiku 中，实现方式大致如下：

```python
def loss_fn(inputs, labels):
  logits = hk.nets.MLP([8, 4, 2])(x)
  return jnp.mean(softmax_cross_entropy(logits, labels))

loss_fn_t = hk.transform(loss_fn)
loss_fn_t = hk.without_apply_rng(loss_fn_t)

# 在单个设备上初始化模型。
rng = jax.random.PRNGKey(428)
sample_image, sample_label = next(input_dataset)
params = loss_fn_t.init(rng, sample_image, sample_label)

# 将参数复制到所有设备上。
num_devices = jax.local_device_count()
params = jax.tree.map(lambda x: np.stack([x] * num_devices), params)

def make_superbatch():
  """构造一个超级批次（superbatch），即每个设备分配一个数据批次。"""
  # 获取 N 个批次，然后拆分为图像列表与标签列表。
  superbatch = [next(input_dataset) for _ in range(num_devices)]
  superbatch_images, superbatch_labels = zip(*superbatch)
  # 将超级批次堆叠为一个具有前导维度的数组，而非 Python 列表。
  # 这是 `jax.pmap` 所期望的输入格式。
  superbatch_images = np.stack(superbatch_images)
  superbatch_labels = np.stack(superbatch_labels)
  return superbatch_images, superbatch_labels

def update(params, inputs, labels, axis_name='i'):
  """根据输入和标签上的前向表现更新参数。"""
  grads = jax.grad(loss_fn_t.apply)(params, inputs, labels)
  # 跨所有数据并行副本计算梯度的平均值。
  grads = jax.lax.pmean(grads, axis_name)
  # 使用 SGD 或 Adam 等优化器更新参数...
  new_params = my_update_rule(params, grads)
  return new_params

# 执行若干次训练更新。
for _ in range(10):
  superbatch_images, superbatch_labels = make_superbatch()
  params = jax.pmap(update, axis_name='i')(params, superbatch_images,
                                           superbatch_labels)
```

如需深入了解分布式 Haiku 训练的完整范例，请参阅我们的 [ImageNet 上的 ResNet-50 示例](https://github.com/deepmind/dm-haiku/tree/main/examples/imagenet/)。

## 引用 Haiku<a id="citing-haiku"></a>

如需引用本仓库：

```bibtex
@software{haiku2020github,
  author = {Tom Hennigan and Trevor Cai and Tamara Norman and Lena Martens and Igor Babuschkin},
  title = {{H}aiku: {S}onnet for {JAX}},
  url = {http://github.com/deepmind/dm-haiku},
  version = {0.0.17},
  year = {2020},
}
```

在此 BibTeX 条目中，版本号建议取自 [`haiku/__init__.py`](https://github.com/deepmind/dm-haiku/blob/main/haiku/__init__.py)，年份对应于该项目的开源发布年份。

[JAX]: https://github.com/jax-ml/jax
[Sonnet]: https://github.com/deepmind/sonnet
[Tensorflow]: https://github.com/tensorflow/tensorflow
[Flax]: https://github.com/google/flax
[Google DeepMind]: https://blog.google/technology/ai/april-ai-update/
[Google Brain]: https://research.google/teams/brain/

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月23日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
