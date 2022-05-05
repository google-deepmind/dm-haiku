# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Benchmark eval_shape for Haiku models."""

import google_benchmark
import haiku as hk
import jax
import jax.numpy as jnp


def init_benchmark(model):
  """Compile/Trace/Run init."""
  input_shape = [100, 100, 100, 100]
  init, _ = hk.transform_with_state(model)

  @google_benchmark.register(name=f'{model.__name__}_init')
  def init_slow_bench(state):
    """Benchmark Jax trace of hk.init_fn of model."""
    x = jnp.ones(input_shape)
    k = jax.random.PRNGKey(42).block_until_ready()
    while state:
      jax.eval_shape(init, k, x)

  @google_benchmark.register(name=f'{model.__name__}_init_fast')
  def init_fast_bench(state):
    """Benchmark runtime of compiled hk.init_fn of model."""
    x = jnp.ones(input_shape)
    k = jax.random.PRNGKey(42).block_until_ready()
    while state:
      hk.experimental.fast_eval_shape(init, k, x)

  return init_slow_bench, init_fast_bench


# Models to be benchmarked
@init_benchmark
def mlp(x):
  return hk.nets.MLP([300, 100, 10])(x)


@init_benchmark
def resnet_50(x):
  return hk.nets.ResNet50(num_classes=10)(x, is_training=True,
                                          test_local_stats=True)

if __name__ == '__main__':
  google_benchmark.main()
