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
"""Benchmark trace/compile/run timings of init functions."""

import google_benchmark
import haiku as hk
import jax
import jax.numpy as jnp


def init_benchmark(model):
  """Compile/Trace/Run init."""
  input_shape = [100, 100, 100, 100]
  init, _ = hk.transform_with_state(model)

  @google_benchmark.register(name=f'trace_{model.__name__}')
  def trace_bench(state):
    """Benchmark Jax trace of hk.init_fn of model."""
    x = jnp.ones(input_shape).block_until_ready()
    k = jax.random.PRNGKey(42)

    while state:
      jax.xla_computation(init)(k, x)

  @google_benchmark.register(name=f'compile_{model.__name__}')
  def compile_bench(state):
    """Benchmark Jax compile of hk.init_fn of model."""
    x = jnp.ones(input_shape).block_until_ready()
    k = jax.random.PRNGKey(42)

    c = jax.xla_computation(init)(k, x)
    b = jax.lib.xla_client.get_local_backend()

    while state:
      b.compile(c)

  @google_benchmark.register(name=f'run_{model.__name__}')
  def run_bench(state):
    """Benchmark runtime of compiled hk.init_fn of model."""
    x = jnp.ones(input_shape).block_until_ready()
    k = jax.random.PRNGKey(42)

    jitted_init = jax.jit(init)
    # run jit once to compile
    jitted_init(k, x)

    while state:
      params, _ = jitted_init(k, x)
      # block on computation to finish
      jax.tree_map(lambda x: x.block_until_ready(), params)

  return trace_bench, compile_bench, run_bench


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
