# Lint as: python3
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
"""Tests for haiku._src.nets.mlp."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import test_utils
from haiku._src.nets import mlp
import jax
import jax.numpy as jnp


class MLPTest(parameterized.TestCase):

  @test_utils.transform_and_run
  def test_b_init_when_with_bias_false(self):
    with self.assertRaisesRegex(ValueError, "b_init must not be set"):
      mlp.MLP([1], with_bias=False, b_init=lambda *_: _)

  @parameterized.parameters(1, 2, 3)
  @test_utils.transform_and_run
  def test_layers(self, num_layers):
    mod = mlp.MLP([1] * num_layers)
    self.assertLen(mod.layers, num_layers)

  @parameterized.parameters(1, 2, 3)
  @test_utils.transform_and_run
  def test_applies_activation(self, num_layers):
    activation = CountingActivation()
    mod = mlp.MLP([1] * num_layers, activation=activation)
    mod(jnp.ones([1, 1]))
    self.assertEqual(activation.count, num_layers - 1)

  @parameterized.parameters(1, 2, 3)
  @test_utils.transform_and_run
  def test_activate_final(self, num_layers):
    activation = CountingActivation()
    mod = mlp.MLP([1] * num_layers, activate_final=True, activation=activation)
    mod(jnp.ones([1, 1]))
    self.assertEqual(activation.count, num_layers)

  @parameterized.parameters(1, 2, 3)
  @test_utils.transform_and_run
  def test_adds_index_to_layer_names(self, num_layers):
    mod = mlp.MLP([1] * num_layers)
    for index, linear in enumerate(mod.layers):
      self.assertEqual(linear.name, "linear_%d" % index)

  @parameterized.parameters(False, True)
  @test_utils.transform_and_run
  def test_passes_with_bias_to_layers(self, with_bias):
    mod = mlp.MLP([1, 1, 1], with_bias=with_bias)
    for linear in mod.layers:
      self.assertEqual(linear.with_bias, with_bias)

  @test_utils.transform_and_run(run_apply=False)
  def test_repeat_initializer(self):
    w_init = CountingInitializer()
    b_init = CountingInitializer()
    mod = mlp.MLP([1, 1, 1], w_init=w_init, b_init=b_init)
    mod(jnp.ones([1, 1]))
    self.assertEqual(w_init.count, 3)
    self.assertEqual(b_init.count, 3)

  @test_utils.transform_and_run
  def test_default_name(self):
    mod = mlp.MLP([1])
    self.assertEqual(mod.name, "mlp")

  @test_utils.transform_and_run
  def test_custom_name(self):
    mod = mlp.MLP([1], name="foobar")
    self.assertEqual(mod.name, "foobar")

  @test_utils.transform_and_run
  def test_reverse_default_name(self):
    mod = reversed_mlp()
    self.assertEqual(mod.name, "mlp_reversed")

  @test_utils.transform_and_run
  def test_reverse_custom_name(self):
    mod = reversed_mlp(name="foobar")
    self.assertEqual(mod.name, "foobar_reversed")

  @test_utils.transform_and_run
  def test_reverse_override_name(self):
    mod = mlp.MLP([2, 3, 4])
    mod(jnp.ones([1, 1]))
    rev = mod.reverse(name="foobar")
    self.assertEqual(rev.name, "foobar")

  @test_utils.transform_and_run
  def test_reverse(self):
    mod = reversed_mlp()
    self.assertEqual([l.output_size for l in mod.layers], [3, 2, 1])

  @parameterized.parameters(True, False)
  @test_utils.transform_and_run
  def test_reverse_passed_with_bias(self, with_bias):
    mod = reversed_mlp(with_bias=with_bias)
    for linear in mod.layers:
      self.assertEqual(linear.with_bias, with_bias)

  @test_utils.transform_and_run
  def test_reverse_w_init(self):
    w_init = CountingInitializer()
    mod = reversed_mlp(w_init=w_init)
    for linear in mod.layers:
      self.assertIs(linear.w_init, w_init)

  @test_utils.transform_and_run
  def test_reverse_b_init(self):
    b_init = CountingInitializer()
    mod = reversed_mlp(b_init=b_init)
    for linear in mod.layers:
      self.assertIs(linear.b_init, b_init)

  @test_utils.transform_and_run
  def test_reverse_activation(self):
    activation = CountingActivation()
    mod = reversed_mlp(activation=activation)
    activation.count = 0
    mod(jnp.ones([1, 1]))
    self.assertEqual(activation.count, 2)

  @test_utils.transform_and_run
  def test_dropout_requires_key(self):
    mod = mlp.MLP([1, 1])
    with self.assertRaisesRegex(ValueError, "rng key must be passed"):
      mod(jnp.ones([1, 1]), dropout_rate=0.5)

  @test_utils.transform_and_run
  def test_no_dropout_rejects_rng(self):
    mod = mlp.MLP([1, 1])
    with self.assertRaisesRegex(ValueError, "only.*when using dropout"):
      mod(jnp.ones([1, 1]), rng=jax.random.PRNGKey(42))

  @parameterized.parameters(False, True)
  @test_utils.transform_and_run
  def test_reverse_activate_final(self, activate_final):
    activation = CountingActivation()
    mod = reversed_mlp(activation=activation, activate_final=activate_final)
    activation.count = 0
    mod(jnp.ones([1, 1]))
    self.assertEqual(activation.count, 3 if activate_final else 2)

  @parameterized.parameters(False, True)
  @test_utils.transform_and_run
  def test_applies_activation_with_dropout(self, use_dropout):
    if use_dropout:
      dropout_rate = 0.5
      rng = jax.random.PRNGKey(42)
    else:
      dropout_rate = rng = None

    activation = CountingActivation()
    mod = mlp.MLP([1, 1, 1], activation=activation)
    mod(jnp.ones([1, 1]), dropout_rate, rng)
    self.assertEqual(activation.count, 2)

  @test_utils.transform_and_run
  def test_repr(self):
    mod = mlp.MLP([1, 2, 3])
    for index, linear in enumerate(mod.layers):
      self.assertEqual(
          repr(linear),
          "Linear(output_size={}, name='linear_{}')".format(index + 1, index))


def reversed_mlp(**kwargs):
  mod = mlp.MLP([2, 3, 4], **kwargs)
  mod(jnp.ones([1, 1]))
  return mod.reverse()


class CountingActivation(object):

  def __init__(self):
    self.count = 0

  def __call__(self, x):
    self.count += 1
    return x


class CountingInitializer(object):

  def __init__(self):
    self.count = 0

  def __call__(self, shape, dtype=jnp.float32):
    self.count += 1
    return jnp.ones(shape, dtype=dtype)


if __name__ == "__main__":
  absltest.main()
