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
"""Tests for haiku._src.nets.resnet."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
from haiku._src import test_utils
from haiku._src import transform
from haiku._src.nets import resnet
import jax
import jax.numpy as jnp
import numpy as np


_RESNETS = ["ResNet{}".format(i) for i in (18, 34, 50, 101, 152, 200)]
_RESNET_NUM_PARAMS = [int(i * 1e6)
                      for i in (11.7, 21.8, 25.6, 44.5, 60.2, 64.7)]
_RESNET_HAS_PROJECTION = [False, False, True, True, True, True]


class ResnetTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(test_utils.named_bools("resnet_v2"),
                                        test_utils.named_bools("bottleneck"))
  @test_utils.transform_and_run
  def test_simple(self, resnet_v2, bottleneck):
    image = jnp.ones([2, 64, 64, 3])
    model = resnet.ResNet([1, 1, 1, 1], 10,
                          resnet_v2=resnet_v2,
                          bottleneck=bottleneck)

    for is_training in (True, False):
      logits = model(image, is_training=is_training)
      self.assertEqual(logits.shape, (2, 10))

  @test_utils.combined_named_parameters(test_utils.named_bools("resnet_v2"),
                                        test_utils.named_bools("bottleneck"))
  def test_local_stats(self, resnet_v2, bottleneck):
    def forward_fn(image):
      model = resnet.ResNet([1, 1, 1, 1], 10,
                            resnet_v2=resnet_v2,
                            bottleneck=bottleneck)
      return model(image, is_training=False, test_local_stats=True)

    forward = transform.transform(forward_fn)
    rng = jax.random.PRNGKey(42)
    image = jnp.ones([2, 64, 64, 3])
    params = forward.init(rng, image)
    logits = forward.apply(params, None, image)
    self.assertEqual(logits.shape, (2, 10))

  @parameterized.parameters(3, 5)
  @test_utils.transform_and_run
  def test_error_incorrect_args_block_list(self, list_length):
    block_list = [i for i in range(list_length)]
    with self.assertRaisesRegex(
        ValueError, "blocks_per_group` must be of length 4 not {}".format(
            list_length)):
      resnet.ResNet(block_list, 10, {"decay_rate": 0.9, "eps": 1e-5})

  @parameterized.parameters(3, 5)
  @test_utils.transform_and_run
  def test_error_incorrect_args_channel_list(self, list_length):
    channel_list = [i for i in range(list_length)]
    with self.assertRaisesRegex(
        ValueError,
        "channels_per_group` must be of length 4 not {}".format(
            list_length)):
      resnet.ResNet([1, 1, 1, 1], 10, {"decay_rate": 0.9, "eps": 1e-5},
                    channels_per_group=channel_list)

  @test_utils.combined_named_parameters(
      [(i, (getattr(resnet, i), n))
       for i, n in zip(_RESNETS, _RESNET_NUM_PARAMS)],
      test_utils.named_bools("resnet_v2"),
  )
  def test_num_params(self, resnet_class_and_num_params, resnet_v2):
    resnet_class, expected_num_params = resnet_class_and_num_params
    def model_func(img):
      model = resnet_class(1000, resnet_v2=resnet_v2)
      return model(img, is_training=True)

    model = hk.transform_with_state(model_func)
    image = jnp.ones([2, 64, 64, 3])
    rng = jax.random.PRNGKey(0)
    params, _ = model.init(rng, image)
    num_params = sum(np.prod(p.shape).item() for p in jax.tree_leaves(params))
    self.assertGreater(num_params, int(0.998 * expected_num_params))
    self.assertLess(num_params, int(1.002 * expected_num_params))

  @test_utils.combined_named_parameters(
      [(i, (getattr(resnet, i), p))
       for i, p in zip(_RESNETS, _RESNET_HAS_PROJECTION)],
      test_utils.named_bools("resnet_v2"),
  )
  @test_utils.transform_and_run
  def test_has_projection(self, resnet_class_and_has_projection, resnet_v2):
    resnet_class, has_projection = resnet_class_and_has_projection
    model = resnet_class(1000, resnet_v2=resnet_v2)
    for i, block_group in enumerate(model.block_groups):
      if i == 0:
        self.assertEqual(hasattr(block_group.blocks[0], "proj_conv"),
                         has_projection)
      else:
        self.assertTrue(hasattr(block_group.blocks[0], "proj_conv"))

      for block in block_group.blocks[1:]:
        self.assertFalse(hasattr(block, "proj_conv"))

  @test_utils.combined_named_parameters(
      [(i, getattr(resnet, i)) for i in _RESNETS],
      test_utils.named_bools("resnet_v2"),
  )
  def test_logits_config(self, resnet_class, resnet_v2):
    def model_func_logits_config_default(img):
      model = resnet_class(1000, resnet_v2=resnet_v2)
      return model(img, is_training=True)

    def model_func_logits_config_modified(img):
      model = resnet_class(1000, resnet_v2=resnet_v2,
                           logits_config=dict(w_init=jnp.ones))
      return model(img, is_training=True)

    image = jnp.ones([2, 64, 64, 3])
    rng = jax.random.PRNGKey(0)

    model = hk.transform_with_state(model_func_logits_config_default)
    params, _ = model.init(rng, image)
    logits_keys = [k for k in params.keys() if "/logits" in k]
    self.assertLen(logits_keys, 1)

    # Check logits params are zeros
    w_logits = params[logits_keys[0]]["w"]
    np.testing.assert_allclose(jnp.zeros_like(w_logits), w_logits)

    model = hk.transform_with_state(model_func_logits_config_modified)
    params, _ = model.init(rng, image)

    # Check logits params are ones
    w_logits = params[logits_keys[0]]["w"]
    np.testing.assert_allclose(jnp.ones_like(w_logits), w_logits)

  @test_utils.combined_named_parameters(
      [(i, getattr(resnet, i)) for i in _RESNETS],
  )
  @test_utils.transform_and_run
  def test_initial_conv_config(self, resnet_cls):
    config = dict(name="custom_name", output_channels=32, kernel_shape=(3, 3),
                  stride=(1, 1), padding="VALID", with_bias=True)
    net = resnet_cls(1000, initial_conv_config=config)
    for key, value in config.items():
      self.assertEqual(getattr(net.initial_conv, key), value)

if __name__ == "__main__":
  absltest.main()
