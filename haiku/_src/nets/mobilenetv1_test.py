# Lint as: python3
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
"""Tests for haiku._src.nets.mobilenetv1."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import test_utils
from haiku._src.nets import mobilenetv1
import jax.numpy as jnp


class MobileNetV1Test(parameterized.TestCase):

  @parameterized.parameters(True, False)
  @test_utils.transform_and_run
  def test_simple(self, use_bn):
    image = jnp.ones([2, 224, 224, 3])
    model = mobilenetv1.MobileNetV1(
        (1, 2, 2, 2, 2),
        (16, 32, 64, 128, 256),
        100,
        use_bn=use_bn
        )

    logits = model(image, is_training=True)
    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, (2, 100))

  @test_utils.transform_and_run
  def test_error_incorrect_args_stride_list(self):
    stride_list = (1, 2, 2, 2, 1, 2)
    channel_list = (64, 128, 128, 256, 256, 512, 512, 512, 512,
                    512, 512, 1024, 1024)
    with self.assertRaisesRegex(
        ValueError, "`strides` and `channels` must have the same length."):
      mobilenetv1.MobileNetV1(stride_list,
                              channel_list,
                              1000,
                              True)


if __name__ == "__main__":
  absltest.main()
