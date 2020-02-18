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
"""Tests for haiku._src.pad."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import pad


class PadTest(parameterized.TestCase):

  def test_padding_2d(self):
    a = pad.create((pad.causal, pad.full), (3), (1, 1), 2)
    self.assertEqual(a, ((2, 0), (2, 2)))

  def test_padding_1d(self):
    a = pad.create(pad.full, 3, 1, 1)
    self.assertEqual(a, ((2, 2),))

  def test_padding_3d(self):
    a = pad.create((pad.causal, pad.full, pad.full), (3, 2, 3), (1), 3)
    self.assertEqual(a, ((2, 0), (1, 1), (2, 2)))

  @parameterized.parameters((2, (2, 2)), (3, (4, 4, 4, 4)), ((2, 2), 3),
                            ((4, 4, 4, 4), 3))
  def test_padding_incorrect_input(self, kernel_size, rate):
    with self.assertRaisesRegex(
        TypeError,
        r"must be a scalar or sequence of length 1 or sequence of length 3."):
      pad.create(pad.full, kernel_size, rate, 3)

  def test_padding_valid(self):
    a = pad.create(pad.valid, 4, 3, 2)
    self.assertEqual(a, ((0, 0), (0, 0)))

  def test_padding_same(self):
    a = pad.create(pad.same, 4, 3, 2)
    self.assertEqual(a, ((4, 5), (4, 5)))

  def test_padding_full(self):
    a = pad.create(pad.full, 4, 3, 2)
    self.assertEqual(a, ((9, 9), (9, 9)))

  def test_padding_causal(self):
    a = pad.create(pad.causal, 4, 3, 2)
    self.assertEqual(a, ((9, 0), (9, 0)))

  def test_padding_reverse_causal(self):
    a = pad.create(pad.reverse_causal, 4, 3, 2)
    self.assertEqual(a, ((0, 9), (0, 9)))


if __name__ == "__main__":
  absltest.main()
