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
"""Tests for haiku._src.utils."""

from absl.testing import absltest
from absl.testing import parameterized
from haiku._src import utils

lines = lambda *a: "\n".join(a)


class UtilsTest(absltest.TestCase):

  def test_indent(self):
    self.assertEqual(
        lines("  foo",
              "  bar"),
        utils.indent(2, lines("foo", "bar")))

  def test_auto_repr(self):
    self.assertEqual("SomeClass(a=1, b=2)",
                     utils.auto_repr(SomeClass, 1, 2))
    self.assertEqual("SomeClass(a=1, b=2, c=3)",
                     utils.auto_repr(SomeClass, 1, 2, 3))
    self.assertEqual("SomeClass(a=1, b=2, c=3)",
                     utils.auto_repr(SomeClass, 1, 2, c=3))
    self.assertEqual("SomeClass(a=1, b=2, c=3)",
                     utils.auto_repr(SomeClass, 1, b=2, c=3))
    self.assertEqual("SomeClass(a=1, b=2, c=3)",
                     utils.auto_repr(SomeClass, a=1, b=2, c=3))


class ReplicateTest(parameterized.TestCase):

  @parameterized.named_parameters(("Int", 42), ("String", "foo"),
                                  ("Callable", lambda a: a))
  def testSingleValue(self, value):
    result = utils.replicate(value, 3, "value")
    self.assertLen(result, 3)
    self.assertEqual(result, (value,) * 3)

  @parameterized.named_parameters(("Int", 42), ("String", "foo"),
                                  ("Callable", lambda a: a))
  def testListLengthOne(self, value):
    result = utils.replicate([value], 3, "value")
    self.assertLen(result, 3)
    self.assertEqual(result, (value,) * 3)

  @parameterized.named_parameters(("Int", 42), ("String", "foo"),
                                  ("Callable", lambda a: a))
  def testTupleLengthN(self, value):
    v = (value,) * 3
    result = utils.replicate(v, 3, "value")
    self.assertLen(result, 3)
    self.assertEqual(result, (value,) * 3)

  @parameterized.named_parameters(("Int", 42), ("String", "foo"),
                                  ("Callable", lambda a: a))
  def testListLengthN(self, value):
    v = list((value,) * 3)
    result = utils.replicate(v, 3, "value")
    self.assertLen(result, 3)
    self.assertEqual(result, (value,) * 3)

  def testIncorrectLength(self):
    v = [2, 2]
    with self.assertRaisesRegex(
        TypeError,
        r"must be a scalar or sequence of length 1 or sequence of length 3"):
      utils.replicate(v, 3, "value")


class SomeClass(object):

  def __init__(self, a, b, c=2):
    pass


class ChannelIndexTest(parameterized.TestCase):

  @parameterized.parameters("channels_first", "NCHW", "NC", "NCDHW")
  def test_returns_index_channels_first(self, data_format):
    self.assertEqual(utils.get_channel_index(data_format), 1)

  @parameterized.parameters("channels_last", "NHWC", "NDHWC", "BTWHD", "TBD")
  def test_returns_index_channels_last(self, data_format):
    self.assertEqual(utils.get_channel_index(data_format), -1)

  @parameterized.parameters("foo", "NCHC", "BTDTD", "chanels_first", "NHW")
  def test_invalid_strings(self, data_format):
    with self.assertRaisesRegex(
        ValueError,
        "Unable to extract channel information from '{}'.".format(data_format)):
      utils.get_channel_index(data_format)

if __name__ == "__main__":
  absltest.main()
