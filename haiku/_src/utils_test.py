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
from haiku._src import test_utils
from haiku._src import utils
import jax.numpy as jnp
import numpy as np

lines = lambda *a: "\n".join(a)


class UtilsTest(parameterized.TestCase):

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

  SHAPES = (("r0", []), ("r1", [1]), ("r2", [200, 200]), ("r3", [2, 3, 4]),
            ("r1_empty", [0]))
  DTYPES = (("f32", np.float32), ("f16", np.float16), ("s8", np.int8),
            ("bf16", jnp.bfloat16))
  CONTAINERS = (("list", lambda x: [x]), ("tuple", lambda x: (x,)),
                ("dict", lambda x: {"a": x}))

  @test_utils.combined_named_parameters(SHAPES, DTYPES, CONTAINERS)
  def test_tree_size(self, shape, dtype, container):
    x = np.ones(shape, dtype=dtype)
    expected_size = np.prod(x.shape) if x.ndim else 1
    self.assertEqual(utils.tree_size(container(x)), expected_size)

  @test_utils.combined_named_parameters(SHAPES, DTYPES, CONTAINERS)
  def test_tree_bytes(self, shape, dtype, container):
    x = np.ones(shape, dtype=dtype)
    expected_bytes = (np.prod(x.shape) if x.ndim else 1) * x.itemsize
    self.assertEqual(utils.tree_bytes(container(x)), expected_bytes)

  def test_format_array(self):
    self.assertEqual(utils.format_array(np.ones([], np.float32)), "f32[]")
    self.assertEqual(utils.format_array(np.ones([1, 2], np.int8)), "s8[1,2]")
    self.assertEqual(utils.format_array(np.ones([], jnp.bfloat16)), "bf16[]")

  def test_format_bytes(self):
    self.assertEqual(utils.format_bytes(0), "0.00 B")
    self.assertEqual(utils.format_bytes(999), "999.00 B")
    self.assertEqual(utils.format_bytes(1234), "1.23 KB")
    self.assertEqual(utils.format_bytes(1235), "1.24 KB")
    self.assertEqual(utils.format_bytes(999010), "999.01 KB")
    self.assertEqual(utils.format_bytes(1e3), "1.00 KB")
    self.assertEqual(utils.format_bytes(2e6), "2.00 MB")
    self.assertEqual(utils.format_bytes(3e9), "3.00 GB")
    self.assertEqual(utils.format_bytes(4e12), "4.00 TB")
    self.assertEqual(utils.format_bytes(5e20), "500000000.00 TB")


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


class SomeClass:

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
