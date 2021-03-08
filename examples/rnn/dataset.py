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
"""Tiny Shakespeare as a language modelling dataset."""

from typing import Iterator, Mapping

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

Batch = Mapping[str, np.ndarray]
NUM_CHARS = 128


def load(
    split: tfds.Split,
    *,
    batch_size: int,
    sequence_length: int,
) -> Iterator[Batch]:
  """Creates the Tiny Shakespeare dataset as a character modelling task."""

  def preprocess_fn(x: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    x = x['text']
    x = tf.strings.unicode_split(x, 'UTF-8')
    x = tf.squeeze(tf.io.decode_raw(x, tf.uint8), axis=-1)
    x = tf.cast(x, tf.int32)
    return {'input': x[:-1], 'target': x[1:]}

  ds = tfds.load(name='tiny_shakespeare', split=split)
  ds = ds.map(preprocess_fn)
  ds = ds.unbatch()
  ds = ds.batch(sequence_length, drop_remainder=True)
  ds = ds.shuffle(100)
  ds = ds.repeat()
  ds = ds.batch(batch_size)
  ds = ds.map(lambda b: tf.nest.map_structure(tf.transpose, b))  # Time major.

  return iter(tfds.as_numpy(ds))


def decode(x: np.ndarray) -> str:
  return ''.join([chr(x) for x in x])


def encode(x: str) -> np.ndarray:
  return np.array([ord(s) for s in x])
