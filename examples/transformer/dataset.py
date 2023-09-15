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
"""A simple example loader for an ASCII language-modelling dataset."""

from collections.abc import Iterable, Iterator
import itertools
import random
from typing import NamedTuple, TypeVar

import numpy as np

VOCAB_SIZE = 128  # Number of ASCII code points.
PAD_TOKEN = 0

_T = TypeVar('_T')


class Batch(NamedTuple):
  inputs: np.ndarray  # Integer tokens, shape [B, T].
  targets: np.ndarray  # Integer tokens, shape [B, T].


def repeat(dataset: Iterable[_T]) -> Iterator[_T]:
  return itertools.cycle(dataset)


def shuffle(dataset: Iterator[_T], buffer_size: int) -> Iterator[_T]:
  buffer = [next(dataset) for _ in range(buffer_size)]
  random.shuffle(buffer)
  for item in dataset:
    idx = random.randint(0, buffer_size - 1)  # Inclusive.
    result = buffer[idx]
    buffer[idx] = item
    yield result


def load_ascii_dataset(
    corpus: str,
    *,
    batch_size: int,
    sequence_length: int,
    num_shuffle_batches: int = 10,
) -> Iterator[Batch]:
  """Loads a single-file ASCII dataset in memory."""

  if not corpus.isascii():
    raise ValueError('Loaded corpus is not ASCII.')

  if chr(PAD_TOKEN) in corpus:  # Reserve 0 codepoint for pad token.
    raise ValueError('Corpus must not contain the null byte.')

  # Naively tokenise by taking ASCII codepoints.
  corpus = np.array([ord(c) for c in corpus]).astype(np.int32)
  assert np.max(corpus) < VOCAB_SIZE

  crop_len = sequence_length + 1
  num_batches, remainder = divmod(corpus.size, batch_size * crop_len)
  if remainder:
    corpus = corpus[:-remainder]  # Drop remainder (incomplete) batch.
  ds = corpus.reshape([-1, crop_len])

  if num_batches < num_shuffle_batches:
    raise ValueError(
        f'Only {num_batches} batches in the dataset; consider using a shorter '
        'sequence length or a smaller batch batch size.',
    )

  ds = repeat(ds)
  ds = shuffle(ds, buffer_size=batch_size * num_shuffle_batches)
  while True:
    batch = np.stack([next(ds) for _ in range(batch_size)])
    yield Batch(inputs=batch[:, :-1], targets=batch[:, 1:])
