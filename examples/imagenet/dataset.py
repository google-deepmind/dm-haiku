# python3
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
"""ImageNet dataset with typical pre-processing."""

import enum
from typing import Generator, Mapping, Optional, Sequence, Text, Tuple

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

Batch = Mapping[Text, np.ndarray]


class Split(enum.Enum):
  """Imagenet dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name: Text) -> 'Split':
    return {'TRAIN': Split.TRAIN, 'TRAIN_AND_VALID': Split.TRAIN_AND_VALID,
            'VALID': Split.VALID, 'VALIDATION': Split.VALID,
            'TEST': Split.TEST}[name.upper()]

  @property
  def num_examples(self):
    return {Split.TRAIN_AND_VALID: 1281167, Split.TRAIN: 1271167,
            Split.VALID: 10000, Split.TEST: 50000}[self]


def load(
    split: Split,
    *,
    batch_dims: Sequence[int],
    bfloat16: bool = False,
) -> Generator[Batch, None, None]:
  """Loads the given split of the dataset."""
  # NOTE: Imagenet did not release labels for the test split used in the
  # competition, so it has been typical at DeepMind to consider the VALID
  # split the TEST split and to reserve 10k images from TRAIN for VALID.
  if split in (Split.TRAIN, Split.TRAIN_AND_VALID, Split.VALID):
    tfds_split = tfds.Split.TRAIN
  else:
    assert split == Split.TEST
    tfds_split = tfds.Split.VALIDATION

  is_training = split in (Split.TRAIN, Split.TRAIN_AND_VALID)
  total_batch_size = np.prod(batch_dims)

  ds = tfds.load('imagenet2012:5.*.*', split=tfds_split,
                 decoders={'image': tfds.decode.SkipDecoding()},
                 shuffle_files=is_training)

  options = ds.options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  if is_training:
    options.experimental_deterministic = False

  if is_training:
    if split == Split.TRAIN:
      ds = ds.skip(Split.VALID.num_examples)
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)

  else:
    if split.num_examples % total_batch_size != 0:
      raise ValueError(f'Test/valid must be divisible by {total_batch_size}')

    if split == Split.VALID:
      ds = ds.take(split.num_examples)

  def preprocess(example):
    dtype = tf.bfloat16 if bfloat16 else tf.float32
    image = _preprocess_image(example['image'], is_training, dtype)
    label = tf.cast(example['label'], tf.int32)
    return {'images': image, 'labels': label}

  ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  for batch_size in reversed(batch_dims):
    ds = ds.batch(batch_size)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  yield from tfds.as_numpy(ds)


def _preprocess_image(
    image_bytes: tf.Tensor,
    is_training: bool,
    dtype: tf.DType,
) -> tf.Tensor:
  """Preprocesses the given image."""
  if is_training:
    image = _decode_and_random_crop(image_bytes)
    image = tf.image.random_flip_left_right(image)
  else:
    image = _decode_and_center_crop(image_bytes)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  image = _normalize_image(image,
                           mean=(0.485, 0.456, 0.406),
                           stddev=(0.229, 0.224, 0.225))
  # TODO(tomhennigan) Update to TF2 symbol and test for equivalence.
  image = tf.compat.v1.image.resize_bicubic([image], [224, 224])[0]
  image = tf.reshape(image, [224, 224, 3])
  return image


def _normalize_image(image, *, mean, stddev):
  """Normalize the image to zero mean and unit variance."""
  image -= tf.constant(mean, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(stddev, shape=[1, 1, 3], dtype=image.dtype)
  return image


def _distorted_bounding_box_crop(
    image_bytes: tf.Tensor,
    *,
    jpeg_shape: tf.Tensor,
    bbox: tf.Tensor,
    min_object_covered: float,
    aspect_ratio_range: Tuple[float, float],
    area_range: Tuple[float, float],
    max_attempts: int,
) -> tf.Tensor:
  """Generates cropped_image using one of the bboxes randomly distorted."""
  bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
      jpeg_shape,
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=aspect_ratio_range,
      area_range=area_range,
      max_attempts=max_attempts,
      use_image_if_no_bounding_boxes=True)

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image


def _decode_and_random_crop(image_bytes: tf.Tensor) -> tf.Tensor:
  """Make a random crop of 224."""
  jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(
      image_bytes,
      jpeg_shape=jpeg_shape,
      bbox=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3 / 4, 4 / 3),
      area_range=(0.08, 1.0),
      max_attempts=10)
  if tf.reduce_all(tf.equal(jpeg_shape, tf.shape(image))):
    # If the random crop failed fall back to center crop.
    image = _decode_and_center_crop(image_bytes, jpeg_shape)
  return image


def _decode_and_center_crop(
    image_bytes: tf.Tensor,
    jpeg_shape: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Crops to center of image with padding then scales."""
  if jpeg_shape is None:
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = jpeg_shape[0]
  image_width = jpeg_shape[1]

  padded_center_crop_size = tf.cast(
      ((224 / (224 + 32)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image
