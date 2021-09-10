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
import itertools as it
import types
from typing import Generator, Iterable, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from packaging import version
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

Batch = Mapping[str, np.ndarray]
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class Split(enum.Enum):
  """Imagenet dataset split."""
  TRAIN = 1
  TRAIN_AND_VALID = 2
  VALID = 3
  TEST = 4

  @classmethod
  def from_string(cls, name: str) -> 'Split':
    return {'TRAIN': Split.TRAIN, 'TRAIN_AND_VALID': Split.TRAIN_AND_VALID,
            'VALID': Split.VALID, 'VALIDATION': Split.VALID,
            'TEST': Split.TEST}[name.upper()]

  @property
  def num_examples(self):
    return {Split.TRAIN_AND_VALID: 1281167, Split.TRAIN: 1271167,
            Split.VALID: 10000, Split.TEST: 50000}[self]


def _check_min_version(mod: types.ModuleType, min_ver: str):
  actual_ver = getattr(mod, '__version__')
  if version.parse(actual_ver) < version.parse(min_ver):
    raise ValueError(
        f'{mod.__name__} >= {min_ver} is required, you have {actual_ver}')


def check_versions():
  _check_min_version(tf, '2.5.0')
  _check_min_version(tfds, '4.2.0')


def load(
    split: Split,
    *,
    is_training: bool,
    batch_dims: Sequence[int],
    dtype: jnp.dtype = jnp.float32,
    transpose: bool = False,
    zeros: bool = False,
) -> Generator[Batch, None, None]:
  """Loads the given split of the dataset."""
  if zeros:
    h, w, c = 224, 224, 3
    if transpose:
      image_dims = (*batch_dims[:-1], h, w, c, batch_dims[0])
    else:
      image_dims = (*batch_dims, h, w, c)
    batch = {'images': np.zeros(image_dims, dtype=dtype),
             'labels': np.zeros(batch_dims, dtype=np.uint32)}
    if is_training:
      yield from it.repeat(batch)
    else:
      num_batches = split.num_examples // np.prod(batch_dims)
      yield from it.repeat(batch, num_batches)

  if is_training:
    start, end = _shard(split, jax.host_id(), jax.host_count())
  else:
    start, end = _shard(split, 0, 1)
  tfds_split = tfds.core.ReadInstruction(_to_tfds_split(split),
                                         from_=start, to=end, unit='abs')
  ds = tfds.load('imagenet2012:5.*.*', split=tfds_split,
                 decoders={'image': tfds.decode.SkipDecoding()})

  total_batch_size = np.prod(batch_dims)

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_optimization.map_parallelization = True
  if is_training:
    options.experimental_deterministic = False
  ds = ds.with_options(options)

  if is_training:
    if jax.host_count() > 1:
      # Only cache if we are reading a subset of the dataset.
      ds = ds.cache()
    ds = ds.repeat()
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=0)

  else:
    if split.num_examples % total_batch_size != 0:
      raise ValueError(f'Test/valid must be divisible by {total_batch_size}')

  def preprocess(example):
    image = _preprocess_image(example['image'], is_training)
    label = tf.cast(example['label'], tf.int32)
    return {'images': image, 'labels': label}

  ds = ds.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def transpose_fn(batch):
    # We use the "double transpose trick" to improve performance for TPUs. Note
    # that this (typically) requires a matching HWCN->NHWC transpose in your
    # model code. The compiler cannot make this optimization for us since our
    # data pipeline and model are compiled separately.
    batch = dict(**batch)
    batch['images'] = tf.transpose(batch['images'], (1, 2, 3, 0))
    return batch

  def cast_fn(batch):
    batch = dict(**batch)
    batch['images'] = tf.cast(batch['images'], tf.dtypes.as_dtype(dtype))
    return batch

  for i, batch_size in enumerate(reversed(batch_dims)):
    ds = ds.batch(batch_size)
    if i == 0:
      if transpose:
        ds = ds.map(transpose_fn)  # NHWC -> HWCN
      # NOTE: You may be tempted to move the casting earlier on in the pipeline,
      # but for bf16 some operations will end up silently placed on the TPU and
      # this causes stalls while TF and JAX battle for the accelerator.
      if dtype != jnp.float32:
        ds = ds.map(cast_fn)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  yield from tfds.as_numpy(ds)


def _device_put_sharded(sharded_tree, devices):
  leaves, treedef = jax.tree_flatten(sharded_tree)
  n = leaves[0].shape[0]
  return jax.device_put_sharded(
      [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)],
      devices)


def double_buffer(ds: Iterable[Batch]) -> Generator[Batch, None, None]:
  """Keeps at least two batches on the accelerator.

  The current GPU allocator design reuses previous allocations. For a training
  loop this means batches will (typically) occupy the same region of memory as
  the previous batch. An issue with this is that it means we cannot overlap a
  host->device copy for the next batch until the previous step has finished and
  the previous batch has been freed.

  By double buffering we ensure that there are always two batches on the device.
  This means that a given batch waits on the N-2'th step to finish and free,
  meaning that it can allocate and copy the next batch to the accelerator in
  parallel with the N-1'th step being executed.

  Args:
    ds: Iterable of batches of numpy arrays.

  Yields:
    Batches of sharded device arrays.
  """
  batch = None
  devices = jax.local_devices()
  for next_batch in ds:
    assert next_batch is not None
    next_batch = _device_put_sharded(next_batch, devices)
    if batch is not None:
      yield batch
    batch = next_batch
  if batch is not None:
    yield batch


def _to_tfds_split(split: Split) -> tfds.Split:
  """Returns the TFDS split appropriately sharded."""
  # NOTE: Imagenet did not release labels for the test split used in the
  # competition, so it has been typical at DeepMind to consider the VALID
  # split the TEST split and to reserve 10k images from TRAIN for VALID.
  if split in (Split.TRAIN, Split.TRAIN_AND_VALID, Split.VALID):
    return tfds.Split.TRAIN
  else:
    assert split == Split.TEST
    return tfds.Split.VALIDATION


def _shard(split: Split, shard_index: int, num_shards: int) -> Tuple[int, int]:
  """Returns [start, end) for the given shard index."""
  assert shard_index < num_shards
  arange = np.arange(split.num_examples)
  shard_range = np.array_split(arange, num_shards)[shard_index]
  start, end = shard_range[0], (shard_range[-1] + 1)
  if split == Split.TRAIN:
    # Note that our TRAIN=TFDS_TRAIN[10000:] and VALID=TFDS_TRAIN[:10000].
    offset = Split.VALID.num_examples
    start += offset
    end += offset
  return start, end


def _preprocess_image(
    image_bytes: tf.Tensor,
    is_training: bool,
) -> tf.Tensor:
  """Returns processed and resized images."""
  if is_training:
    image = _decode_and_random_crop(image_bytes)
    image = tf.image.random_flip_left_right(image)
  else:
    image = _decode_and_center_crop(image_bytes)
  assert image.dtype == tf.uint8
  # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
  # clamping overshoots. This means values returned will be outside the range
  # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
  image = tf.image.resize(image, [224, 224], tf.image.ResizeMethod.BICUBIC)
  image = _normalize_image(image)
  return image


def _normalize_image(image: tf.Tensor) -> tf.Tensor:
  """Normalize the image to zero mean and unit variance."""
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
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
