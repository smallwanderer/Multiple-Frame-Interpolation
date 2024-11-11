
from typing import Callable, Dict, List, Optional

from absl import logging
import gin.tf
import tensorflow as tf


def _create_feature_map() -> Dict[str, tf.io.FixedLenFeature]:
    """Creates the feature map for extracting 10 frames."""
    feature_map = {}
    for i in range(10):
        feature_map[f'frame_{i}/encoded'] = tf.io.FixedLenFeature((), tf.string, default_value='')
        feature_map[f'frame_{i}/format'] = tf.io.FixedLenFeature((), tf.string, default_value='jpg')
        feature_map[f'frame_{i}/height'] = tf.io.FixedLenFeature((), tf.int64, default_value=0)
        feature_map[f'frame_{i}/width'] = tf.io.FixedLenFeature((), tf.int64, default_value=0)
    feature_map['path'] = tf.io.FixedLenFeature((), tf.string, default_value='')
    return feature_map


def _parse_example(sample):
    """Parses a serialized sample to assign frame0, frame5, and intermediate frames.

    Args:
      sample: A serialized tf.Example to be parsed.

    Returns:
      A dictionary containing:
        - x0: Tensor representing frame0.
        - x1: Tensor representing frame5.
        - y: List of Tensors representing frames from frame1 to frame4.
        - path: Path to the mid-frame.
    """
    feature_map = _create_feature_map()
    features = tf.io.parse_single_example(sample, feature_map)

    # Assign x0 to frame0, x1 to frame5, and y to frames from frame1 to frame4
    output_dict = {
        'x0': tf.io.decode_image(features['frame_0/encoded'], dtype=tf.float32),
        'x1': tf.io.decode_image(features['frame_5/encoded'], dtype=tf.float32),
        'y': [tf.io.decode_image(features[f'frame_{i}/encoded'], dtype=tf.float32) for i in range(1, 5)],
        'path': features['path'],
    }

    return output_dict

def crop_example(example: Dict[str, tf.Tensor], crop_size: int) -> Dict[str, tf.Tensor]:
    """Random crops x0, x1, and each frame in y to the given size.

    Args:
      example: Input dictionary containing frames to be cropped.
      crop_size: The size to crop frames to. This value is used for both
        height and width.

    Returns:
      Example with cropping applied to x0, x1, and each frame in y.
    """
    def random_crop(frame):
        return tf.image.random_crop(frame, size=[crop_size, crop_size, frame.shape[-1]])

    # Apply cropping to x0 and x1
    example['x0'] = random_crop(example['x0'])
    example['x1'] = random_crop(example['x1'])

    # Apply cropping to each frame in y
    example['y'] = [random_crop(frame) for frame in example['y']]

    return example


def apply_data_augmentation(augmentation_fns, example, augmentation_keys=None):
    """Applies augmentation to frames collectively and then reverts to original format."""
    # Step 1: Convert example to a unified dictionary of frames.
    frames = {f'frame{i}': example['x0'] if i == 0 else example['x1'] if i == 5 else example['y'][i - 1] for i in
              range(6)}

    # Step 2: Apply each augmentation in sequence to all frames.
    for augmentation_function in augmentation_fns.values():
        frames = augmentation_function(frames)

    # Step 3: Reconstruct example with augmented frames.
    example['x0'] = frames['frame0']
    example['x1'] = frames['frame5']
    example['y'] = [frames[f'frame{i}'] for i in range(1, 5)]

    return example


def _create_from_tfrecord(batch_size, file, augmentation_fns, crop_size) -> tf.data.Dataset:
    """Creates a dataset from TFRecord with parsed frames and applies augmentation and cropping."""
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.map(
        _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Perform data augmentation before cropping and batching
    if augmentation_fns is not None:
        dataset = dataset.map(
            lambda x: apply_data_augmentation(augmentation_fns, x),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Apply cropping
    if crop_size > 0:
        dataset = dataset.map(
            lambda x: crop_example(x, crop_size=crop_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch and prefetch for performance optimization
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def _generate_sharded_filenames(filename: str) -> List[str]:
    """Generates filenames of the each file in the sharded filepath.

    Based on github.com/google/revisiting-self-supervised/blob/master/datasets.py.

    Args:
      filename: The sharded filepath.

    Returns:
      A list of filepaths for each file in the shard.
    """
    base, count = filename.split('@')
    count = int(count)
    return ['{}-{:05d}-of-{:05d}'.format(base, i, count) for i in range(count)]


def _create_from_sharded_tfrecord(batch_size,
                                  train_mode,
                                  file,
                                  augmentation_fns,
                                  crop_size,
                                  max_examples=-1) -> tf.data.Dataset:
    """Creates a dataset from a sharded tfrecord."""
    dataset = tf.data.Dataset.from_tensor_slices(
        _generate_sharded_filenames(file))

    # pylint: disable=g-long-lambda
    dataset = dataset.interleave(
        lambda x: _create_from_tfrecord(
            batch_size,
            file=x,
            augmentation_fns=augmentation_fns,
            crop_size=crop_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not train_mode)
    # pylint: enable=g-long-lambda
    dataset = dataset.prefetch(buffer_size=2)
    if max_examples > 0:
        return dataset.take(max_examples)
    return dataset


@gin.configurable('training_dataset')
def create_training_dataset(
        batch_size: int,
        file: Optional[str] = None,
        files: Optional[List[str]] = None,
        crop_size: int = -1,
        crop_sizes: Optional[List[int]] = None,
        augmentation_fns: Optional[Dict[str, Callable[..., tf.Tensor]]] = None
) -> tf.data.Dataset:
    """Creates the training dataset with sequences of frames.

    Args:
      batch_size: Number of images to batch per example.
      file: A path to a sharded tfrecord in <tfrecord>@N format.
      files: A list of paths to sharded tfrecords in <tfrecord>@N format.
      crop_size: If > 0, images are cropped to crop_size x crop_size.
      crop_sizes: List of crop sizes. Each entry applies to the corresponding file in `files`.
      augmentation_fns: A Dict of Callables for data augmentation functions.

    Returns:
      A tensorflow dataset containing sequences of frames.
    """
    if file:
        logging.warning('gin-configurable training_dataset.file is deprecated. '
                        'Use training_dataset.files instead.')
        return _create_from_sharded_tfrecord(batch_size, True, file,
                                             augmentation_fns, crop_size)
    else:
        if not crop_sizes or len(crop_sizes) != len(files):
            raise ValueError('Please pass crop_sizes[] with training_dataset.files.')
        if crop_size > 0:
            raise ValueError(
                'crop_size should not be used with files[], use crop_sizes[] instead.'
            )
        tables = []
        for file, crop_size in zip(files, crop_sizes):
            tables.append(
                _create_from_sharded_tfrecord(batch_size, True, file,
                                              augmentation_fns, crop_size))
        return tf.data.experimental.sample_from_datasets(tables)


@gin.configurable('eval_datasets')
def create_eval_datasets(batch_size: int,
                         files: List[str],
                         names: List[str],
                         crop_size: int = -1,
                         max_examples: int = -1) -> Dict[str, tf.data.Dataset]:
    """Creates the evaluation datasets with sequences of 10 frames.

    Args:
      batch_size: The number of images to batch per example.
      files: List of paths to a sharded tfrecord in <tfrecord>@N format.
      names: List of names of eval datasets.
      crop_size: If > 0, images are cropped to crop_size x crop_size.
      max_examples: If > 0, truncate the dataset to 'max_examples' in length. This
        can be useful for speeding up evaluation loop in case the tfrecord for the
        evaluation set is very large.

    Returns:
      A dict of name to tensorflow dataset for accessing examples that contain the
      input frames 'frame0' to 'frame5' as a sequence tensor in 'frames'.
    """
    return {
        name: _create_from_sharded_tfrecord(batch_size, False, file, None,
                                            crop_size, max_examples)
        for name, file in zip(names, files)
    }
