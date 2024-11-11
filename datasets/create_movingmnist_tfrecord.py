# create_moving_mnist_tfrecord.py
import os
from datasets import util
import numpy as np
import apache_beam as beam
import tensorflow as tf
from absl import app, flags, logging
from typing import Mapping, Any, List

_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    default='moving_mnist_triplets_test',
    help='Path to the root directory of the moving_mnist_frames dataset.')

_OUTPUT_TFRECORD_FILEPATH = flags.DEFINE_string(
    'output_tfrecord_filepath',
    default='moving_mnist.tfrecord',
    help='Filepath to the output TFRecord file.')

_NUM_SHARDS = flags.DEFINE_integer(
    'num_shards',
    default=3,
    help='Number of shards used for the output.')

_INTERPOLATOR_IMAGES_MAP = {
    'frame_0': 'frame0.png',
    'frame_1': 'frame1.png',
    'frame_2': 'frame2.png',
}


def main(unused_argv):
    """Creates and runs a Beam pipeline to write frame triplets as a TFRecord."""
    # Collect the list of folder paths containing the input and golen frames.
    sequence_list = tf.io.gfile.listdir(_INPUT_DIR.value)

    triplet_dicts = []
    for sequence_folder in sequence_list:
        triplet_folders = tf.io.gfile.listdir(os.path.join(_INPUT_DIR.value, sequence_folder))

        triplet_dict = {
            image_key: os.path.join(
                _INPUT_DIR.value, sequence_folder, image_basename
            )
            for image_key, image_basename in _INTERPOLATOR_IMAGES_MAP.items()
        }
        triplet_dicts.append(triplet_dict)

    # Apache Beam 파이프라인 설정 및 실행
    p = beam.Pipeline('DirectRunner')
    (p | 'ReadInputTripletDicts' >> beam.Create(triplet_dicts)
     | 'GenerateSingleExample' >> beam.ParDo(
                util.ExampleGenerator(_INTERPOLATOR_IMAGES_MAP))
     | 'WriteToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=_OUTPUT_TFRECORD_FILEPATH.value,
                num_shards=_NUM_SHARDS.value,
                coder=beam.coders.BytesCoder()))
    result = p.run()
    result.wait_until_finish()

    logging.info('성공적으로 TFRecord 파일이 생성되었습니다: \'%s@%s\'.',
                 _OUTPUT_TFRECORD_FILEPATH.value, str(_NUM_SHARDS.value))

if __name__ == '__main__':
    app.run(main)
