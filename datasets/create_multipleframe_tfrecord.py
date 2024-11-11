import os
import random
from datasets import my_util as util
import apache_beam as beam
import tensorflow as tf
from absl import app, flags, logging
from typing import List

_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    default='moving_mnist_frames_all',
    help='Path to the root directory of the moving_mnist_frames dataset.'
)

_OUTPUT_TFRECORD_FILEPATH = flags.DEFINE_string(
    'output_tfrecord_filepath',
    default='mnist-tfrecord',
    help='Filepath prefix for the output TFRecord files.'
)

# Set num_shards to 1
_NUM_SHARDS = 1

# 20-frame map
_INTERPOLATOR_IMAGES_MAP = {
    f'frame_{i}': f'frame{i}.png' for i in range(6)
}


class LoggingFn(beam.DoFn):
    def __init__(self, dataset_dir):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.counter = 0

    def process(self, element):
        self.counter += 1
        logging.info(f"Processing sequence {self.counter} from dataset directory: {self.dataset_dir}")
        logging.info(f"Sequence content: {element}")
        yield element


def main(unused_argv):
    sequence_list = tf.io.gfile.listdir(_INPUT_DIR.value)
    frame_dicts = []
    for sequence_folder in sequence_list:
        frame_files = tf.io.gfile.listdir(os.path.join(_INPUT_DIR.value, sequence_folder))
        frame_dict = {
            image_key: os.path.join(_INPUT_DIR.value, sequence_folder, image_basename)
            for image_key, image_basename in _INTERPOLATOR_IMAGES_MAP.items()
            if image_basename in frame_files
        }
        if len(frame_dict) >= 6:
            frame_dicts.append(frame_dict)

    split_ratio = 0.8
    split_index = int(len(frame_dicts) * split_ratio)
    train_frame_dicts = frame_dicts[:split_index]
    eval_frame_dicts = frame_dicts[split_index:]

    spit_datasets = 10

    semi_train_frame_dicts = random.sample(train_frame_dicts, len(train_frame_dicts) // spit_datasets)
    semi_eval_frame_dicts = random.sample(eval_frame_dicts, len(eval_frame_dicts) // spit_datasets)


    # Semi-supervised training TFRecord
    with beam.Pipeline('DirectRunner') as p:
        (p | 'ReadSemiTrainInputFrameDicts' >> beam.Create(semi_train_frame_dicts)
           | 'LogSemiTrainData' >> beam.ParDo(LoggingFn(_INPUT_DIR.value))  # Log transformed data
           | 'GenerateSemiTrainExample' >> beam.ParDo(util.ExampleGenerator(_INTERPOLATOR_IMAGES_MAP))
           | 'WriteSemiTrainToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=_OUTPUT_TFRECORD_FILEPATH.value + '_semi_train',
                num_shards=_NUM_SHARDS,
                coder=beam.coders.BytesCoder()))

    # Semi-supervised evaluation TFRecord
    with beam.Pipeline('DirectRunner') as p:
        (p | 'ReadSemiEvalInputFrameDicts' >> beam.Create(semi_eval_frame_dicts)
           | 'LogSemiEvalData' >> beam.ParDo(LoggingFn(_INPUT_DIR.value))  # Log transformed data
           | 'GenerateSemiEvalExample' >> beam.ParDo(util.ExampleGenerator(_INTERPOLATOR_IMAGES_MAP))
           | 'WriteSemiEvalToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix=_OUTPUT_TFRECORD_FILEPATH.value + '_semi_eval',
                num_shards=_NUM_SHARDS,
                coder=beam.coders.BytesCoder()))

    logging.info('Successfully created semi-supervised training and evaluation TFRecord files.')

if __name__ == '__main__':
    app.run(main)