# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Evaluate the frame interpolation model from a tfrecord and store results.

This script runs the inference on examples in a tfrecord and generates images
and numeric results according to the gin config. For details, see the
run_evaluation() function below.

Usage example:
  python3 -m frame_interpolation.eval.eval_cli -- \
    --gin_config <path to eval_dataset.gin> \
    --base_folder <the root directory to all training sessions> \
    --label < the foldername of the training session>

or

  python3 -m frame_interpolation.eval.eval_cli -- \
    --gin_config <path to eval_dataset.gin> \
    --model_path <The filepath of the TF2 saved model>

The output is saved at the parent directory of the `model_path`:
<parent directory of model_path>/batch_eval.

The evaluation is run on a GPU by default. Add the `--mode` argument for others.
"""
import collections
import os
from typing import Any, Dict

from eval import util
from absl import app
from absl import flags
from absl import logging
import gin.tf
from losses import losses
import numpy as np
import tensorflow as tf
from training import data_lib


_GIN_CONFIG = flags.DEFINE_string('gin_config', None, 'Gin config file.')
_LABEL = flags.DEFINE_string(
    'label', None, 'Descriptive label for the training session to eval.')
_BASE_FOLDER = flags.DEFINE_string('base_folder', None,
                                   'Root folder of training sessions.')
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use. If _MODEL_PATH argument is '
    'directly specified, _LABEL and _BASE_FOLDER arguments will be ignored.')
_OUTPUT_FRAMES = flags.DEFINE_boolean(
    name='output_frames',
    default=False,
    help='If true, saves the the inputs, groud-truth and interpolated frames.')
_MODE = flags.DEFINE_enum('mode', 'gpu', ['cpu', 'gpu'],
                          'Device to run evaluations.')


@gin.configurable('experiment')
def _get_experiment_config(name) -> Dict[str, Any]:
  """Fetches the gin config."""
  return {
      'name': name,
  }


def _set_visible_devices():
  """Set the visible devices according to running mode."""
  mode_devices = tf.config.list_physical_devices(_MODE.value.upper())
  tf.config.set_visible_devices([], 'GPU')
  tf.config.set_visible_devices([], 'TPU')
  tf.config.set_visible_devices(mode_devices, _MODE.value.upper())
  return


@gin.configurable('evaluation')
def run_evaluation(model_path, tfrecord, output_dir, max_examples, metrics):
    """Runs the eval loop for examples in the tfrecord."""
    model = tf.saved_model.load(model_path)
    with tf.io.gfile.GFile(os.path.join(output_dir, 'readme.txt'), mode='w') as f:
        print('Results for:', file=f)
        print(f' model:   {model_path}', file=f)
        print(f' tfrecord: {tfrecord}', file=f)

    with tf.io.gfile.GFile(
            os.path.join(output_dir, 'results.csv'), mode='w') as csv_file:
        test_losses = losses.test_losses(metrics, [1.0] * len(metrics))
        title_row = ['key'] + list(test_losses)
        print(', '.join(title_row), file=csv_file)

        datasets = data_lib.create_eval_datasets(
            batch_size=1,
            files=[tfrecord],
            names=[os.path.basename(output_dir)],
            max_examples=max_examples)
        dataset = datasets[os.path.basename(output_dir)]

        all_losses = collections.defaultdict(list)
        for example in dataset:
            # 각 중간 프레임 생성
            frame_outputs = {}
            for i in range(1, 19):  # frame1 ~ frame18 중간 프레임
                inputs = {
                    f'x{i-1}': example[f'x{i-1}'],
                    f'x{i+1}': example[f'x{i+1}'],
                    'time': example['time'][i - 1][..., tf.newaxis]
                }
                prediction = model(inputs, training=False)
                frame_outputs[f'frame{i}'] = prediction['image']

            # 경로 및 결과 저장
            path = example['path'][0].numpy().decode('utf-8')
            key = path.rsplit('.', 1)[0].rsplit(os.sep)[-1]
            for name, image in frame_outputs.items():
                util.write_image(
                    os.path.join(output_dir, f'{key}_{name}.png'), image[0].numpy())


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _MODEL_PATH.value is not None:
    model_path = _MODEL_PATH.value
  else:
    model_path = os.path.join(_BASE_FOLDER.value, _LABEL.value, 'saved_model')

  gin.parse_config_files_and_bindings(
      config_files=[_GIN_CONFIG.value],
      bindings=None,
      skip_unknown=True)

  config = _get_experiment_config()  # pylint: disable=no-value-for-parameter
  eval_name = config['name']
  output_dir = os.path.join(
      os.path.dirname(model_path), 'batch_eval', eval_name)
  logging.info('Creating output_dir @ %s ...', output_dir)

  # Copy config file to <base_folder>/<label>/batch_eval/<eval_name>/config.gin.
  tf.io.gfile.makedirs(output_dir)
  tf.io.gfile.copy(
      _GIN_CONFIG.value, os.path.join(output_dir, 'config.gin'), overwrite=True)

  _set_visible_devices()
  logging.info('Evaluating %s on %s ...', eval_name, [
      el.name.split('/physical_device:')[-1]
      for el in tf.config.get_visible_devices()
  ])
  run_evaluation(model_path=model_path, output_dir=output_dir)  # pylint: disable=no-value-for-parameter

  logging.info('Done. Evaluations saved @ %s.', output_dir)

if __name__ == '__main__':
  app.run(main)
