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
r"""Runs the FILM frame interpolator on a pair of frames on beam.

This script is used evaluate the output quality of the FILM Tensorflow frame
interpolator. Optionally, it outputs a video of the interpolated frames.

A beam pipeline for invoking the frame interpolator on a set of directories
identified by a glob (--pattern). Each directory is expected to contain two
input frames that are the inputs to the frame interpolator. If a directory has
more than two frames, then each contiguous frame pair is treated as input to
generate in-between frames.

The output video is stored to interpolator.mp4 in each directory. The number of
frames is determined by --times_to_interpolate, which controls the number of
times the frame interpolator is invoked. When the number of input frames is 2,
the number of output frames is 2^times_to_interpolate+1.

This expects a directory structure such as:
  <root directory of the eval>/01/frame1.png
                                  frame2.png
  <root directory of the eval>/02/frame1.png
                                  frame2.png
  <root directory of the eval>/03/frame1.png
                                  frame2.png
  ...

And will produce:
  <root directory of the eval>/01/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  <root directory of the eval>/02/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  <root directory of the eval>/03/interpolated_frames/frame0.png
                                                      frame1.png
                                                      frame2.png
  ...

And optionally will produce:
  <root directory of the eval>/01/interpolated.mp4
  <root directory of the eval>/02/interpolated.mp4
  <root directory of the eval>/03/interpolated.mp4
  ...

Usage example:
  python3 -m frame_interpolation.eval.interpolator_cli \
    --model_path <path to TF2 saved model> \
    --pattern "<root directory of the eval>/*" \
    --times_to_interpolate <Number of times to interpolate>
"""

import functools
import os
from typing import List, Sequence

from eval import interpolator as interpolator_lib
from eval import util
from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import mediapy as media
import natsort
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

_PATTERN = flags.DEFINE_string(
    name='pattern',
    default=None,
    help='The pattern to determine the directories with the input frames.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_TIMES_TO_INTERPOLATE = flags.DEFINE_integer(
    name='times_to_interpolate',
    default=5,
    help='The number of times to run recursive midpoint interpolation. '
         'The number of output frames will be 2^times_to_interpolate+1.')
_FPS = flags.DEFINE_integer(
    name='fps',
    default=30,
    help='Frames per second to play interpolated videos in slow motion.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=64,
    help='If >1, pad the input size so it is evenly divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='An int >= 1, number of patches along height, '
         'patch_height = height//block_height, should be evenly divisible.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='An int >= 1, number of patches along width, '
         'patch_width = width//block_width, should be evenly divisible.')
_OUTPUT_VIDEO = flags.DEFINE_boolean(
    name='output_video',
    default=False,
    help='If true, creates a video of the frames in the interpolated_frames/ '
         'subdirectory')

# Add other extensions, if not either.
_INPUT_EXT = ['png', 'jpg', 'jpeg']


def _output_frames(frames: List[np.ndarray], frames_dir: str):
    """Writes PNG-images to a directory.

    If frames_dir doesn't exist, it is created. If frames_dir contains existing
    PNG-files, they are removed before saving the new ones.

    Args:
      frames: List of images to save.
      frames_dir: The output directory to save the images.

    """
    if tf.io.gfile.isdir(frames_dir):
        old_frames = tf.io.gfile.glob(f'{frames_dir}/frame_*.png')
        if old_frames:
            logging.info('Removing existing frames from %s.', frames_dir)
            for old_frame in old_frames:
                tf.io.gfile.remove(old_frame)
    else:
        tf.io.gfile.makedirs(frames_dir)
    for idx, frame in tqdm(
            enumerate(frames), total=len(frames), ncols=100, colour='green'):
        util.write_image(f'{frames_dir}/frame_{idx:03d}.png', frame)
    logging.info('Output frames saved in %s.', frames_dir)


class ProcessDirectory(beam.DoFn):
    def setup(self):
        # Interpolator 객체를 초기화하여 사용
        self.interpolator = interpolator_lib.Interpolator(
            model_path=_MODEL_PATH.value,
            align=_ALIGN.value,
            block_shape=[_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value]
        )

        if _OUTPUT_VIDEO.value:
            ffmpeg_path = util.get_ffmpeg_path()
            media.set_ffmpeg(ffmpeg_path)

    def process(self, directory: str):
        input_frames_list = [
            natsort.natsorted(tf.io.gfile.glob(f'{directory}/*.{ext}'))
            for ext in _INPUT_EXT
        ]
        input_frames = functools.reduce(lambda x, y: x + y, input_frames_list)
        frames = [util.read_image(f) for f in input_frames]

        # Define time steps for each interpolation (between 0 and 1)
        times = np.linspace(0, 1, 18, endpoint=False)

        frame_outputs = []
        for i in range(len(frames) - 1):
            x0, x1 = frames[i], frames[i + 1]
            for t in times:
                # 중간 프레임 생성
                interpolated_frame = self.interpolator.interpolate(
                    np.expand_dims(x0, axis=0),  # 첫 번째 프레임 (배치 축 추가)
                    np.expand_dims(x1, axis=0),  # 두 번째 프레임 (배치 축 추가)
                    np.array([t], dtype=np.float32)  # 중간 시간 값 (배치 축 추가)
                )[0]  # 첫 번째 배치 결과를 사용
                frame_outputs.append(interpolated_frame)

        for idx, frame in enumerate(frame_outputs):
            util.write_image(f'{directory}/interpolated_frames/frame_{idx + 1:03d}.png', frame)


def _run_pipeline() -> None:
    directories = tf.io.gfile.glob(_PATTERN.value)
    pipeline = beam.Pipeline('DirectRunner')
    (pipeline | 'Create directory names' >> beam.Create(directories)  # pylint: disable=expression-not-assigned
     | 'Process directories' >> beam.ParDo(ProcessDirectory()))

    result = pipeline.run()
    result.wait_until_finish()


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    _run_pipeline()


if __name__ == '__main__':
    app.run(main)
