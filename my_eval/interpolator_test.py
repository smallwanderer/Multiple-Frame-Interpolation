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
r"""A test script for mid frame interpolation from two input frames.

Usage example:
 python3 -m frame_interpolation.eval.interpolator_test \
   --frame1 <filepath of the first frame> \
   --frame2 <filepath of the second frame> \
   --model_path <The filepath of the TF2 saved model to use>

The output is saved to <the directory of the input frames>/output_frame.png. If
`--output_frame` filepath is provided, it will be used instead.
"""
import os
from typing import Sequence

from my_eval import interpolator as interpolator_lib
from my_eval import util
from absl import app
from absl import flags
import numpy as np

# Controls TF_CCP log level.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

_FRAME1 = flags.DEFINE_string(
    name='frame1',
    default=None,
    help='The filepath of the first input frame.',
    required=True)
_FRAME2 = flags.DEFINE_string(
    name='frame2',
    default=None,
    help='The filepath of the second input frame.',
    required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
_OUTPUT_FRAME = flags.DEFINE_string(
    name='output_frame',
    default=None,
    help='The output filepath of the interpolated mid-frame.')
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


def check_file_exists(filepath: str, file_description: str) -> None:
    abs_path = os.path.abspath(filepath)
    print(f"Checking {file_description} path: {abs_path}")
    if not os.path.isfile(abs_path):
        print(f"Current working directory: {os.getcwd()}")
        raise FileNotFoundError(f"{file_description} not found: {abs_path}")
    else:
        print(f"{file_description} found: {abs_path}")


def check_model_directory(directory: str) -> None:
    abs_directory = os.path.abspath(directory)
    print(f"Checking model directory: {abs_directory}")
    if not os.path.isdir(abs_directory):
        print(f"Current working directory: {os.getcwd()}")
        raise FileNotFoundError(f"Model directory not found: {abs_directory}")
    else:
        print(f"Model directory found: {abs_directory}")
        files = os.listdir(abs_directory)
        if files:
            print("Files in model directory:")
            for file in files:
                print(file)
        else:
            raise FileNotFoundError(f"No files found in model directory: {abs_directory}")


def _run_interpolator() -> None:
    """Writes interpolated frames from a given two input frame filepaths."""

    abs_frame1 = os.path.abspath(_FRAME1.value)
    abs_frame2 = os.path.abspath(_FRAME2.value)
    abs_model_path = os.path.abspath(_MODEL_PATH.value)
    abs_output_dir = os.path.abspath(_OUTPUT_FRAME.value)

    # Ensure output directory exists
    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)

    print(f"Frame1 path: {abs_frame1}")
    print(f"Frame2 path: {abs_frame2}")
    print(f"Model path: {abs_model_path}")
    print(f"Output path (base): {abs_output_dir}")

    check_file_exists(abs_frame1, "First input frame")
    check_file_exists(abs_frame2, "Second input frame")
    check_model_directory(abs_model_path)

    interpolator = interpolator_lib.Interpolator(
        model_path=abs_model_path,
        align=_ALIGN.value,
        block_shape=[_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])

    # Prepare input frames for interpolation
    image_1 = util.read_image(abs_frame1)
    image_batch_1 = np.expand_dims(image_1, axis=0)
    image_2 = util.read_image(abs_frame2)
    image_batch_2 = np.expand_dims(image_2, axis=0)

    # Define `dt_values` for interpolation points
    num_intermediate_frames = 4  # Define desired number of intermediate frames

    # Interpolate frames
    interpolated_frames = interpolator(image_batch_1, image_batch_2, num_intermediate_frames)
    print("Shape of interpolated_frames:", interpolated_frames[0])

    # Save each frame in the interpolated results list
    for idx, frame_data in enumerate(interpolated_frames):
        frame_filename = os.path.join(abs_output_dir, f"interpolated_frame_{idx}.png")
        util.write_image(frame_filename, frame_data)
        print(f"Saved frame {idx} to: {frame_filename}")


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    _run_interpolator()


if __name__ == '__main__':
    app.run(main)
