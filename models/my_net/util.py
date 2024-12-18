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
"""Various utilities used in the film_net frame interpolator model."""
from typing import List

from models.my_net.options import Options
import tensorflow as tf
import tensorflow_addons.image as tfa_image


def build_image_pyramid(image: tf.Tensor,
                        options: Options) -> List[tf.Tensor]:
    """Builds an image pyramid from a given image.

    The original image is included in the pyramid and the rest are generated by
    successively halving the resolution.

    Args:
      image: the input image.
      options: film_net options object

    Returns:
      A list of images starting from the finest with options.pyramid_levels items
    """
    levels = options.pyramid_levels
    pyramid = []
    pool = tf.keras.layers.AveragePooling2D(
        pool_size=2, strides=2, padding='valid')
    for i in range(0, levels):
        pyramid.append(image)
        if i < levels - 1:
            image = pool(image)
    return pyramid


def warp(image: tf.Tensor, flow: tf.Tensor) -> tf.Tensor:
    """Backward warps the image using the given flow.

    Specifically, the output pixel in batch b, at position x, y will be computed
    as follows:
      (flowed_y, flowed_x) = (y+flow[b, y, x, 1], x+flow[b, y, x, 0])
      output[b, y, x] = bilinear_lookup(image, b, flowed_y, flowed_x)

    Note that the flow vectors are expected as [x, y], e.g. x in position 0 and
    y in position 1.

    Args:
      image: An image with shape BxHxWxC.
      flow: A flow with shape BxHxWx2, with the two channels denoting the relative
        offset in order: (dx, dy).
    Returns:
      A warped image.
    """
    # tfa_image.dense_image_warp expects unconventional negated optical flow, so
    # negate the flow here. Also revert x and y for compatibility with older saved
    # models trained with custom warp op that stored (x, y) instead of (y, x) flow
    # vectors.
    flow = -flow[..., ::-1]

    # Note: we have to wrap tfa_image.dense_image_warp into a Keras Lambda,
    # because it is not compatible with Keras symbolic tensors and we want to use
    # this code as part of a Keras model.  Wrapping it into a lambda has the
    # consequence that tfa_image.dense_image_warp is only called once the tensors
    # are concrete, e.g. actually contain data. The inner lambda is a workaround
    # for passing two parameters, e.g you would really want to write:
    # tf.keras.layers.Lambda(tfa_image.dense_image_warp)(image, flow), but this is
    # not supported by the Keras Lambda.
    warped = tf.keras.layers.Lambda(
        lambda x: tfa_image.dense_image_warp(*x))((image, flow))
    return tf.reshape(warped, shape=tf.shape(image))


def multiply_pyramid(pyramid: List[List[tf.Tensor]],
                     scalars: List[tf.Tensor]) -> List[List[tf.Tensor]]:
    """
    Multiplies all image batches in the pyramid by a corresponding batch of scalars
    for each time step.

    Args:
      pyramid: A nested list where each level contains a list of time_step image batches.
      scalars: A list of scalars for each time_step.

    Returns:
      A nested list with each level containing time_step images multiplied by
      their corresponding scalar.
    """
    return [
        [
            tf.transpose(tf.transpose(image, [3, 1, 2, 0]) * scalar, [3, 1, 2, 0])
            for image, scalar in zip(time_step_images, scalars)
        ]
        for time_step_images in pyramid
    ]


def flow_pyramid_synthesis(
        residual_pyramids: List[List[tf.Tensor]]) -> List[List[tf.Tensor]]:
    """Converts a list of residual flow pyramids (one for each time step) into flow pyramids.

    Args:
      residual_pyramids: A list where each element is a residual pyramid for a specific time step.
                         Each residual pyramid is a list of tensors in fine-to-coarse order.

    Returns:
      A list of flow pyramids, one for each time step, each containing tensors in fine-to-coarse order.
    """
    flow_pyramids = []

    for residual_pyramid in residual_pyramids:
        # Initialize with the coarsest level flow from the current residual pyramid
        flow = residual_pyramid[-1]
        flow_pyramid = [flow]

        # Coarse-to-fine refinement for the current time step
        for residual_flow in reversed(residual_pyramid[:-1]):
            level_size = tf.shape(residual_flow)[1:3]
            flow = tf.image.resize(images=2 * flow, size=level_size)
            flow = residual_flow + flow
            flow_pyramid.append(flow)

        # Reverse the current flow pyramid to fine-to-coarse order and add to the output
        flow_pyramids.append(list(reversed(flow_pyramid)))

    return flow_pyramids


from typing import List
import tensorflow as tf


def pyramid_warp(feature_pyramid: List[tf.Tensor],
                 flow_pyramid: List[List[tf.Tensor]]) -> List[List[tf.Tensor]]:
    """Warps the feature pyramid for each time step using the flow pyramid.

    Args:
      feature_pyramid: Feature pyramid with multiple levels (List of tensors, each tensor is one level).
      flow_pyramid: Flow fields with multiple time steps (List of levels, each level with multiple time steps).

    Returns:
      A warped feature pyramid for each time step (List of time steps, each containing a list of warped levels).
    """
    warped_feature_pyramids = []

    # Iterate over each time step in the flow pyramid
    for time_step_idx, time_step_flows in enumerate(flow_pyramid):
        time_step_warped = []

        # Warp each level in the feature pyramid with the corresponding flow level
        for level_idx, (features, flow) in enumerate(zip(feature_pyramid, time_step_flows)):
            if flow.shape[-1] != 2:
                raise ValueError(f"Flow tensor must have 2 channels in the last dimension, got shape: {flow.shape}")

            # Warp the feature map at the current level using the flow at the same level
            warped_features = warp(features, flow)
            time_step_warped.append(warped_features)

        warped_feature_pyramids.append(time_step_warped)

    return warped_feature_pyramids


def concatenate_pyramids(pyramid1: List[tf.Tensor],
                         pyramid2: List[tf.Tensor]) -> List[tf.Tensor]:
    """Concatenates each pyramid level together in the channel dimension."""
    result = []
    for features1, features2 in zip(pyramid1, pyramid2):
        result.append(tf.concat([features1, features2], axis=-1))
    return result


def concatenate_pyramids_with_time(pyramid1: List[List[tf.Tensor]],
                                  pyramid2: List[List[tf.Tensor]]) -> List[List[tf.Tensor]]:
    """Concatenates each pyramid level together in the channel dimension for each time step."""
    result = []
    for time_step_pyramid1, time_step_pyramid2 in zip(pyramid1, pyramid2):
        time_step_result = []
        for features1, features2 in zip(time_step_pyramid1, time_step_pyramid2):
            # Concatenate along the channel dimension (-1) for each level at each time step
            time_step_result.append(tf.concat([features1, features2], axis=-1))
        result.append(time_step_result)
    return result