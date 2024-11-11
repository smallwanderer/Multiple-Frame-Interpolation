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
"""TF2 layer for estimating optical flow by a residual flow pyramid.

This approach of estimating optical flow between two images can be traced back
to [1], but is also used by later neural optical flow computation methods such
as SpyNet [2] and PWC-Net [3].

The basic idea is that the optical flow is first estimated in a coarse
resolution, then the flow is upsampled to warp the higher resolution image and
then a residual correction is computed and added to the estimated flow. This
process is repeated in a pyramid on coarse to fine order to successively
increase the resolution of both optical flow and the warped image.

In here, the optical flow predictor is used as an internal component for the
film_net frame interpolator, to warp the two input images into the inbetween,
target frame.

[1] F. Glazer, Hierarchical motion detection. PhD thesis, 1987.
[2] A. Ranjan and M. J. Black, Optical Flow Estimation using a Spatial Pyramid
    Network. 2016
[3] D. Sun X. Yang, M-Y. Liu and J. Kautz, PWC-Net: CNNs for Optical Flow Using
    Pyramid, Warping, and Cost Volume, 2017
"""

from typing import List

from models.my_net import options
from models.my_net import util
import tensorflow as tf


def _relu(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.leaky_relu(x, alpha=0.2)


class FlowEstimator(tf.keras.layers.Layer):
    """Small-receptive field predictor for computing the flow between two images.

    This is used to compute the residual flow fields in PyramidFlowEstimator.

    Note that while the number of 3x3 convolutions & filters to apply is
    configurable, two extra 1x1 convolutions are appended to extract the flow in
    the end.

    Attributes:
      name: The name of the layer
      num_convs: Number of 3x3 convolutions to apply
      num_filters: Number of filters in each 3x3 convolution
    """

    def __init__(self, name: str, num_convs: int, num_filters: int):
        super(FlowEstimator, self).__init__(name=name)

        def conv(filters, size, name, activation=_relu):
            return tf.keras.layers.Conv2D(
                name=name,
                filters=filters,
                kernel_size=size,
                padding='same',
                activation=activation)

        self._convs = []
        for i in range(num_convs):
            self._convs.append(conv(filters=num_filters, size=3, name=f'conv_{i}'))
        self._convs.append(conv(filters=num_filters / 2, size=1, name=f'conv_{i + 1}'))
        # For the final convolution, we want no activation at all to predict the
        # optical flow vector values. We have done extensive testing on explicitly
        # bounding these values using sigmoid, but it turned out that having no
        # activation gives better results.
        self._convs.append(
            conv(filters=2, size=1, name=f'conv_{i + 2}', activation=None))

    def call(self, features_a: tf.Tensor, features_b: tf.Tensor) -> tf.Tensor:
        """Estimates optical flow between two images.

        Args:
          features_a: per pixel feature vectors for image A (B x H x W x C)
          features_b: per pixel feature vectors for image B (B x H x W x C)

        Returns:
          A tensor with optical flow from A to B
        """
        net = tf.concat([features_a, features_b], axis=-1)
        for conv in self._convs:
            net = conv(net)
        return net


class PyramidFlowEstimator(tf.keras.layers.Layer):
    """Predicts optical flow by coarse-to-fine refinement for multiple time steps.

    Attributes:
      name: The name of the layer
      config: Options for the film_net frame interpolator
    """

    def __init__(self, name: str, config: options.Options):
        super(PyramidFlowEstimator, self).__init__(name=name)
        self._predictors = []
        for i in range(config.specialized_levels):
            self._predictors.append(
                FlowEstimator(
                    name=f'flow_predictor_{i}',
                    num_convs=config.flow_convs[i],
                    num_filters=config.flow_filters[i]))
        shared_predictor = FlowEstimator(
            name='flow_predictor_shared',
            num_convs=config.flow_convs[-1],
            num_filters=config.flow_filters[-1])
        for i in range(config.specialized_levels, config.pyramid_levels):
            self._predictors.append(shared_predictor)

    def call(self, feature_pyramid_a: List[tf.Tensor], feature_pyramid_b: List[tf.Tensor], time: tf.Tensor) -> List[
        List[tf.Tensor]]:
        """Estimates residual flow pyramids for multiple time steps between two image pyramids.

        Args:
          feature_pyramid_a: image pyramid as a list in fine-to-coarse order
          feature_pyramid_b: image pyramid as a list in fine-to-coarse order
          time: Tensor of shape [B, num_frames], with each element representing a time step.

        Returns:
          A list of flow pyramid lists, each corresponding to a specific time step.
          Each pyramid list contains flow tensors in fine-to-coarse order.
        """
        levels = len(feature_pyramid_a)
        batch_size, num_frames = time.shape

        # List to hold the flow pyramids for each time step.
        flow_pyramids = []

        for frame_idx in range(num_frames):
            # Get the specific time step for the current frame.
            time_step = time[:, frame_idx]  # Shape [B]

            # Initialize flow estimation at the coarsest level.
            v = self._predictors[-1](feature_pyramid_a[-1], feature_pyramid_b[-1])
            residuals = [v]

            # Coarse-to-fine refinement.
            for i in reversed(range(0, levels - 1)):
                level_size = tf.shape(feature_pyramid_a[i])[1:3]
                v = tf.image.resize(images=2 * v, size=level_size)

                # Warp feature_pyramid_b[i] based on the current flow estimate.
                warped = util.warp(feature_pyramid_b[i], v)

                # Estimate the residual flow and add it to the current flow estimate.
                v_residual = self._predictors[i](feature_pyramid_a[i], warped)
                residuals.append(v_residual)
                v = v_residual + v

            # Reversing to maintain the finest-to-coarsest order.
            residuals = list(reversed(residuals))

            # Scale flows according to the time step.
            flow_scaled = [residual * time_step[:, None, None, None] for residual in residuals]
            flow_pyramids.append(flow_scaled)

        return flow_pyramids