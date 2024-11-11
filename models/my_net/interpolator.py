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
"""The film_net frame interpolator main model code.

Basics
======
The film_net is an end-to-end learned neural frame interpolator implemented as
a TF2 model. It has the following inputs and outputs:

Inputs:
  x0: image A.
  x1: image B.
  time: desired sub-frame time.

Outputs:
  image: the predicted in-between image at the chosen time in range [0, 1].

Additional outputs include forward and backward warped image pyramids, flow
pyramids, etc., that can be visualized for debugging and analysis.

Note that many training sets only contain triplets with ground truth at
time=0.5. If a model has been trained with such training set, it will only work
well for synthesizing frames at time=0.5. Such models can only generate more
in-between frames using recursion.

Architecture
============
The inference consists of three main stages: 1) feature extraction 2) warping
3) fusion. On high-level, the architecture has similarities to Context-aware
Synthesis for Video Frame Interpolation [1], but the exact architecture is
closer to Multi-view Image Fusion [2] with some modifications for the frame
interpolation use-case.

Feature extraction stage employs the cascaded multi-scale architecture described
in [2]. The advantage of this architecture is that coarse level flow prediction
can be learned from finer resolution image samples. This is especially useful
to avoid overfitting with moderately sized datasets.

The warping stage uses a residual flow prediction idea that is similar to
PWC-Net [3], Multi-view Image Fusion [2] and many others.

The fusion stage is similar to U-Net's decoder where the skip connections are
connected to warped image and feature pyramids. This is described in [2].

Implementation Conventions
====================
Pyramids
--------
Throughtout the model, all image and feature pyramids are stored as python lists
with finest level first followed by downscaled versions obtained by successively
halving the resolution. The depths of all pyramids are determined by
options.pyramid_levels. The only exception to this is internal to the feature
extractor, where smaller feature pyramids are temporarily constructed with depth
options.sub_levels.

Color ranges & gamma
--------------------
The model code makes no assumptions on whether the images are in gamma or
linearized space or what is the range of RGB color values. So a model can be
trained with different choices. This does not mean that all the choices lead to
similar results. In practice the model has been proven to work well with RGB
scale = [0,1] with gamma-space images (i.e. not linearized).

[1] Context-aware Synthesis for Video Frame Interpolation, Niklaus and Liu, 2018
[2] Multi-view Image Fusion, Trinidad et al, 2019
[3] PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume
"""

from models.my_net import feature_extractor
from models.my_net import fusion
from models.my_net import options
from models.my_net import pyramid_flow_estimator
from models.my_net import util
import tensorflow as tf
import numpy as np


def create_model(x0: tf.Tensor, x1: tf.Tensor,
                 config: options.Options) -> tf.keras.Model:
    """Creates a frame interpolator model.

    This frame interpolator model is designed to generate multiple intermediate
    frames at different time steps between two input images. The model uses
    multi-scale feature warping and fusion techniques to achieve accurate frame
    predictions for each specified time step.

    Given two input images (`x0` and `x1`) and a tensor of time values, the model
    predicts interpolated frames at each specified time, enabling interpolation
    for arbitrary points between `x0` and `x1`. This structure is useful for generating
    smooth transitions or slow-motion effects across multiple frames.

    Key features:
      - Multi-time-step prediction: The model can predict frames at various
        intermediate times within a single pass, rather than only at a fixed t = 0.5.
      - Multi-scale processing: Leveraging pyramidal feature hierarchies, the model
        warps and aligns input features at multiple resolutions.
      - Optional auxiliary outputs: Provides additional diagnostic information,
        such as intermediate warped images and flow fields.

    Args:
      x0: First input image as a BxHxWxC tensor.
      x1: Second input image as a BxHxWxC tensor.
      time: A tensor of time values (B x T) indicating the desired interpolation
            times, where T is the number of intermediate frames to generate.
      config: FilmNetOptions object containing model configurations.

    Returns:
      A `tf.keras.Model` that takes 'x0', 'x1', and 'time' as inputs and returns
      a dictionary containing:
        - 'image_t{n}': An interpolated frame at each time step `{n}`, representing
          intermediate frames generated at each specified time step.

      If `config.use_aux_outputs` is set to `True`, the model also returns the
      following auxiliary outputs for each time step:
        - 'x0_warped_t{n}': An intermediate result from warping `x0` at time step `{n}`
        - 'x1_warped_t{n}': An intermediate result from warping `x1` at time step `{n}`
        - 'forward_residual_flow_pyramid_t{n}': Pyramid of forward residual flows at time step `{n}`
        - 'backward_residual_flow_pyramid_t{n}': Pyramid of backward residual flows at time step `{n}`
        - 'forward_flow_pyramid_t{n}': Pyramid of forward flows at time step `{n}`
        - 'backward_flow_pyramid_t{n}': Pyramid of backward flows at time step `{n}`

    Raises:
      ValueError: If `config.pyramid_levels` is less than `config.fusion_pyramid_levels`.
    """

    if config.pyramid_levels < config.fusion_pyramid_levels:
        raise ValueError('config.pyramid_levels must be greater than or equal to '
                         'config.fusion_pyramid_levels.')

    x0_decoded = x0
    x1_decoded = x1

    batch_size = x0.shape[0] if x0.shape[0] is not None else 1
    num_frames = 4  # Number of intermediate frames
    time_values = np.linspace(0, 1, num_frames + 2)[1:-1]
    time = tf.constant(time_values.reshape(1, -1).repeat(batch_size, axis=0), dtype=tf.float32)

    # shuffle images
    image_pyramids = [
        util.build_image_pyramid(x0_decoded, config),
        util.build_image_pyramid(x1_decoded, config)
    ]

    # Siamese feature pyramids:
    extract = feature_extractor.FeatureExtractor('feat_net', config)
    feature_pyramids = [extract(image_pyramids[0]), extract(image_pyramids[1])]

    # Initialize PyramidFlowEstimator instance
    predict_flow = pyramid_flow_estimator.PyramidFlowEstimator(
        'predict_flow', config)

    # Predict forward flow.
    forward_residual_flow_pyramid = predict_flow(feature_pyramids[0],
                                                           feature_pyramids[1],
                                                           time)
    # Predict backward flow.
    backward_residual_flow_pyramid = predict_flow(feature_pyramids[1],
                                                  feature_pyramids[0],
                                                  time)

    # Concatenate features and images:

    # Note that we keep up to 'fusion_pyramid_levels' levels as only those
    # are used by the fusion module.
    fusion_pyramid_levels = config.fusion_pyramid_levels

    # Note that we keep up to 'fusion_pyramid_levels' levels as only those
    # are used by the fusion module.
    forward_flow_pyramid = util.flow_pyramid_synthesis(forward_residual_flow_pyramid)
    backward_flow_pyramid = util.flow_pyramid_synthesis(backward_residual_flow_pyramid)

    # Limit the flow pyramids to 'fusion_pyramid_levels'
    forward_flow_pyramid = [flow_pyramid[:fusion_pyramid_levels] for flow_pyramid in forward_flow_pyramid]
    backward_flow_pyramid = [flow_pyramid[:fusion_pyramid_levels] for flow_pyramid in backward_flow_pyramid]

    # Multiply the flows with t and 1-t to warp to the desired fractional time.
    # This code no longer fixes the time to 0.5 and instead directly uses the `time` tensor values.

    # Determine the batch size and create a mid_time tensor directly from `time`.
    # `mid_time` is shaped to match each target intermediate frame.
    # Convert `time` to a list of time steps, where each element is a `(batch_size,)` tensor
    scalars = [time[:, i] for i in range(time.shape[1])]

    # Adjust the backward and forward flow pyramids using multiply_pyramid
    backward_flow = util.multiply_pyramid(backward_flow_pyramid, scalars)
    forward_flow = util.multiply_pyramid(forward_flow_pyramid, [1 - s for s in scalars])

    pyramids_to_warp = [
        util.concatenate_pyramids(image_pyramids[0][:fusion_pyramid_levels],
                                  feature_pyramids[0][:fusion_pyramid_levels]),
        util.concatenate_pyramids(image_pyramids[1][:fusion_pyramid_levels],
                                  feature_pyramids[1][:fusion_pyramid_levels])
    ]

    # Warp features and images using the flow. Note that we use backward warping
    # and backward flow is used to read from image 0 and forward flow from
    # image 1.
    forward_warped_pyramid = util.pyramid_warp(pyramids_to_warp[0], backward_flow)
    backward_warped_pyramid = util.pyramid_warp(pyramids_to_warp[1], forward_flow)

    aligned_pyramid = util.concatenate_pyramids_with_time(forward_warped_pyramid,
                                                backward_warped_pyramid)
    aligned_pyramid = util.concatenate_pyramids_with_time(aligned_pyramid, backward_flow)
    aligned_pyramid = util.concatenate_pyramids_with_time(aligned_pyramid, forward_flow)

    # Initialize Fusion layer
    fuse = fusion.Fusion('fusion', config)

    # Call Fusion layer on aligned_pyramid for multiple intermediate frames
    predictions = fuse(aligned_pyramid)  # predictions is now a list of tensors, each for a different time step

    # Initialize output dictionary
    outputs = {}

    # Process each time step's prediction and add to outputs
    for time_step_idx, prediction in enumerate(predictions):
        output_color = prediction[..., :3]  # Extract RGB channels
        outputs[f'image_t{time_step_idx}'] = output_color  # Store in outputs with time step label

    # Add auxiliary outputs if specified in config
    if config.use_aux_outputs:
        for time_step_idx in range(len(predictions)):
            outputs.update({
                f'x0_warped_t{time_step_idx}': forward_warped_pyramid[time_step_idx][0][..., :3],
                f'x1_warped_t{time_step_idx}': backward_warped_pyramid[time_step_idx][0][..., :3],
                f'forward_residual_flow_pyramid_t{time_step_idx}': forward_residual_flow_pyramid[time_step_idx],
                f'backward_residual_flow_pyramid_t{time_step_idx}': backward_residual_flow_pyramid[time_step_idx],
                f'forward_flow_pyramid_t{time_step_idx}': forward_flow_pyramid[time_step_idx],
                f'backward_flow_pyramid_t{time_step_idx}': backward_flow_pyramid[time_step_idx],
            })

    # Build and return model
    model = tf.keras.Model(
        inputs={
            'x0': x0,
            'x1': x1,
        }, outputs=outputs)
    return model
