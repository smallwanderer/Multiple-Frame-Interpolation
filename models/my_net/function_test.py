import tensorflow as tf
import numpy as np
import options
import os
import fusion
import util
import matplotlib.pyplot as plt
from PIL import Image


# Mock options for testing
class MockOptions:
    def __init__(self):
        self.pyramid_levels = 4
        self.specialized_levels = 2
        self.flow_convs = [3, 3, 3, 3]
        self.flow_filters = [32, 64, 128, 128]
        self.sub_levels = 3  # depth of cascaded feature tree
        self.filters = 16  # base number of filters
        self.fusion_pyramid_levels = 3  # Number of levels used by fusion module


# Step 1: Image Loading from Files
def generate_images_from_files(file_path1, file_path2, height=64, width=64):
    # Load images using PIL
    img1 = Image.open(file_path1).resize((width, height))
    img2 = Image.open(file_path2).resize((width, height))

    # Convert images to tensors
    img1 = tf.convert_to_tensor(np.array(img1), dtype=tf.float32) / 255.0
    img2 = tf.convert_to_tensor(np.array(img2), dtype=tf.float32) / 255.0

    # Add batch dimension
    img1 = tf.expand_dims(img1, axis=0)
    img2 = tf.expand_dims(img2, axis=0)

    print(f"Loaded images shape: img1 {img1.shape}, img2 {img2.shape}")
    return img1, img2


# Step 2: Feature Extraction
def extract_features(img_py_1, img_py_2, config):
    from feature_extractor import FeatureExtractor  # Import feature extractor class

    # Initialize the feature extractor
    feature_extractor = FeatureExtractor(name="test_feature_extractor", config=config)

    # Extract features for both images
    feature_pyramids = [feature_extractor(img_py_1), feature_extractor(img_py_2)]
    print("Feature pyramid shapes (img1):", [feat.shape for feat in feature_pyramids[0]])
    print("Feature pyramid shapes (img2):", [feat.shape for feat in feature_pyramids[1]])

    return feature_pyramids


# Step 3: Flow Estimation
def estimate_flow(feature_pyramid_a, feature_pyramid_b, config, time: tf.Tensor):
    from pyramid_flow_estimator import PyramidFlowEstimator  # Import flow estimator class

    # Initialize PyramidFlowEstimator instance
    pyramid_flow_estimator = PyramidFlowEstimator(name="test_flow_estimator", config=config)

    # Generate a mock time tensor with shape [B, num_frames]
    print(f"Time tensor shape: {time.shape}, values: {time.numpy()}")

    # Call the flow estimator to get forward and backward residual flow pyramids
    forward_residual_flow_pyramid = pyramid_flow_estimator(feature_pyramid_a, feature_pyramid_b, time)
    backward_residual_flow_pyramid = pyramid_flow_estimator(feature_pyramid_b, feature_pyramid_a, time)

    # Print shape information for forward and backward residual flow pyramids
    print("\nForward Residual Flow Pyramid Shapes:")
    for time_step, flow_pyramid in enumerate(forward_residual_flow_pyramid):
        print(f"  Time step {time_step}:")
        for level, flow in enumerate(flow_pyramid):
            print(f"    Level {level} flow shape: {flow.shape}")

    print("Flow Estimation Done")

    # Return both residual flow pyramids
    return forward_residual_flow_pyramid, backward_residual_flow_pyramid


# Step 4: Flow Pyramid Synthesis Testing using flow_pyramids
def test_flow_pyramid_synthesis(forward_residual_flow_pyramid, backward_residual_flow_pyramid, config):
    # Convert residual flow pyramids into final flow pyramids for forward and backward directions
    forward_flow_pyramids = util.flow_pyramid_synthesis(forward_residual_flow_pyramid)
    backward_flow_pyramids = util.flow_pyramid_synthesis(backward_residual_flow_pyramid)

    # Limit the flow pyramids to 'fusion_pyramid_levels'
    forward_flow_pyramids = [flow_pyramid[:config.fusion_pyramid_levels] for flow_pyramid in forward_flow_pyramids]
    backward_flow_pyramids = [flow_pyramid[:config.fusion_pyramid_levels] for flow_pyramid in backward_flow_pyramids]

    # Print results for forward and backward flow pyramids
    print("\nForward Flow Pyramid (synthesized):")
    for time_step, flow_pyramid in enumerate(forward_flow_pyramids):
        print(f"\nTime step {time_step}:")
        for level, flow in enumerate(flow_pyramid):
            print(f"  Level {level} flow shape: {flow.shape}")

    print("\nFlow pyramid synthesis tests completed successfully.")

    return forward_flow_pyramids, backward_flow_pyramids


# Step 5
def adjust_flow_pyramid_with_time_scaling(backward_flow_pyramid, forward_flow_pyramid, time):
    """
    Adjusts the backward and forward flow pyramids by scaling each time step based on provided time values.

    Args:
        backward_flow_pyramid: Pyramid of backward flow image batches.
        forward_flow_pyramid: Pyramid of forward flow image batches.
        time: Tensor of time values for scaling each time step (shape: [batch_size, num_frames]).

    Returns:
        Scaled backward and forward flow pyramids.
    """
    # Convert `time` to a list of time steps, where each element is a `(batch_size,)` tensor
    scalars = [time[:, i] for i in range(time.shape[1])]

    # Adjust the backward and forward flow pyramids using multiply_pyramid
    adjusted_backward_flow = util.multiply_pyramid(backward_flow_pyramid, scalars)
    adjusted_forward_flow = util.multiply_pyramid(forward_flow_pyramid, [1 - s for s in scalars])

    return adjusted_backward_flow, adjusted_forward_flow


# Step 6
def img_feature_concatenate_test(image_pyramids, feature_pyramids, config):
    print("\n img_featyre_concatenate_test Start")
    pyramids_to_warp = [
        util.concatenate_pyramids(image_pyramids[0][:config.fusion_pyramid_levels],
                                  feature_pyramids[0][:config.fusion_pyramid_levels]),
        util.concatenate_pyramids(image_pyramids[1][:config.fusion_pyramid_levels],
                                  feature_pyramids[1][:config.fusion_pyramid_levels])
    ]

    # 결합된 피라미드 구조 출력
    print("\nStructure of pyramids_to_warp[0]:")
    for level_idx, level in enumerate(pyramids_to_warp[0]):
        print(f"  Level {level_idx} - Shape: {level.shape}")

    print("img_feature_concatenate_test passed.")

    return pyramids_to_warp


# Step 7
def warp_pyramid_with_pyrmids_to_warp_and_flows(pyramids_to_warp, forward_flow, backward_flow):
    print("\nWarp Test Start")

    # Check input structure for pyramids_to_warp and flow
    print("\nInput Structure:")
    print("pyramids_to_warp[0] (for backward_flow):")
    for level_idx, level in enumerate(pyramids_to_warp[0]):
        print(f"  Level {level_idx} - Shape: {level.shape}")

    print("\nbackward_flow:")
    for time_step_idx, time_step_flow in enumerate(backward_flow):
        print(f"  Time step {time_step_idx}:")
        for level_idx, level_flow in enumerate(time_step_flow):
            print(f"    Level {level_idx} flow shape: {level_flow.shape}")

    # Perform warping
    forward_warped_pyramid = util.pyramid_warp(pyramids_to_warp[0], backward_flow)
    backward_warped_pyramid = util.pyramid_warp(pyramids_to_warp[1], forward_flow)

    # Output structure after warping
    print("\nOutput Structure:")
    print("forward_warped_pyramid:")
    for time_step_idx, time_step_warped in enumerate(forward_warped_pyramid):
        print(f"  Time step {time_step_idx}:")
        for level_idx, level_warped in enumerate(time_step_warped):
            print(f"    Level {level_idx} warped shape: {level_warped.shape}")

    print("\nWarp Test Passed.")

    return forward_warped_pyramid, backward_warped_pyramid


# Full Test Function integrating all steps
def test_feature_extractor_and_flow_estimator(file_path1, file_path2):
    # Initialize mock options
    config = MockOptions()

    # Step 1: Load images from files
    img1, img2 = generate_images_from_files(file_path1, file_path2)
    # Create image pyramids for both images
    image_pyramids = [
        util.build_image_pyramid(img1, config),
        util.build_image_pyramid(img2, config)
    ]
    print(f"Image pyramids shapes: {[pyr.shape for pyr in image_pyramids[0]]}")
    print("Image loading and preprocessing passed.")

    batch_size = img1.shape[0]  # Assuming img1 has batch size info
    num_frames = 8  # Number of intermediate frames
    time_values = np.linspace(0, 1, num_frames + 2)[1:-1]
    time = tf.constant(time_values.reshape(1, -1).repeat(batch_size, axis=0), dtype=tf.float32)

    # Step 2: Feature extraction
    feature_pyramids = extract_features(image_pyramids[0], image_pyramids[1], config)

    # Step 3: Flow estimation to obtain residual flow pyramids
    forward_residual_flow_pyramid, backward_residual_flow_pyramid = estimate_flow(
        feature_pyramids[0], feature_pyramids[1], config, time
    )

    # Step 4: Flow pyramid synthesis testing using the synthesized flow pyramids
    forward_flow_pyramid, backward_flow_pyramid = test_flow_pyramid_synthesis(forward_residual_flow_pyramid,
                                                                              backward_residual_flow_pyramid, config)

    # Step 5
    backward_flows, forward_flows = adjust_flow_pyramid_with_time_scaling(backward_flow_pyramid,
                                                                          forward_flow_pyramid, time)

    # Verification: Check if the adjusted flows match expected values
    success = True  # Overall test pass flag
    tolerance = 1e-5  # Allowed tolerance for comparison

    for level_idx, (backward_level, forward_level) in enumerate(zip(backward_flows, forward_flows)):
        for time_step_idx, (adjusted_backward, adjusted_forward) in enumerate(zip(backward_level, forward_level)):
            # Calculate expected values
            expected_backward = backward_flow_pyramid[level_idx][time_step_idx] * time[:, time_step_idx][:, None, None,
                                                                                  None]
            expected_forward = forward_flow_pyramid[level_idx][time_step_idx] * (
                    1 - time[:, time_step_idx][:, None, None, None])

            # Check if results are within tolerance
            if not (tf.reduce_all(tf.abs(adjusted_backward - expected_backward) < tolerance) and
                    tf.reduce_all(tf.abs(adjusted_forward - expected_forward) < tolerance)):
                success = False
                print(f"Mismatch found at level {level_idx}, time step {time_step_idx}")
                print("Adjusted Backward Flow:", adjusted_backward.numpy())
                print("Expected Backward Flow:", expected_backward.numpy())
                print("Adjusted Forward Flow:", adjusted_forward.numpy())
                print("Expected Forward Flow:", expected_forward.numpy())

    # Step 6
    pyramids_to_warp = img_feature_concatenate_test(image_pyramids, feature_pyramids, config)

    # Step 7. warp feature and images using flow.
    forward_warped_pyramid, backward_warped_pyramid = warp_pyramid_with_pyrmids_to_warp_and_flows(
        pyramids_to_warp, forward_flows, backward_flows
    )

    # Step 8.
    aligned_pyramid = util.concatenate_pyramids_with_time(forward_warped_pyramid,
                                                backward_warped_pyramid)
    aligned_pyramid = util.concatenate_pyramids_with_time(aligned_pyramid, backward_flows)
    aligned_pyramid = util.concatenate_pyramids_with_time(aligned_pyramid, forward_flows)
    print("\n concatenate pyramids passed. ")

    # Step 9: Fusion
    print("\nStarting Fusion for multiple time steps")

    # Initialize Fusion layer
    fuse = fusion.Fusion('fusion', config)

    # Call Fusion layer on aligned_pyramid for multiple intermediate frames
    predictions = fuse(aligned_pyramid)

    # Print output structure
    print("\nFusion output structure:")
    for time_step_idx, prediction in enumerate(predictions):
        print(f"  Time step {time_step_idx} - Predicted image shape: {prediction.shape}")

    # Display and save RGB image for each time step
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    for time_step_idx, prediction in enumerate(predictions):
        output_color = prediction[..., :3]  # Extract RGB channels
        output_color = tf.clip_by_value(output_color, 0, 1)  # Clip values to [0, 1] range if necessary

        # Convert tensor to numpy array
        img_array = (output_color.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array[0])  # Select first image in batch

        # Save image
        img_save_path = os.path.join(output_dir, f"frame_t{time_step_idx}.png")
        img.save(img_save_path)
        print(f"Saved frame for time step {time_step_idx} to {img_save_path}")

        # Display image
        plt.figure()
        plt.imshow(img)
        plt.title(f"Intermediate Frame at Time Step {time_step_idx}")
        plt.axis('off')
        plt.show()

    print("\nFusion for multiple time steps completed.")

    # Final result output
    if success:
        print("All tests passed. The outputs match the expected scaled values.")
    else:
        print("Some tests failed. See details above.")


# Specify paths to the two image files
file_path1 = 'frame0.png'
file_path2 = 'frame19.png'

# Run the full test function with actual images
test_feature_extractor_and_flow_estimator(file_path1, file_path2)
