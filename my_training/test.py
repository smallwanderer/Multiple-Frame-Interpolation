import tensorflow as tf
from typing import Dict, Text, Tuple, Callable
from train_lib import _distributed_train_step, _concat_tensors
from model_lib import create_mock_model  # Ensure this function is available
from losses.losses import create_mock_loss_functions  # Ensure compatibility with loss functions


# Setup function to create mock data batches
def create_mock_batch():
    x0 = tf.random.uniform(shape=[1, 64, 64, 3], dtype=tf.float32)
    x1 = tf.random.uniform(shape=[1, 64, 64, 3], dtype=tf.float32)
    y = [tf.random.uniform(shape=[1, 64, 64, 3], dtype=tf.float32) for _ in range(8)]
    return {'x0': x0, 'x1': x1, 'y': y}


# Set up a distribution strategy for testing purposes
strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

with strategy.scope():
    # Initialize model, optimizer, and data batch within the strategy scope
    mock_model = create_mock_model()
    mock_batch = create_mock_batch()
    mock_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Perform distributed training step
    distributed_step_output = _distributed_train_step(
        strategy=strategy,
        batch=mock_batch,
        model=mock_model,
        loss_functions=create_mock_loss_functions(),
        optimizer=mock_optimizer,
        iterations=1
    )

    # Display the output for verification
    print("Distributed Step Output:")
    print("Loss:", distributed_step_output['loss'].numpy())

    # Print summaries for image tensors across time steps
    for key, value in distributed_step_output['image_summaries'].items():
        if isinstance(value, list):
            for i, v in enumerate(value):
                print(f"{key}[{i}]: shape = {v.shape}")
        else:
            print(f"{key}: shape = {value.shape}")

    # Check and print each intermediate time step output for both y and prediction
    for i in range(8):
        print(f"y[{i}]:", distributed_step_output['image_summaries'].get(f'training/y/{i}', "Not found"))
        print(f"pred_y[{i}]:", distributed_step_output['image_summaries'].get(f'training/image_t{i}', "Not found"))
