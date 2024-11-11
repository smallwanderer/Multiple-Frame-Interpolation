import os
from my_training import augmentation_lib, data_lib, eval_lib, metrics_lib, model_lib, train_lib
from absl import app, flags, logging
import gin.tf
from losses import losses
import tensorflow as tf

# Reduce tensorflow logs to ERRORs only.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf  # pylint: disable=g-import-not-at-top

tf.get_logger().setLevel('ERROR')

_GIN_CONFIG = flags.DEFINE_string('gin_config', None, 'Gin config file.')
_LABEL = flags.DEFINE_string('label', 'run0',
                             'Descriptive label for this run.')
_BASE_FOLDER = flags.DEFINE_string('base_folder', None,
                                   'Path to checkpoints/summaries.')
_MODE = flags.DEFINE_enum('mode', 'gpu', ['cpu', 'gpu'],
                          'Distributed strategy approach.')


@gin.configurable('training')
class TrainingOptions(object):
    """Training-related options."""

    def __init__(self, learning_rate: float, learning_rate_decay_steps: int,
                 learning_rate_decay_rate: int, learning_rate_staircase: int,
                 num_steps: int):
        self.learning_rate = learning_rate
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.learning_rate_staircase = learning_rate_staircase
        self.num_steps = num_steps


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    gin_config_path = flags.FLAGS.gin_config
    if not os.path.exists(gin_config_path):
        raise FileNotFoundError(f"Gin config file not found at {gin_config_path}")
    else:
        logging.info(f"Gin config file found at {gin_config_path}")

    # Read and parse the gin config file
    with open(gin_config_path, 'r', encoding='utf-8') as f:
        gin_config_content = f.read()
    try:
        gin.parse_config(gin_config_content)
        logging.info("Gin configuration successfully loaded.")
    except Exception as e:
        raise ValueError(f"Failed to load gin configuration: {e}")

    # Set up output directory
    output_dir = os.path.join(_BASE_FOLDER.value, _LABEL.value)
    logging.info('Creating output_dir @ %s ...', output_dir)

    # Check if output directory can be created
    if not os.path.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory could not be created at {output_dir}")
    else:
        logging.info(f"Output directory successfully created at {output_dir}")

    # Copy gin config file to output directory for record
    copied_gin_config_path = os.path.join(output_dir, 'config.gin')
    tf.io.gfile.copy(gin_config_path, copied_gin_config_path, overwrite=True)
    if not os.path.exists(copied_gin_config_path):
        raise FileNotFoundError(f"Failed to copy gin config file to {copied_gin_config_path}")
    else:
        logging.info(f"Gin config file successfully copied to {copied_gin_config_path}")

    # Initialize training options
    try:
        training_options = TrainingOptions()  # pylint: disable=no-value-for-parameter
    except Exception as e:
        raise ValueError(f"Failed to initialize training options: {e}")

    # Set up learning rate schedule
    try:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            training_options.learning_rate,
            training_options.learning_rate_decay_steps,
            training_options.learning_rate_decay_rate,
            training_options.learning_rate_staircase,
            name='learning_rate')
    except Exception as e:
        raise ValueError(f"Failed to set up learning rate schedule: {e}")

    # Data augmentation and folders
    augmentation_fns = augmentation_lib.data_augmentations()
    saved_model_folder = os.path.join(_BASE_FOLDER.value, _LABEL.value, 'saved_model')
    train_folder = os.path.join(_BASE_FOLDER.value, _LABEL.value, 'train')
    eval_folder = os.path.join(_BASE_FOLDER.value, _LABEL.value, 'eval')

    # Check that folders can be created/accessed
    for folder in [saved_model_folder, train_folder, eval_folder]:
        if not os.path.exists(folder):
            tf.io.gfile.makedirs(folder)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Failed to create/access folder at {folder}")
        else:
            logging.info(f"Folder successfully created/accessed at {folder}")

    # Simple sanity checks for training options
    assert training_options.num_steps > 0, "Number of steps must be greater than 0"
    assert training_options.learning_rate > 0, "Learning rate must be positive"

    # Start training
    try:
        train_lib.train(
            strategy=train_lib.get_strategy(_MODE.value),
            train_folder=train_folder,
            saved_model_folder=saved_model_folder,
            n_iterations=training_options.num_steps,
            create_model_fn=model_lib.create_model,
            create_losses_fn=losses.training_losses,
            create_metrics_fn=metrics_lib.create_metrics_fn,
            dataset=data_lib.create_training_dataset(augmentation_fns=augmentation_fns),
            learning_rate=learning_rate,
            eval_loop_fn=eval_lib.eval_loop,
            eval_folder=eval_folder,
            eval_datasets=data_lib.create_eval_datasets() or None)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")


if __name__ == '__main__':
    app.run(main)
