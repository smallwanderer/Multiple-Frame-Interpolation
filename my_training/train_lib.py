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
r"""Training library for frame interpolation using distributed strategy."""
import functools
from typing import Any, Callable, Dict, Text, Tuple

from absl import logging
import tensorflow as tf


def _concat_tensors(tensors: tf.Tensor) -> tf.Tensor:
    """Concat tensors of the different replicas."""
    return tf.concat(tf.nest.flatten(tensors, expand_composites=True), axis=0)


@tf.function
def _distributed_train_step(strategy: tf.distribute.Strategy,
                            batch: Dict[Text, tf.Tensor], model: tf.keras.Model,
                            loss_functions: Dict[Text,
                                                 Tuple[Callable[..., tf.Tensor],
                                                       Callable[...,
                                                                tf.Tensor]]],
                            optimizer: tf.keras.optimizers.Optimizer,
                            iterations: int) -> Dict[Text, Any]:
    """Distributed training step."""

    def _train_step(batch: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
        """Train for one step with losses for each intermediate frame."""
        with tf.GradientTape() as tape:
            # Make predictions with the model
            predictions = model({'x0': batch['x0'], 'x1': batch['x1']}, training=True)

            # Initialize a list to store individual frame losses
            frame_losses = []

            # Unstack the frames in 'y' along the batch dimension
            target_frames = tf.unstack(batch['y'], axis=1)  # Unstacking along the frame dimension

            # Iterate over intermediate frames in predictions and target frames
            for idx, target_frame in enumerate(target_frames):
                # Construct the expected key name for the predicted frame (e.g., 'image_t0', 'image_t1', ...)
                pred_key = f'image_t{idx}'

                # Ensure that the prediction key exists
                if pred_key in predictions:
                    predicted_frame = predictions[pred_key]

                    # Calculate the loss for the current frame and apply the weighting function
                    for (loss_fn, loss_weight_fn) in loss_functions.values():
                        frame_loss = loss_fn({'y': target_frame}, {'image': predicted_frame})
                        weighted_frame_loss = frame_loss * loss_weight_fn(iterations)

                        # Append the weighted loss to the list
                        frame_losses.append(weighted_frame_loss)
                else:
                    raise KeyError(f"Expected prediction key '{pred_key}' not found in predictions.")

            # Aggregate the total loss by summing individual frame losses
            total_loss = tf.add_n(frame_losses)

        # Compute gradients and apply them
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Prepare all data to return, including loss, batch, and predictions
        all_data = {'loss': total_loss}
        all_data.update(batch)
        all_data.update(predictions)
        return all_data

    # 분산 학습 실행
    step_outputs = strategy.run(_train_step, args=(batch,))

    # 각 장치의 결과를 결합
    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, step_outputs['loss'], axis=None)

    x0 = _concat_tensors(step_outputs['x0'])
    x1 = _concat_tensors(step_outputs['x1'])
    y_frames = tf.unstack(step_outputs['y'], axis=1)
    y = [_concat_tensors(frame) for frame in y_frames]
    pred_y = [_concat_tensors(step_outputs[f'image_t{i}']) for i in range(4)]

    # 스칼라 요약
    scalar_summaries = {'training_loss': loss}
    for key, scalar_value in scalar_summaries.items():
        tf.summary.scalar(key, scalar_value, step=iterations)

    # 이미지 요약
    image_summaries = {
        'x0': x0,
        'x1': x1,
        'y': y,  # 여러 프레임을 가진 y
        'pred_y': pred_y  # 여러 프레임을 가진 pred_y
    }

    # 보조 이미지 요약 (필요 시)
    extra_images = {
        'importance0', 'importance1', 'x0_warped', 'x1_warped', 'fg_image',
        'bg_image', 'fg_alpha', 'x1_unfiltered_warped'
    }
    for image in extra_images:
        if image in step_outputs:
            image_summaries[image] = _concat_tensors(step_outputs[image])

    return {
        'loss': loss,
        'scalar_summaries': scalar_summaries,
        'image_summaries': {
            f'training/{name}': value for name, value in image_summaries.items()
        }
    }


def _summary_writer(summaries_dict: Dict[Text, Any]) -> None:
    """Adds scalar and image summaries."""
    # Adds scalar summaries.
    for key, scalars in summaries_dict['scalar_summaries'].items():
        tf.summary.scalar(key, scalars)

    # Adds image summaries.
    for key, images in summaries_dict['image_summaries'].items():
        # 'images'가 리스트인지 확인
        if isinstance(images, list):
            for i, image in enumerate(images):
                frame_key = f"{key}_frame_{i}"
                if isinstance(image, tf.Tensor) and image.shape.rank == 5:
                    # [batch, time_steps, height, width, channels]을 [batch * time_steps, height, width, channels]로 변환
                    reshaped_images = tf.reshape(image, [-1, image.shape[2], image.shape[3], image.shape[4]])
                    tf.summary.image(frame_key, tf.clip_by_value(reshaped_images, 0.0, 1.0))
                elif isinstance(image, tf.Tensor):
                    tf.summary.image(frame_key, tf.clip_by_value(image, 0.0, 1.0))
                tf.summary.histogram(frame_key + '_h', image)
        else:
            # 단일 텐서일 경우 직접 처리
            if isinstance(images, tf.Tensor) and images.shape.rank == 5:
                # [batch, time_steps, height, width, channels]을 시간 차원에 대해 평균화
                averaged_images = tf.reduce_mean(images, axis=1)  # [batch, height, width, channels]
                tf.summary.image(key, tf.clip_by_value(averaged_images, 0.0, 1.0))
            elif isinstance(images, tf.Tensor):
                tf.summary.image(key, tf.clip_by_value(images, 0.0, 1.0))
            tf.summary.histogram(key + '_h', images)



def train_loop(
        strategy: tf.distribute.Strategy,
        train_set: tf.data.Dataset,
        create_model_fn: Callable[..., tf.keras.Model],
        create_losses_fn: Callable[..., Dict[str, Tuple[Callable[..., tf.Tensor],
                                                        Callable[..., tf.Tensor]]]],
        create_optimizer_fn: Callable[..., tf.keras.optimizers.Optimizer],
        distributed_train_step_fn: Callable[[
                                                tf.distribute.Strategy, Dict[str, tf.Tensor], tf.keras.Model, Dict[
                str,
                Tuple[Callable[..., tf.Tensor],
                      Callable[..., tf.Tensor]]], tf.keras.optimizers.Optimizer, int
                                            ], Dict[str, Any]],
        eval_loop_fn: Callable[..., None],
        create_metrics_fn: Callable[..., Dict[str, tf.keras.metrics.Metric]],
        eval_folder: Dict[str, Any],
        eval_datasets: Dict[str, tf.data.Dataset],
        summary_writer_fn: Callable[[Dict[str, Any]], None],
        train_folder: str,
        saved_model_folder: str,
        num_iterations: int,
        save_summaries_frequency: int = 500,
        save_checkpoint_frequency: int = 500,
        checkpoint_max_to_keep: int = 10,
        checkpoint_save_every_n_hours: float = 2.,
        timing_frequency: int = 100,
        logging_frequency: int = 10):
    """A Tensorflow 2 eager mode training loop.

    Args:
      strategy: A Tensorflow distributed strategy.
      train_set: A tf.data.Dataset to loop through for training.
      create_model_fn: A callable that returns a tf.keras.Model.
      create_losses_fn: A callable that returns a tf.keras.losses.Loss.
      create_optimizer_fn: A callable that returns a
        tf.keras.optimizers.Optimizer.
      distributed_train_step_fn: A callable that takes a distribution strategy, a
        Dict[Text, tf.Tensor] holding the batch of training data, a
        tf.keras.Model, a tf.keras.losses.Loss, a tf.keras.optimizers.Optimizer,
        iteartion number to sample a weight value to loos functions,
        and returns a dictionary to be passed to the summary_writer_fn.
      eval_loop_fn: Eval loop function.
      create_metrics_fn: create_metric_fn.
      eval_folder: A path to where the summary event files and checkpoints will be
        saved.
      eval_datasets: A dictionary of evalution tf.data.Dataset to loop through for
        evaluation.
      summary_writer_fn: A callable that takes the output of
        distributed_train_step_fn and writes summaries to be visualized in
        TensorBoard.
      train_folder: A path to where the summaries event files and checkpoints
        will be saved.
      saved_model_folder: A path to where the saved models are stored.
      num_iterations: An integer, the number of iterations to train for.
      save_summaries_frequency: The iteration frequency with which summaries are
        saved.
      save_checkpoint_frequency: The iteration frequency with which model
        checkpoints are saved.
      checkpoint_max_to_keep: The maximum number of checkpoints to keep.
      checkpoint_save_every_n_hours: The frequency in hours to keep checkpoints.
      timing_frequency: The iteration frequency with which to log timing.
      logging_frequency: How often to output with logging.info().
    """
    logging.info('Creating training tensorboard summaries ...')
    summary_writer = tf.summary.create_file_writer(train_folder)

    if eval_datasets is not None:
        logging.info('Creating eval tensorboard summaries ...')
        eval_summary_writer = tf.summary.create_file_writer(eval_folder)

    train_set = strategy.experimental_distribute_dataset(train_set)
    with strategy.scope():
        logging.info('Building model ...')
        model = create_model_fn()
        loss_functions = create_losses_fn()
        optimizer = create_optimizer_fn()
        if eval_datasets is not None:
            metrics = create_metrics_fn()

    logging.info('Creating checkpoint ...')
    checkpoint = tf.train.Checkpoint(
        model=model,
        optimizer=optimizer,
        step=optimizer.iterations,
        epoch=tf.Variable(0, dtype=tf.int64, trainable=False),
        training_finished=tf.Variable(False, dtype=tf.bool, trainable=False))

    logging.info('Restoring old model (if exists) ...')
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=train_folder,
        max_to_keep=checkpoint_max_to_keep,
        keep_checkpoint_every_n_hours=checkpoint_save_every_n_hours)

    with strategy.scope():
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)

    logging.info('Creating Timer ...')
    timer = tf.estimator.SecondOrStepTimer(every_steps=timing_frequency)
    timer.update_last_triggered_step(optimizer.iterations.numpy())

    logging.info('Training on devices: %s.', [
        el.name.split('/physical_device:')[-1]
        for el in tf.config.get_visible_devices()
    ])

    # Re-assign training_finished=False, in case we restored a checkpoint.
    checkpoint.training_finished.assign(False)
    while optimizer.iterations.numpy() < num_iterations:
        for i_batch, batch in enumerate(train_set):
            summary_writer.set_as_default()
            iterations = optimizer.iterations.numpy()

            if iterations % logging_frequency == 0:
                # Log epoch, total iterations and batch index.
                logging.info('epoch %d; iterations %d; i_batch %d',
                             checkpoint.epoch.numpy(), iterations,
                             i_batch)

            # Break if the number of iterations exceeds the max.
            if iterations >= num_iterations:
                break

            # Compute distributed step outputs.
            distributed_step_outputs = distributed_train_step_fn(
                strategy, batch, model, loss_functions, optimizer, iterations)

            # Save checkpoint, and optionally run the eval loops.
            if iterations % save_checkpoint_frequency == 0:
                checkpoint_manager.save(checkpoint_number=iterations)
                if eval_datasets is not None:
                    eval_loop_fn(
                        strategy=strategy,
                        eval_base_folder=eval_folder,
                        model=model,
                        metrics=metrics,
                        datasets=eval_datasets,
                        summary_writer=eval_summary_writer,
                        checkpoint_step=iterations)

            # Write summaries.
            if iterations % save_summaries_frequency == 0:
                tf.summary.experimental.set_step(step=iterations)
                summary_writer_fn(distributed_step_outputs)
                tf.summary.scalar('learning_rate',
                                  optimizer.learning_rate(iterations).numpy())

            # Log steps/sec.
            if timer.should_trigger_for_step(iterations):
                elapsed_time, elapsed_steps = timer.update_last_triggered_step(
                    iterations)
                if elapsed_time is not None:
                    steps_per_second = elapsed_steps / elapsed_time
                    tf.summary.scalar(
                        'steps/sec', steps_per_second, step=optimizer.iterations)

        # Increment epoch.
        checkpoint.epoch.assign_add(1)

        # Log the train loss after completing an epoch
        train_loss = distributed_step_outputs['scalar_summaries']['training_loss']
        logging.info('End of epoch %d; Training loss: %.4f', checkpoint.epoch.numpy(), train_loss.numpy())

    # Assign training_finished variable to True after training is finished and
    # save the last checkpoint.
    checkpoint.training_finished.assign(True)
    checkpoint_manager.save(checkpoint_number=optimizer.iterations.numpy())

    # Generate a saved model.
    model.save(saved_model_folder)


def train(strategy: tf.distribute.Strategy, train_folder: str,
          saved_model_folder: str, n_iterations: int,
          create_model_fn: Callable[..., tf.keras.Model],
          create_losses_fn: Callable[..., Dict[str,
                                               Tuple[Callable[..., tf.Tensor],
                                                     Callable[...,
                                                              tf.Tensor]]]],
          create_metrics_fn: Callable[..., Dict[str, tf.keras.metrics.Metric]],
          dataset: tf.data.Dataset,
          learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
          eval_loop_fn: Callable[..., None],
          eval_folder: str,
          eval_datasets: Dict[str, tf.data.Dataset]):
    """Training function that is strategy agnostic.

    Args:
      strategy: A Tensorflow distributed strategy.
      train_folder: A path to where the summaries event files and checkpoints
        will be saved.
      saved_model_folder: A path to where the saved models are stored.
      n_iterations: An integer, the number of iterations to train for.
      create_model_fn: A callable that returns tf.keras.Model.
      create_losses_fn: A callable that returns the losses.
      create_metrics_fn: A function that returns the metrics dictionary.
      dataset: The tensorflow dataset object.
      learning_rate: Keras learning rate schedule object.
      eval_loop_fn: eval loop function.
      eval_folder: A path to where eval summaries event files and checkpoints
        will be saved.
      eval_datasets: The tensorflow evaluation dataset objects.
    """
    train_loop(
        strategy=strategy,
        train_set=dataset,
        create_model_fn=create_model_fn,
        create_losses_fn=create_losses_fn,
        create_optimizer_fn=functools.partial(
            tf.keras.optimizers.Adam, learning_rate=learning_rate),
        distributed_train_step_fn=_distributed_train_step,
        eval_loop_fn=eval_loop_fn,
        create_metrics_fn=create_metrics_fn,
        eval_folder=eval_folder,
        eval_datasets=eval_datasets,
        summary_writer_fn=_summary_writer,
        train_folder=train_folder,
        saved_model_folder=saved_model_folder,
        num_iterations=n_iterations,
        save_summaries_frequency=3000,
        save_checkpoint_frequency=3000)


def get_strategy(mode) -> tf.distribute.Strategy:
    """Creates a distributed strategy."""
    strategy = None
    if mode == 'cpu':
        strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
    elif mode == 'gpu':
        strategy = tf.distribute.MirroredStrategy()
    else:
        raise ValueError('Unsupported distributed mode.')
    return strategy
