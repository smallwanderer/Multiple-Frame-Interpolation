from typing import Dict, Mapping, Text
from absl import logging
import tensorflow as tf


def _collect_tensors(tensors: tf.Tensor) -> tf.Tensor:
    """Collect tensors of the different replicas into a list."""
    return tf.nest.flatten(tensors, expand_composites=True)


def summerize_images_tensors(images: Dict[Text, tf.Tensor], target_shape=(64, 64)) -> tf.Tensor:
    """Summarizes a list of image tensors into a single concatenated tensor along width."""
    # Ensure each image tensor is resized to the target shape for consistent concatenation
    resized_images = [tf.image.resize(img, target_shape) for img in images.values()]

    # Concatenate along the width (axis=2)
    return tf.concat(resized_images, axis=2)

@tf.function
def _distributed_eval_step(strategy, batch, model, metrics, checkpoint_step):
    def _eval_step(batch, model, metrics):
        # Ground truth frames (실제 중간 프레임들)
        target_frames = tf.unstack(batch['y'], axis=1)  # Unstack along frame dimension

        # Model predictions
        predictions = model({'x0': batch['x0'], 'x1': batch['x1']}, training=False)

        # 각 프레임의 손실을 계산하고 누적
        total_loss = 0
        for idx, target_frame in enumerate(target_frames):
            pred_key = f'image_t{idx}'  # 예측 프레임 키 예: image_t0, image_t1 등
            if pred_key in predictions:
                predicted_frame = predictions[pred_key]
                single_gt = {'y': target_frame}
                single_pred = {'image': predicted_frame}

                # 각 메트릭 업데이트
                for metric_name, metric in metrics.items():
                    metric.update_state(single_gt, single_pred)

                # 손실 누적
                total_loss += tf.reduce_mean([metric.result() for metric in metrics.values()])
            else:
                raise KeyError(f"Expected prediction key '{pred_key}' not found in predictions.")

        return {'total_loss': total_loss / len(target_frames), 'extra_images': predictions}

    return strategy.run(_eval_step, args=(batch, model, metrics))


def _summarize_image_tensors(combined, prefix, step):
    """Summarizes image tensors into TensorBoard."""
    for name, image in combined.items():
        if isinstance(image, tf.Tensor) and len(image.shape) == 4:
            tf.summary.image(prefix + '/' + name, image, step=step)


def eval_loop(strategy: tf.distribute.Strategy,
              eval_base_folder: str,
              model: tf.keras.Model,
              metrics: Dict[str, tf.keras.metrics.Metric],
              datasets: Mapping[str, tf.data.Dataset],
              summary_writer: tf.summary.SummaryWriter,
              checkpoint_step: int):
    """Eval loop that supports summarizing multiple frame predictions with debugging info."""
    logging.info('Saving eval summaries to: %s...', eval_base_folder)
    summary_writer.set_as_default()

    for dataset_name, dataset in datasets.items():
        for metric in metrics.values():
            metric.reset_states()

        logging.info('Loading %s testing data ...', dataset_name)
        dataset = strategy.experimental_distribute_dataset(dataset)

        logging.info('Evaluating %s ...', dataset_name)
        batch_idx = 0
        max_batches_to_summarize = 10

        for batch in dataset:
            step_outputs = _distributed_eval_step(strategy, batch, model, metrics, checkpoint_step)
            extra_images = step_outputs['extra_images']

            # 개별 이미지에 클리핑 적용
            try:
                extra_images_clipped = {key: tf.clip_by_value(image, 0., 1.) for key, image in extra_images.items()}
            except Exception as e:
                logging.error(f'Error during clipping: {e}')
                logging.info(f'Failed clipping structure: {extra_images}')
                raise  # Re-raise to stop execution and debug further

            # 주기적으로 로그 출력
            if batch_idx % 10 == 0:
                logging.info('Evaluating batch %s', batch_idx)
            batch_idx += 1

            # 요약된 중간 프레임 시각화
            if batch_idx <= max_batches_to_summarize:
                prefix = f'{dataset_name}/eval_{batch_idx}'
                try:
                    combined_images = summerize_images_tensors(extra_images_clipped)  # 클리핑된 이미지를 사용
                    _summarize_image_tensors({'Predicted Frames Summary': combined_images}, prefix,
                                             step=checkpoint_step)
                except Exception as e:
                    logging.error(f'Error during tensor summarization: {e}')
                    logging.info(f'Failed summarization structure: {extra_images_clipped}')
                    raise  # Re-raise to stop execution and debug further

        # 메트릭 요약 기록
        for name, metric in metrics.items():
            tf.summary.scalar(f'{dataset_name}/{name}', metric.result(), step=checkpoint_step)
            logging.info('Step {:2}, {} {}'.format(checkpoint_step, f'{dataset_name}/{name}', metric.result().numpy()))
            metric.reset_states()


def eval_loop(strategy: tf.distribute.Strategy,
              eval_base_folder: str,
              model: tf.keras.Model,
              metrics: Dict[str, tf.keras.metrics.Metric],
              datasets: Mapping[str, tf.data.Dataset],
              summary_writer: tf.summary.SummaryWriter,
              checkpoint_step: int):
    """Eval loop that supports summarizing multiple frame predictions with debugging info."""
    logging.info('Saving eval summaries to: %s...', eval_base_folder)
    summary_writer.set_as_default()

    for dataset_name, dataset in datasets.items():
        for metric in metrics.values():
            metric.reset_states()

        logging.info('Loading %s testing data ...', dataset_name)
        dataset = strategy.experimental_distribute_dataset(dataset)

        logging.info('Evaluating %s ...', dataset_name)
        batch_idx = 0
        max_batches_to_summarize = 10

        for batch in dataset:
            step_outputs = _distributed_eval_step(strategy, batch, model, metrics, checkpoint_step)
            extra_images = step_outputs['extra_images']


            # 클리핑 적용: 중간 프레임 및 워핑된 이미지에만 적용
            extra_images_clipped = {
                key: tf.clip_by_value(image, 0., 1.)
                for key, image in extra_images.items()
                if 'flow' not in key  # flow 관련 키를 제외하여 클리핑 적용
            }

            # 주기적으로 로그 출력
            if batch_idx % 10 == 0:
                logging.info('Evaluating batch %s', batch_idx)
            batch_idx += 1

            # 요약된 중간 프레임 시각화
            if batch_idx <= max_batches_to_summarize:
                prefix = f'{dataset_name}/eval_{batch_idx}'
                try:
                    combined_images = summerize_images_tensors(extra_images_clipped)  # 클리핑된 이미지를 사용
                    _summarize_image_tensors({'Predicted Frames Summary': combined_images}, prefix,
                                             step=checkpoint_step)
                except Exception as e:
                    logging.error(f'Error during tensor summarization: {e}')
                    logging.info(f'Failed summarization structure: {extra_images_clipped}')
                    raise  # Re-raise to stop execution and debug further

        # 메트릭 요약 기록
        for name, metric in metrics.items():
            tf.summary.scalar(f'{dataset_name}/{name}', metric.result(), step=checkpoint_step)
            logging.info('Step {:2}, {} {}'.format(checkpoint_step, f'{dataset_name}/{name}', metric.result().numpy()))
            metric.reset_states()
