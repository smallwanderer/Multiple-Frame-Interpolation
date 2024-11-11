from typing import List
import tensorflow as tf


def _relu(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.leaky_relu(x, alpha=0.2)


_NUMBER_OF_COLOR_CHANNELS = 3


class Fusion(tf.keras.layers.Layer):
    """The decoder for generating intermediate frames at multiple time steps."""

    def __init__(self, name: str, config):
        super().__init__(name=name)

        # Each item 'convs[i]' will contain the list of convolutions to be applied
        # for pyramid level 'i'.
        self.convs: List[List[tf.keras.layers.Layer]] = []

        # Store the levels, so we can verify right number of levels in call().
        self.levels = config.fusion_pyramid_levels

        # Create the convolutions for each level
        for i in range(config.fusion_pyramid_levels - 1):
            m = config.specialized_levels
            k = config.filters
            num_filters = (k << i) if i < m else (k << m)

            convs: List[tf.keras.layers.Layer] = []
            convs.append(
                tf.keras.layers.Conv2D(
                    filters=num_filters, kernel_size=[2, 2], padding='same'))
            convs.append(
                tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=[3, 3],
                    padding='same',
                    activation=_relu))
            convs.append(
                tf.keras.layers.Conv2D(
                    filters=num_filters,
                    kernel_size=[3, 3],
                    padding='same',
                    activation=_relu))
            self.convs.append(convs)

        # The final convolution that outputs RGB for each intermediate frame:
        self.output_conv = tf.keras.layers.Conv2D(
            filters=_NUMBER_OF_COLOR_CHANNELS, kernel_size=1)

    def call(self, pyramid_list: List[List[tf.Tensor]]) -> List[tf.Tensor]:
        """Runs the fusion module for multiple time steps.

        Args:
          pyramid_list: List of feature pyramids for each time step. Each item is a
            feature pyramid list for a single time step, where each tensor is in
            (B x H x W x C) format, with the finest level tensor first.

        Returns:
          A list of RGB images for each time step.
        Raises:
          ValueError, if len(pyramid) != config.fusion_pyramid_levels as provided in
            the constructor.
        """
        predictions = []

        for pyramid in pyramid_list:
            if len(pyramid) != self.levels:
                raise ValueError(
                    'Fusion called with different number of pyramid levels '
                    f'{len(pyramid)} than it was configured for, {self.levels}.')

            # Process each pyramid level (same as single frame generation)
            net = pyramid[-1]  # Start from the coarsest level

            for i in reversed(range(0, self.levels - 1)):
                # Resize and concatenate with finer level
                level_size = tf.shape(pyramid[i])[1:3]
                net = tf.image.resize(net, level_size,
                                      tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                net = self.convs[i][0](net)
                net = tf.concat([pyramid[i], net], axis=-1)
                net = self.convs[i][1](net)
                net = self.convs[i][2](net)

            # Final RGB output for the current time step
            net = self.output_conv(net)
            predictions.append(net)

        return predictions
