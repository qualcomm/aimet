# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


# TODO Need to exclude this file for PyLint checking. We get the following error that needs to be investigated:
# RecursionError: maximum recursion depth exceeded while calling a Python object
# pylint: skip-file

"""Sub-sample data for weight reconstruction for channel pruning feature"""

from typing import Tuple
import numpy as np

from .utils import AimetLogger

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Test)


class InputMatchSearch:
    """Utilities to find a set of input pixels corresponding to an output pixel for weight reconstruction"""

    @staticmethod
    def _check_and_update_pixel_sampled_from_output_data(
        input_data_shape: tuple, layer_attributes: tuple, pixel: tuple
    ):
        """
        Function gets shape, layer, pixel indices (height, width) and first check if given height and width satisfy
         O = (I - F + 2P) / S + 1 formula, and then update them using stride and
        padding values of given layer.

        :param input_data_shape: input data shape (Cin, Hin, Win)
        :param layer_attributes: (kernel_size, stride, padding)
        :param pixel: (height, width)
        :return: new_pixel (height, width)
        """

        # assert if provided shape is not of size (Cin, Hin, Win)
        assert len(input_data_shape) == 3
        # assert if provided pixel is not of size (height, width)
        assert len(pixel) == 2

        input_height = input_data_shape[1]
        input_width = input_data_shape[2]

        height, width = pixel
        kernel_size, stride, padding = layer_attributes

        # check if there exist a match for given pixel (height, width)
        output_height = (input_height - kernel_size[0] + 2 * padding[0]) / stride[0] + 1
        output_width = (input_width - kernel_size[1] + 2 * padding[1]) / stride[1] + 1

        if not 0 <= height <= output_height or not 0 <= width <= output_width:
            raise ValueError(
                "input match can not exist for given height and width indices!"
            )

        # firstly, multiply strides to output data height and width indices
        height = stride[0] * height
        width = stride[1] * width

        # secondly, subtract padding from output data height and width indices
        height -= padding[0]
        width -= padding[1]

        new_pixel = (height, width)
        return new_pixel

    @staticmethod
    def _find_pixel_range_for_rectangle_input_match(
        shape: tuple, layer_attributes: tuple, pixel: tuple
    ):
        """
        Function gets input_data shape, layer, pixel (height, width) from output data and calculates
        corresponding input match height and width index ranges.
        :param shape: input data shape (Cin, Hin, Win)
        :param layer_attributes: (kernel_size, stride, padding)
        :param pixel: (height, width)
        :return: input match height and width indices ranges
        """

        # assert if provided shape is not of size (Cin, Hin, Win)
        assert len(shape) == 3
        # assert if provided pixel is not of size (height, width)
        assert len(pixel) == 2
        # first check whether provided pixel (height, width) is valid and then update
        new_pixel = InputMatchSearch._check_and_update_pixel_sampled_from_output_data(
            shape, layer_attributes, pixel
        )

        height, width = new_pixel

        kernel_size = layer_attributes[0]

        # variables to calculate start and stop indices (height and width) for input match rectangle
        height_start = 0
        width_start = 0
        height_stop = kernel_size[0]
        width_stop = kernel_size[1]

        # Two corner cases occur when there is padding > 0.
        # 1) sampled output data height and width indices become negative.
        if height < 0:
            height_start = -height

        if width < 0:
            width_start = -width

        # 2) sampled output data height and width indices plus kernel size exceeds the input data size.
        if height + kernel_size[0] > shape[1]:
            height_overshoot = height + kernel_size[0] - shape[1]
            height_stop = kernel_size[0] - height_overshoot

        if width + kernel_size[1] > shape[2]:
            width_overshoot = width + kernel_size[1] - shape[2]
            width_stop = kernel_size[1] - width_overshoot

        height_range = (height_start, height_stop)
        width_range = (width_start, width_stop)

        return height_range, width_range

    @staticmethod
    def _find_pixel_range_for_input_data(
        input_data_shape: tuple, layer_attributes: tuple, pixel: tuple
    ):
        """
        Function gets input_data shape, layer_attributes, pixel (height, width) and calculates
        corresponding input_data height and width index ranges.

        :param input_data_shape: input data shape (Cin, Hin, Win)
        :param layer_attributes: (kernel_size, stride, padding)
        :param pixel: pixel (height, width) sampled randomly from output data
        :return: height and width pixel ranges of input data
        """

        # assert if provided shape is not of size (Cin, Hin, Win)
        assert len(input_data_shape) == 3
        # assert if provided pixel is not of size (height, width)
        assert len(pixel) == 2

        # first check whether provided pixel (height, width) is valid and then update
        new_pixel = InputMatchSearch._check_and_update_pixel_sampled_from_output_data(
            input_data_shape, layer_attributes, pixel
        )

        height, width = new_pixel
        kernel_size = layer_attributes[0]

        height_start = max(0, height)
        width_start = max(0, width)
        height_stop = height + kernel_size[0]
        width_stop = width + kernel_size[1]

        height_range = (height_start, height_stop)
        width_range = (width_start, width_stop)

        return height_range, width_range

    @staticmethod
    def _find_input_match(
        input_data: np.ndarray,
        layer_attributes: tuple,
        pixel_range_for_data: tuple,
        pixel_range_for_match: tuple,
    ):
        """
        Function gets input_data, layer_attributes, pixel ranges for input data and input match, and return
        the input match.
        :param input_data (np.ndarray): input data
        :param kernel_size (tuple): kernel size
        :param pixel_range_for_data (tuple): (height range, width range)
        :param pixel_range_for_match (tuple): (height range, width range)
        :return: input match of size (Cin, k_h, k_w)
        """

        assert len(pixel_range_for_data) == 2
        assert len(pixel_range_for_match) == 2

        kernel_size = layer_attributes[0]

        # create input match of size [Cin, k_h, k_w] filled with zeros
        input_match = np.zeros(
            [input_data.shape[0], kernel_size[0], kernel_size[1]],
            dtype=input_data.dtype,
        )

        height_range = pixel_range_for_data[0]
        width_range = pixel_range_for_data[1]

        # make sure start index is always equal or lesser than end index
        assert height_range[0] <= height_range[1]
        assert width_range[0] <= width_range[1]

        # extract the appropriate data using height and width range
        extracted_data = input_data[
            :, height_range[0] : height_range[1], width_range[0] : width_range[1]
        ]

        height_range = pixel_range_for_match[0]
        width_range = pixel_range_for_match[1]

        # make sure start index is always equal or lesser than end index
        assert height_range[0] <= height_range[1]
        assert width_range[0] <= width_range[1]

        # set extracted data appropriately in input match
        input_match[
            :, height_range[0] : height_range[1], width_range[0] : width_range[1]
        ] = extracted_data

        return input_match

    @classmethod
    def _find_input_match_for_output_pixel(
        cls, input_data: np.ndarray, layer_attributes: tuple, pixel: tuple
    ):
        """
        Function finds the input match that generated the output of a conv2d layer at that specified output pixel.
        It looks at one output pixel (height, width), and finds which input pixels (input match) corresponded to that,
        for different kernel sizes, strides and padding settings. The size of input match is [Cin, k_h, k_w]

        :param input_data: input data (Nic, act_h, act_w)
        :param layer_attributes: (kernel_size, stride, padding)
        :param pixel: (height, width)
        :return: input match of size (Cin, k_h, k_w)
        """

        # calculate pixel (height and width) ranges for input data
        pixel_range_for_data = cls._find_pixel_range_for_input_data(
            input_data.shape, layer_attributes, pixel
        )

        # calculate pixel (height and width) ranges for input match rectangle (place holder)
        pixel_range_for_match = cls._find_pixel_range_for_rectangle_input_match(
            input_data.shape, layer_attributes, pixel
        )

        # calculate input match for corresponding output_data_pixel (height, width)
        input_match = cls._find_input_match(
            input_data, layer_attributes, pixel_range_for_data, pixel_range_for_match
        )

        return input_match

    @classmethod
    def _determine_output_pixel_height_width_range_for_random_selection(
        cls, layer_attributes: tuple, out_shape: tuple
    ) -> Tuple[tuple, tuple]:
        """
        Function returns height range and width range based on the Kernel size and Padding size.
        If the Kernel size is bigger than or equal to the Padding size, this function returns the height range and
        width range that includes all the pixels. If the Padding is bigger than Kernel size, this function returns a
        smaller height and and width range. In this scenario, randomly picking an output pixel close to the border
        of the output is not going to contribute to the reconstruction process since the padded values are all 0s.

        :param layer_attributes: (kernel_size, stride, padding)
        :param out_shape: shape of the output (Num Samples, Num Output channels, activation height, activation width)
        :return: tuple(height start, height end), tuple(width start, width end)
        """
        kernel, _, padding = layer_attributes
        kernel_height, kernel_width = kernel
        padding_height, padding_width = padding

        # output data shape: Number of Samples, Number of Output channels, activation height, activation width
        _, _, activation_height, activation_width = out_shape

        if kernel_height >= padding_height:
            height_range = (0, activation_height)
        else:
            # Padding is bigger than Kernel size.
            height_range = (padding_height, (activation_height - padding_height))

        if kernel_width >= padding_width:
            width_range = (0, activation_width)
        else:
            # Padding is bigger than Kernel size.
            width_range = (padding_width, (activation_width - padding_width))

        return height_range, width_range

    @classmethod
    def subsample_data(
        cls,
        layer_attributes: tuple,
        input_data: np.ndarray,
        output_data: np.ndarray,
        samples_per_image: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function takes layer_attributes, input data (collected from pruned model) and output data (collected from
        original layer) and returns sub sampled input and their corresponding sub sampled output

        input_data and output_data must be in channels_first format (Common format - Ns, Nin/Noc, act_h, act_w)

        :param layer_attributes: (kernel_size, stride, padding)
        :param input_data: input data (Ns, Nic, act_h, act_w)
        :param output_data: output data (Ns, Noc, act_h, act_w)
        :param samples_per_image: number of samples per image (default : 10)
        :return:
               sampled input (Nb * Ns, Nic, kh, kw), sampled output (Nb * Ns, Noc)

        # Ns = samples_per_image
        # Nb = total images (batch size of data loader * number of batches)
        # Nic, Noc = input channels and output channels for layer
        # kh, kw = kernel weight dimensions (height, width)

        # sampled input shape should be [Nb * Ns, Nic, kh, kw]
        # sampled output shape should be [Nb * Ns, Noc]
        """

        assert input_data.shape[0] == output_data.shape[0]

        batch_size = output_data.shape[0]

        sampled_input = []
        sampled_output = []

        height_range, width_range = (
            cls._determine_output_pixel_height_width_range_for_random_selection(
                layer_attributes=layer_attributes, out_shape=output_data.shape
            )
        )

        # TODO: The PyLint RecursionError that needs to be investigated occurs from this code onwards
        # iterate over all images in one batch
        for image_index in range(batch_size):
            # randomly pick samples per image for height and width dimension
            heights = np.random.choice(
                range(*height_range), size=[samples_per_image], replace=True
            )
            widths = np.random.choice(
                range(*width_range), size=[samples_per_image], replace=True
            )

            # iterate over all samples
            for sample in range(samples_per_image):
                # output pixel
                output_pixel = (heights[sample], widths[sample])

                # find input match for given output pixel
                input_match = cls._find_input_match_for_output_pixel(
                    input_data[image_index], layer_attributes, output_pixel
                )
                # find output match for given output pixel
                output_match = output_data[
                    image_index, :, heights[sample], widths[sample]
                ]

                sampled_input.append(input_match)
                sampled_output.append(output_match)

        sampled_input = np.array(sampled_input)
        sampled_output = np.vstack(sampled_output)

        # shape of sampled input should be [Nb * Ns, Nic, kh, kw]
        assert len(sampled_input.shape) == 4
        # shape of sampled output should be [Nb * Ns, Noc]
        assert len(sampled_output.shape) == 2

        return sampled_input, sampled_output
