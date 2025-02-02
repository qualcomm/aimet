# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021-2022, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" This file contains unit tests for testing cross layer scaling feature of CLE """
import pytest
import numpy as np
import tensorflow as tf

from aimet_tensorflow.examples.test_models import keras_functional_conv_net
from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.keras.cross_layer_equalization import CrossLayerScaling, ClsSetInfo
from aimet_tensorflow.keras.graphsearchtuils import GraphSearchUtils
from aimet_tensorflow.keras.utils.weight_tensor_utils import WeightTensorUtils


class TestTrainingExtensionsCrossLayerScaling:
    """ Test methods for Cross layer equalization """

    @pytest.fixture(scope='class')
    def front_part_of_mobile_net_v1(self):
        """
        Front part of MobileNetV1
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same"),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same"),
            tf.keras.layers.Conv2D(64, kernel_size=1, strides=1, padding="same"),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same"),
            tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same"),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same"),
            tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same")
        ])

        return model

    @pytest.fixture(scope='class')
    def front_part_of_mobile_net_v1_like(self):
        """
        Network structure which is similar front part of MobileNetV1
        Changed some layer (Conv2D -> Conv2DTranspose) or layer parameter (Disable bias) for the test
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same", use_bias=False),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same"),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=1, strides=1, padding="same"),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", use_bias=False),
            tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same"),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same"),
            tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding="same")
        ])

        return model

    def test_find_layer_groups_in_vgg16(self):
        """
        VGG16 is a typical sequential structure
        Input -> Conv -> Conv -> Pooling -> Conv -> Conv -> Pooling
               -> Conv -> Conv -> Conv -> Pooling
               -> Conv -> Conv -> Conv -> Pooling
               -> Conv -> Conv -> Conv -> Pooling
               -> Dense -> Dense -> Dense
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(512, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)
        ])
        model.build((None, 224, 224, 3))

        graph_search_utils = GraphSearchUtils(model)
        layer_groups = graph_search_utils.find_layer_groups_to_scale()

        assert len(layer_groups) == 5

    def test_find_layer_groups_in_network_with_residual(self):
        """
        Sample network that has residual connection (branching)
        Input -> Conv -> ReLU -> MaxPool ->
               -> (In Residual) Conv -> ReLU -> Conv -> ReLU
               -> (In Residual) Conv -> ReLU -> Conv -> ReLU
               -> Dense
        """
        def residual_block(inputs):
            x = tf.keras.layers.Conv2D(32, kernel_size=3, padding="same")(inputs)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(64, kernel_size=1)(x)

            x += inputs
            x = tf.keras.layers.Activation(tf.nn.relu)(x)
            return x

        input_layer = tf.keras.layers.Input(shape=(28, 28, 3))
        model = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(input_layer)
        model = tf.keras.layers.ReLU()(model)
        model = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(model)
        model = residual_block(model)
        model = residual_block(model)
        output = tf.keras.layers.Dense(units=10)(model)

        keras_model = tf.keras.Model(inputs=input_layer, outputs=output)
        graph_search_utils = GraphSearchUtils(keras_model)
        layer_groups = graph_search_utils.find_layer_groups_to_scale()
        assert len(layer_groups) == 2

    def test_find_layer_groups_in_network_with_multiple_inputs(self):
        """
        Sample network that has multiple inbounds
        Left Input -> Conv -> Conv -> MaxPool -> Conv -> Conv -> Flatten
        Right Input -> Conv -> Conv -> Conv -> Flatten
        Output -> Concat([Left Output, Right Output])
        """

        def generate_model():
            x1 = tf.keras.layers.Input(shape=(28, 28, 1))
            left_input = tf.keras.layers.Conv2D(16, 3, activation="relu")(x1)
            left_input = tf.keras.layers.Conv2D(32, 3, activation="relu")(left_input)
            left_input = tf.keras.layers.MaxPool2D(3)(left_input)
            left_input = tf.keras.layers.Conv2D(32, 3, activation="relu")(left_input)
            left_input = tf.keras.layers.Conv2D(16, 3, activation="relu")(left_input)
            left_input = tf.keras.layers.Flatten()(left_input)

            x2 = tf.keras.layers.Input(shape=(28, 28, 1))
            right_input = tf.keras.layers.Conv2D(64, 3, activation="relu")(x2)
            right_input = tf.keras.layers.Conv2D(32, 3, activation="relu")(right_input)
            right_input = tf.keras.layers.Conv2D(16, 3, activation="relu")(right_input)
            right_input = tf.keras.layers.Flatten()(right_input)

            y = tf.keras.layers.Concatenate()([left_input, right_input])
            return tf.keras.models.Model([x1, x2], y)

        model = generate_model()
        graph_search_utils = GraphSearchUtils(model)
        layer_groups = graph_search_utils.find_layer_groups_to_scale()

        assert len(layer_groups) == 3

    def test_convert_layer_group_to_cls_sets_for_consecutive_convolution(self):
        """
        Layer group (Consecutive Convolution)
        """

        conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3)
        conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=5)
        conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=5)
        conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=7)
        layer_group = [conv1, conv2, conv3, conv4]

        actual = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
        expected = [(conv1, conv2), (conv2, conv3), (conv3, conv4)]

        assert actual == expected

    def test_convert_layer_group_to_cls_sets_for_depthwise_separable_convolution(self):
        """
        Layer group (Typical Depthwise Separable Convolution)
        """

        conv = tf.keras.layers.Conv2D(filters=8, kernel_size=3)
        depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3)
        pointwise_conv = tf.keras.layers.Conv2D(filters=16, kernel_size=1)
        layer_group = [conv, depthwise_conv, pointwise_conv]

        actual = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
        expected = [(conv, depthwise_conv, pointwise_conv)]

        assert actual == expected

    def test_convert_layer_group_to_cls_sets_for_part_of_mobile_net_v1(self, front_part_of_mobile_net_v1):
        """
        Layer group from the earlier part of MobileNetV1
        """

        layers = front_part_of_mobile_net_v1.layers

        conv1, dw_conv1, pointwise_conv1, dw_conv2, pointwise_conv2, dw_conv3, pointwise_conv3 = layers
        layer_group = [conv1, dw_conv1, pointwise_conv1, dw_conv2, pointwise_conv2, dw_conv3, pointwise_conv3]

        actual = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
        expected = [
            (conv1, dw_conv1, pointwise_conv1),
            (pointwise_conv1, dw_conv2, pointwise_conv2),
            (pointwise_conv2, dw_conv3, pointwise_conv3)
        ]

        assert actual == expected

    def test_convert_layer_group_to_cls_sets_for_consecutive_depthwise_convolution(self):
        """
        Layer group (Consecutive Depthwise Separable Convolution)
        """

        conv = tf.keras.layers.Conv2D(filters=8, kernel_size=3)
        depthwise_conv1 = tf.keras.layers.DepthwiseConv2D(kernel_size=3)
        depthwise_conv2 = tf.keras.layers.DepthwiseConv2D(kernel_size=5)
        layer_group = [conv, depthwise_conv1, depthwise_conv2]

        with pytest.raises(NotImplementedError):
            _ = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)

    def test_scale_cls_set_with_conv_layer(self):
        """
        Test scaling logic for cls set consisting of two convolution layers
        Possible two convolution layer cases are
          - (Conv2D, Conv2D)
          - (Conv2D, Conv2DTranspose)
          - (Conv2DTranspose, Conv2D)
          - (Conv2DTranspose, Conv2DTranspose)
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(4, kernel_size=3, activation='relu', bias_initializer='normal'),
            tf.keras.layers.Conv2D(16, kernel_size=5, activation='relu', use_bias=False),
            tf.keras.layers.Conv2DTranspose(16, kernel_size=7, use_bias=False),
            tf.keras.layers.Conv2DTranspose(8, kernel_size=5, strides=2, bias_initializer='normal'),
            tf.keras.layers.Conv2D(4, kernel_size=3, activation='relu', bias_initializer='normal'),
            tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', bias_initializer='normal'),
            tf.keras.layers.Flatten()
        ])

        np.random.seed(42)
        inputs = np.random.randn(1, 28, 28, 3).astype('f')
        outputs_before_scaling = model.predict(inputs)

        conv1, conv2, conv_transpose1, conv_transpose2, conv3, conv4, _ = model.layers

        # (Conv2D, Conv2D w/o bias) case
        CrossLayerScaling.scale_cls_set_with_conv_layers((conv1, conv2))
        conv1_output_range = WeightTensorUtils.get_max_abs_val_per_channel(conv1, axis=(2, 0, 1))
        conv2_input_range = WeightTensorUtils.get_max_abs_val_per_channel(conv2, axis=(3, 0, 1))
        assert np.allclose(conv1_output_range, conv2_input_range)

        # (Conv2D w/o bias, Conv2DTranspose w/o bias) case
        CrossLayerScaling.scale_cls_set_with_conv_layers((conv2, conv_transpose1))
        conv2_output_range = WeightTensorUtils.get_max_abs_val_per_channel(conv2, axis=(2, 0, 1))
        conv_transpose1_input_range = WeightTensorUtils.get_max_abs_val_per_channel(conv_transpose1, axis=(2, 0, 1))
        assert np.allclose(conv2_output_range, conv_transpose1_input_range)

        # (Conv2DTranspose w/o bias, Conv2DTranspose) case
        CrossLayerScaling.scale_cls_set_with_conv_layers((conv_transpose1, conv_transpose2))
        conv_transpose1_output_range = WeightTensorUtils.get_max_abs_val_per_channel(conv_transpose1, axis=(3, 0, 1))
        conv_transpose2_input_range = WeightTensorUtils.get_max_abs_val_per_channel(conv_transpose2, axis=(2, 0, 1))
        assert np.allclose(conv_transpose1_output_range, conv_transpose2_input_range)

        # (Conv2DTranspose, Conv2D) case
        CrossLayerScaling.scale_cls_set_with_conv_layers((conv_transpose2, conv3))
        conv_transpose2_output_range = WeightTensorUtils.get_max_abs_val_per_channel(conv_transpose2, axis=(3, 0, 1))
        conv3_input_range = WeightTensorUtils.get_max_abs_val_per_channel(conv3, axis=(3, 0, 1))
        assert np.allclose(conv_transpose2_output_range, conv3_input_range)

        # (Conv2D, Conv2D) case
        CrossLayerScaling.scale_cls_set_with_conv_layers((conv3, conv4))
        conv3_output_range = WeightTensorUtils.get_max_abs_val_per_channel(conv3, axis=(2, 0, 1))
        conv4_input_range = WeightTensorUtils.get_max_abs_val_per_channel(conv4, axis=(3, 0, 1))
        assert np.allclose(conv3_output_range, conv4_input_range)

        outputs_after_scaling = model.predict(inputs)
        np.allclose(outputs_before_scaling, outputs_after_scaling)

    def test_scale_cls_set_with_depthwise_separable_conv_layer(self, front_part_of_mobile_net_v1_like):
        """
        Test scaling logic for cls set consisting of depthwise convolution layers
        """
        model = front_part_of_mobile_net_v1_like

        np.random.seed(42)
        inputs = np.random.randn(1, 224, 224, 3).astype('f')
        outputs_before_scaling = model.predict(inputs)

        conv1, depthwise_conv1, conv_transpose1, depthwise_conv2, conv2, _, _ = model.layers

        # (Conv2D w/o bias, DepthwiseConv2D, ConvTranspose2D) case
        CrossLayerScaling.scale_cls_set_with_depthwise_conv_layers((conv1, depthwise_conv1, conv_transpose1))
        conv1_output_range = WeightTensorUtils.get_max_abs_val_per_channel(conv1, axis=(2, 0, 1))
        depthwise_conv1_output_range = WeightTensorUtils.get_max_abs_val_per_channel(depthwise_conv1, axis=(3, 0, 1))
        conv_transpose1_input_range = WeightTensorUtils.get_max_abs_val_per_channel(conv_transpose1, axis=(2, 0, 1))
        assert np.allclose(conv1_output_range, depthwise_conv1_output_range)
        assert np.allclose(depthwise_conv1_output_range, conv_transpose1_input_range)

        # (ConvTranspose2D, DepthwiseConv2D w/o bias, Conv2D) case
        CrossLayerScaling.scale_cls_set_with_depthwise_conv_layers((conv_transpose1, depthwise_conv2, conv2))
        conv_transpose1_output_range = WeightTensorUtils.get_max_abs_val_per_channel(conv2, axis=(3, 0, 1))
        depthwise_conv2_output_range = WeightTensorUtils.get_max_abs_val_per_channel(depthwise_conv2, axis=(3, 0, 1))
        conv2_input_range = WeightTensorUtils.get_max_abs_val_per_channel(conv2, axis=(3, 0, 1))
        assert np.allclose(conv_transpose1_output_range, depthwise_conv2_output_range)
        assert np.allclose(depthwise_conv2_output_range, conv2_input_range)

        outputs_after_scaling = model.predict(inputs)
        np.allclose(outputs_before_scaling, outputs_after_scaling)

    def test_is_relu_activation_present_in_cls_sets(self):
        """
        Test ReLU activation present in cls sets
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(4, kernel_size=3, activation=None),                              # conv1
            tf.keras.layers.Conv2D(8, kernel_size=3, activation='relu'),                            # conv2
            tf.keras.layers.Conv2D(16, kernel_size=3, activation=tf.keras.activations.relu),        # conv3
            tf.keras.layers.Conv2D(8, kernel_size=3, activation=tf.keras.layers.PReLU()),           # conv4
            tf.keras.layers.Conv2D(4, kernel_size=3, activation=tf.keras.layers.ReLU()),            # conv5
            tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),                           # conv6
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, activation=tf.keras.layers.PReLU()),     # dw_conv1
            tf.keras.layers.Conv2D(64, kernel_size=1, activation=tf.keras.layers.ReLU()),           # pw_conv1
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, activation=None),                        # dw_conv2
            tf.keras.layers.Conv2D(32, kernel_size=1, activation=tf.keras.layers.PReLU())           # pw_conv2
        ])

        # dw means Depthwise / pw means Pointwise
        conv1, conv2, conv3, conv4, conv5, conv6, dw_conv1, pw_conv1, dw_conv2, pw_conv2 = model.layers
        cls_sets = [
            (conv1, conv2),
            (conv2, conv3),
            (conv3, conv4),
            (conv4, conv5),
            (conv5, conv6),
            (conv6, dw_conv1, pw_conv1),
            (pw_conv1, dw_conv2, pw_conv2)
        ]

        expected = [
            False,
            True,
            True,
            True,
            True,
            (True, True),
            (True, False)
        ]
        actual = GraphSearchUtils.is_relu_activation_present_in_cls_sets(cls_sets)

        assert actual == expected

    def test_is_relu_activation_present_when_declared_separately(self):
        """
        Test ReLU activation present even if ReLU declared separately
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(4, kernel_size=3),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(16, kernel_size=3, activation=None),
            tf.keras.layers.PReLU(),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation=None),
            tf.keras.layers.Conv2D(8, kernel_size=3, activation='relu')
        ])

        conv1, _, conv2, _, conv3, conv4 = model.layers
        cls_sets = [
            (conv1, conv2),
            (conv2, conv3),
            (conv3, conv4)
        ]

        expected = [
            True,
            True,
            False
        ]
        actual = GraphSearchUtils.is_relu_activation_present_in_cls_sets(cls_sets)

        assert actual == expected

    def test_is_relu_activation_present_where_there_is_folded_batchnorm(self):
        """
        Test ReLU activation present where there is folded batchnorm between Conv layers
        """
        model = keras_functional_conv_net()

        _, model = fold_all_batch_norms(model)
        _, conv1, _, conv2, _, conv3, conv4 = model.layers
        cls_sets = [(conv1, conv2), (conv2, conv3), (conv3, conv4)]

        expected = [True, True, False]
        actual = GraphSearchUtils.is_relu_activation_present_in_cls_sets(cls_sets)

        assert actual == expected

    def test_scale_cls_sets(self):
        """
        Test scale cls sets
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(8, kernel_size=2, activation='relu'),
            tf.keras.layers.Conv2D(16, kernel_size=2, activation=None),
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=2, activation=None),
            tf.keras.layers.Conv2D(4, kernel_size=1, activation=None),
            tf.keras.layers.Conv2D(8, kernel_size=2, activation='relu')
        ])

        inp_array = np.random.randn(10, 28, 28, 3)
        before_scaling_conv0_weight = model.layers[0].get_weights()[0]
        before_scaling_ouptut = model.predict(inp_array)

        cls_sets = [
            (model.layers[0], model.layers[1]),
            (model.layers[1], model.layers[3], model.layers[4]),
            (model.layers[4], model.layers[5])
        ]

        scaling_factors = CrossLayerScaling.scale_cls_sets(cls_sets)
        assert len(scaling_factors) == 3
        assert len(scaling_factors[1]) == 2

        after_scaling_conv0_weight = model.layers[0].get_weights()[0]
        assert not np.array_equal(before_scaling_conv0_weight, after_scaling_conv0_weight)
        after_scaling_output = model.predict(inp_array)

        assert np.allclose(before_scaling_ouptut, after_scaling_output, rtol=1.e-2, atol=1.e-2)

    def test_create_cls_set_info_list(self):
        """
        Test create cls set info list for normal cases
        """
        np.random.seed(42)
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(4, kernel_size=3, activation=None),
            tf.keras.layers.Conv2D(8, kernel_size=3, activation='relu'),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, activation=tf.keras.layers.PReLU()),
            tf.keras.layers.Conv2D(64, kernel_size=1, activation=tf.keras.layers.ReLU()),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, activation=None),
            tf.keras.layers.Conv2D(32, kernel_size=1, activation=tf.keras.layers.PReLU())
        ])

        # dw means Depthwise / pw means Pointwise
        conv1, conv2, conv3, dw_conv1, pw_conv1, dw_conv2, pw_conv2 = model.layers
        cls_sets = [
            (conv1, conv2),
            (conv2, conv3),
            (conv3, dw_conv1, pw_conv1),
            (pw_conv1, dw_conv2, pw_conv2)
        ]

        scale_factor1 = np.random.randn(4)
        scale_factor2 = np.random.randn(8)
        scale_factor3 = (np.random.randn(32), np.random.randn(32))
        scale_factor4 = (np.random.randn(64), np.random.randn(64))
        scale_factors = [
            scale_factor1, scale_factor2, scale_factor3, scale_factor4
        ]

        is_relu_activation_in_cls_sets = [
            False,
            True,
            (True, True),
            (True, False)
        ]

        actual = CrossLayerScaling.create_cls_set_info_list(cls_sets, scale_factors, is_relu_activation_in_cls_sets)
        expected = [
            ClsSetInfo(ClsSetInfo.ClsSetLayerPairInfo(conv1, conv2, scale_factor1, False)),
            ClsSetInfo(ClsSetInfo.ClsSetLayerPairInfo(conv2, conv3, scale_factor2, True)),
            ClsSetInfo(ClsSetInfo.ClsSetLayerPairInfo(conv3, dw_conv1, scale_factor3[0], True),
                       ClsSetInfo.ClsSetLayerPairInfo(dw_conv1, pw_conv1, scale_factor3[1], True)),
            ClsSetInfo(ClsSetInfo.ClsSetLayerPairInfo(pw_conv1, dw_conv2, scale_factor4[0], True),
                       ClsSetInfo.ClsSetLayerPairInfo(dw_conv2, pw_conv2, scale_factor4[1], False)),
        ]

        assert actual == expected

    def test_create_cls_set_info_list_for_exceptional_case(self):
        """
        Test create cls set info list for exceptional case
        """
        np.random.seed(42)
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 3)),
            tf.keras.layers.Conv2D(4, kernel_size=3, activation=None),
            tf.keras.layers.Conv2D(8, kernel_size=3, activation='relu'),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, activation=tf.keras.layers.PReLU()),
            tf.keras.layers.Conv2D(64, kernel_size=1, activation=tf.keras.layers.ReLU()),
        ])

        conv1, conv2, dw_conv1, pw_conv1 = model.layers
        cls_sets = [(conv1, conv2), (conv2, dw_conv1, pw_conv1)]
        is_relu_activation_in_cls_sets = [False, (True, True)]
        scale_factors1 = [np.random.randn(4)]

        # Raise assertion if the length of the three input arrays do not match
        # assert len(cls_sets) == len(scale_factors) == len(is_relu_activation_in_cls_sets)
        with pytest.raises(AssertionError):
            CrossLayerScaling.create_cls_set_info_list(cls_sets, scale_factors1, is_relu_activation_in_cls_sets)

        # Raise assertion if the length of cls set is not three in case of depthwise separable conv layer
        # assert len(cls_set) == 3
        cls_sets2 = [(conv1, conv2), (conv2, dw_conv1)]
        scale_factors2 = [np.random.randn(4), (np.random.randn(8), np.random.randn(8))]
        with pytest.raises(AssertionError):
            CrossLayerScaling.create_cls_set_info_list(cls_sets2, scale_factors2, is_relu_activation_in_cls_sets)

        # Raise assertion if scale factors or relu flags are not tuple in case of depthwise separable conv layer
        # assert len(scale_factor) == len(has_relu_activation) == 2
        scale_factors3 = [np.random.randn(4), (np.random.randn(8),)]
        with pytest.raises(AssertionError):
            CrossLayerScaling.create_cls_set_info_list(cls_sets, scale_factors3, is_relu_activation_in_cls_sets)
