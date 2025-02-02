# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2021, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" This file contains unit tests for testing high bias fold feature of CLE """
import numpy as np
import tensorflow as tf

from aimet_tensorflow.keras.cross_layer_equalization import HighBiasFold, ClsSetInfo


class TestHighBiasFold:
    """Test methods for High bias folding"""

    def test_high_bias_fold_for_standard_conv(self):
        """
        Test high bias folding for standard convolution - batch norm - relu - convolution case
        """
        seed = 42
        np.random.seed(seed)

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, input_shape=(32, 32, 3),
                                   bias_initializer=tf.keras.initializers.RandomNormal(seed=seed)),
            tf.keras.layers.BatchNormalization(beta_initializer=tf.keras.initializers.RandomNormal(3, 0.1, seed=seed),
                                               gamma_initializer=tf.keras.initializers.RandomNormal(0, 1, seed=seed)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, 3, bias_initializer="normal")
        ])

        conv1, bn1, _, conv2 = model.layers
        bn_dict = {conv1: bn1}

        scale_factor = np.array(np.random.randn(conv1.kernel.shape[3]))
        cls_pair_info = ClsSetInfo.ClsSetLayerPairInfo(conv1, conv2, scale_factor, True)
        cls_set_info = ClsSetInfo(cls_pair_info)

        _, bias1_before_folding = conv1.get_weights()
        _, bias2_before_folding = conv2.get_weights()
        HighBiasFold.bias_fold([cls_set_info], bn_dict)
        _, bias1_after_folding = conv1.get_weights()
        _, bias2_after_folding = conv2.get_weights()

        # hat of bias1 = bias1 - c, c = max(0, beta - 3 * gamma)
        # Bias value of previous layer after folding should be less than or equal to the value before folding
        for bias_val_before, bias_val_after in zip(bias1_before_folding, bias1_after_folding):
            assert bias_val_after <= bias_val_before

        # hat of bias2 = weight2 * hat of h + bias2
        # Bias value of current layer after folding should be different the value before folding
        assert not np.allclose(bias2_before_folding, bias2_after_folding)

    def test_high_bias_fold_for_transposed_conv(self):
        """
        Test high bias folding for transposed convolution - batch norm - relu - transposed convolution case
        """
        seed = 42
        np.random.seed(seed)

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(32, 3, input_shape=(32, 32, 3), bias_initializer="normal"),
            tf.keras.layers.BatchNormalization(beta_initializer=tf.keras.initializers.RandomNormal(3, 0.1, seed=seed),
                                               gamma_initializer=tf.keras.initializers.RandomNormal(0, 1, seed=seed)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(64, 3, bias_initializer="normal")
        ])

        transpose_conv1, bn1, _, transpose_conv2 = model.layers
        bn_dict = {transpose_conv1: bn1}

        scale_factor = np.array(np.random.randn(transpose_conv1.kernel.shape[2]))
        cls_pair_info = ClsSetInfo.ClsSetLayerPairInfo(transpose_conv1, transpose_conv2, scale_factor, True)
        cls_set_info = ClsSetInfo(cls_pair_info)

        _, bias1_before_folding = transpose_conv1.get_weights()
        _, bias2_before_folding = transpose_conv2.get_weights()
        HighBiasFold.bias_fold([cls_set_info], bn_dict)
        _, bias1_after_folding = transpose_conv1.get_weights()
        _, bias2_after_folding = transpose_conv2.get_weights()

        # hat of bias1 = bias1 - c, c = max(0, beta - 3 * gamma)
        # Bias value of previous layer after folding should be less than or equal to the value before folding
        for bias_val_before, bias_val_after in zip(bias1_before_folding, bias1_after_folding):
            assert bias_val_after <= bias_val_before

        # hat of bias2 = weight2 * hat of h + bias2
        # Bias value of current layer after folding should be different the value before folding
        assert not np.allclose(bias2_before_folding, bias2_after_folding)

    def test_high_bias_fold_for_depthwise_separable_conv(self):
        """
        Test high bias folding for depthwise separable convolution case
        """
        seed = 42
        np.random.seed(seed)

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(32, 32, 3), strides=2,
                                   bias_initializer=tf.keras.initializers.RandomNormal(seed=seed)),
            tf.keras.layers.BatchNormalization(beta_initializer=tf.keras.initializers.RandomNormal(3, 0.1, seed=seed),
                                               gamma_initializer=tf.keras.initializers.RandomNormal(1, 0.5, seed=seed)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same",
                                            bias_initializer=tf.keras.initializers.RandomNormal(seed=seed)),
            tf.keras.layers.BatchNormalization(beta_initializer=tf.keras.initializers.RandomNormal(1, 0.5, seed=seed),
                                               gamma_initializer=tf.keras.initializers.RandomNormal(0, 0.5, seed=seed)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, kernel_size=1, bias_initializer=tf.keras.initializers.RandomNormal(seed=seed)),
            tf.keras.layers.BatchNormalization(beta_initializer=tf.keras.initializers.RandomNormal(seed=seed),
                                               gamma_initializer=tf.keras.initializers.RandomNormal(seed=seed))
        ])

        conv1, bn1, _, dw_conv1, bn2, _, pw_conv1, _ = model.layers
        bn_dict = {conv1: bn1, dw_conv1: bn2}

        conv1_weight_tensor, _ = conv1.get_weights()
        scale_factor1 = np.array(np.random.randn(conv1_weight_tensor.shape[3]))
        cls_pair_info1 = ClsSetInfo.ClsSetLayerPairInfo(conv1, dw_conv1, scale_factor1, True)

        dw_conv_weight_tensor, _ = dw_conv1.get_weights()
        scale_factor2 = np.array(np.random.randn(dw_conv_weight_tensor.shape[2]))
        cls_pair_info2 = ClsSetInfo.ClsSetLayerPairInfo(dw_conv1, pw_conv1, scale_factor2, True)

        cls_set_info = ClsSetInfo(cls_pair_info1, cls_pair_info2)

        _, bias1_before_folding = conv1.get_weights()
        _, bias2_before_folding = dw_conv1.get_weights()
        _, bias3_before_folding = pw_conv1.get_weights()
        HighBiasFold.bias_fold([cls_set_info], bn_dict)
        _, bias1_after_folding = conv1.get_weights()
        _, bias2_after_folding = dw_conv1.get_weights()
        _, bias3_after_folding = pw_conv1.get_weights()

        # hat of bias1 = bias1 - c, c = max(0, beta - 3 * gamma)
        # Bias value of previous layer after folding should be less than or equal to the value before folding
        for bias_val_before, bias_val_after in zip(bias1_before_folding, bias1_after_folding):
            assert bias_val_after <= bias_val_before

        # hat of bias2 = weight2 * hat of h + bias2
        # Bias value of current layer after folding should be different the value before folding
        assert not np.allclose(bias2_before_folding, bias2_after_folding)
        assert not np.allclose(bias3_before_folding, bias3_after_folding)
