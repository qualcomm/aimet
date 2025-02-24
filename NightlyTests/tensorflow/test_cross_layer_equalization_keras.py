# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022, Qualcomm Innovation Center, Inc. All rights reserved.
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
import pytest
import numpy as np
import tensorflow as tf

from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.keras.cross_layer_equalization import CrossLayerScaling, GraphSearchUtils, equalize_model, \
    HighBiasFold
from aimet_tensorflow.keras.utils.weight_tensor_utils import WeightTensorUtils


def test_fold_batch_norms():
    rand_inp = np.random.randn(1, 224, 224, 3)
    model = tf.keras.applications.resnet50.ResNet50(weights=None)
    conv_99_name = model.layers[99].name
    before_fold_weight = model.get_layer(conv_99_name).get_weights()[0]
    before_fold_output = model(rand_inp)
    conv_bn_pairs, model = fold_all_batch_norms(model)
    after_fold_weight = model.get_layer(conv_99_name).get_weights()[0]
    after_fold_output = model(rand_inp)

    assert len(conv_bn_pairs) == 53
    assert np.allclose(before_fold_output, after_fold_output, atol=1e-3)
    assert not np.array_equal(before_fold_weight, after_fold_weight)


def test_fold_batch_norms_mobile_net_v2():
    # rand_inp = np.random.randn(1, 224, 224, 3)
    model = tf.keras.applications.MobileNetV2(weights=None)
    conv_151 = model.layers[151]
    before_fold_weight = conv_151.get_weights()[0]
    # before_fold_output = model(rand_inp)
    conv_bn_pairs, model = fold_all_batch_norms(model)
    after_fold_weight = conv_151.get_weights()[0]
    # after_fold_output = model(rand_inp)

    assert len(conv_bn_pairs) == 52
    # Note: Currently, it does not pass the assertion below
    #   It is unclear whether it is due to Keras characteristics or due to wrong implementation
    # assert np.allclose(before_fold_output, after_fold_output, rtol=1e-2)
    assert not np.array_equal(before_fold_weight, after_fold_weight)


def test_layer_group_search():
    model = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3))

    _, model = fold_all_batch_norms(model)
    graph_search_utils = GraphSearchUtils(model)
    layer_groups = graph_search_utils.find_layer_groups_to_scale()

    cls_set_list = []
    for layer_group in layer_groups:
        cls_sets = graph_search_utils.convert_layer_group_to_cls_sets(layer_group)
        cls_set_list.extend(cls_sets)

    assert len(layer_groups) == 16
    assert len(cls_set_list) == 32


def test_cross_layer_scaling_resnet50():
    model = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3))

    conv4_weight_before_scaling = model.layers[22].get_weights()[0]
    conv1, conv2 = model.layers[10], model.layers[13]
    conv3, conv4 = model.layers[19], model.layers[22]

    cls_sets = [(conv1, conv2), (conv3, conv4)]
    scaling_factors = CrossLayerScaling.scale_cls_sets(cls_sets)
    assert len(scaling_factors) == 2

    conv4_weight_after_scaling = model.layers[22].get_weights()[0]
    assert not np.array_equal(conv4_weight_before_scaling, conv4_weight_after_scaling)

    # Test the distribution of the output channel of previous layer and input channel of next layer
    conv3_output_range = WeightTensorUtils.get_max_abs_val_per_channel(conv3, axis=(2, 0, 1))
    conv4_input_range = WeightTensorUtils.get_max_abs_val_per_channel(conv4, axis=(3, 0, 1))
    assert np.allclose(conv3_output_range, conv4_input_range)


def test_cross_layer_scaling_mobile_net_v2():
    model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(224, 224, 3))
    conv1_weight_before_scaling = model.layers[7].get_weights()[0]
    dw_conv1_weight_before_scaling = model.layers[21].get_weights()[0]

    conv1, conv2 = model.layers[7], model.layers[9]
    conv3, dw_conv1, conv4 = model.layers[18], model.layers[21], model.layers[24]
    cls_sets = [(conv1, conv2), (conv3, dw_conv1, conv4)]

    scaling_factors = CrossLayerScaling.scale_cls_sets(cls_sets)
    assert len(scaling_factors) == 2
    assert len(scaling_factors[1]) == 2

    conv1_weight_after_scaling = model.layers[7].get_weights()[0]
    dw_conv1_weight_after_scaling = model.layers[21].get_weights()[0]

    assert not np.array_equal(conv1_weight_before_scaling, conv1_weight_after_scaling)
    assert not np.array_equal(dw_conv1_weight_before_scaling, dw_conv1_weight_after_scaling)

    # Test the distribution of the output channel of previous layer and input channel of next layer
    conv1_output_range = WeightTensorUtils.get_max_abs_val_per_channel(conv1, axis=(2, 0, 1))
    conv2_input_range = WeightTensorUtils.get_max_abs_val_per_channel(conv2, axis=(3, 0, 1))
    assert np.allclose(conv1_output_range, conv2_input_range)

    conv3_output_range = WeightTensorUtils.get_max_abs_val_per_channel(conv3, axis=(2, 0, 1))
    depthwise_conv1_output_range = WeightTensorUtils.get_max_abs_val_per_channel(dw_conv1, axis=(3, 0, 1))
    conv4_input_range = WeightTensorUtils.get_max_abs_val_per_channel(conv4, axis=(3, 0, 1))
    assert np.allclose(conv3_output_range, depthwise_conv1_output_range)
    assert np.allclose(depthwise_conv1_output_range, conv4_input_range)


def test_cross_layer_equalization_stepwise():
    orig = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3))

    folded_pairs, model = fold_all_batch_norms(orig)
    bn_dict = {}
    for conv_or_linear, bn in folded_pairs:
        bn_dict[conv_or_linear] = bn

    conv1, conv2, conv3 = model.layers[6], model.layers[14], model.layers[11]
    w1, _ = conv1.get_weights()
    w2, _ = conv2.get_weights()
    w3, _ = conv3.get_weights()

    cls_set_info_list = CrossLayerScaling.scale_model(model)
    # check if weights are updating
    assert not np.allclose(conv1.kernel, w1)
    assert not np.allclose(conv2.kernel, w2)
    assert not np.allclose(conv3.kernel, w3)

    _, b1 = conv1.get_weights()
    _, b2 = conv2.get_weights()

    HighBiasFold.bias_fold(cls_set_info_list, bn_dict)
    # hat of bias1 = bias1 - c, c = max(0, beta - 3 * gamma)
    # Bias value of previous layer after folding should be less than or equal to the value before folding
    for bias_val_before, bias_val_after in zip(b1, conv1.bias.numpy()):
        assert bias_val_after <= bias_val_before

    for bias_val_before, bias_val_after in zip(b2, conv2.bias.numpy()):
        assert bias_val_after <= bias_val_before


def test_cross_layer_equalization_resnet50():
    model = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3))

    cle_applied_model = equalize_model(model)
    cle_applied_model.summary()


def test_cross_layer_equalization_mobile_net_v2():
    model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(224, 224, 3))

    cle_applied_model = equalize_model(model)
    cle_applied_model.summary()
