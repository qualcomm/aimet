# -*- mode: python -*-
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
import tensorflow as tf

from aimet_tensorflow.keras.batch_norm_fold import fold_all_batch_norms
from aimet_tensorflow.keras.batch_norm_fold import fold_given_batch_norms
from aimet_tensorflow.keras.cross_layer_equalization import HighBiasFold, CrossLayerScaling
from aimet_tensorflow.keras.cross_layer_equalization import equalize_model
from aimet_tensorflow.keras.utils.model_transform_utils import replace_relu6_with_relu


def cross_layer_equalization_auto():
    input_shape = (224, 224, 3)
    model = tf.keras.applications.ResNet50()

    cle_applied_model = equalize_model(model)


def cross_layer_equalization_auto_stepwise():
    """
    Individual api calls to perform cross layer equalization one step at a time. Pairs to fold and
    scale are found automatically.
    1. Replace Relu6 with Relu
    2. Fold batch norms
    3. Perform cross layer scaling
    4. Perform high bias fold
    """

    # Load the model to equalize
    model = tf.keras.applications.resnet50.ResNet50(weights=None, classes=10)

    # 1. Replace Relu6 layer with Relu
    model_for_cle, _ = replace_relu6_with_relu(model)

    # 2. Fold all batch norms
    folded_pairs, model = fold_all_batch_norms(model_for_cle)

    bn_dict = {}
    for conv_or_linear, bn in folded_pairs:
        bn_dict[conv_or_linear] = bn

    # 3. Perform cross-layer scaling on applicable layer groups
    cls_set_info_list = CrossLayerScaling.scale_model(model_for_cle)

    # 4. Perform high bias fold
    HighBiasFold.bias_fold(cls_set_info_list, bn_dict)

    return model_for_cle


def cross_layer_equalization_manual():
    """
    Individual api calls to perform cross layer equalization one step at a time. Pairs to fold and
    scale are provided by the user.
    1. Replace Relu6 with Relu
    2. Fold batch norms
    3. Perform cross layer scaling
    4. Perform high bias fold
    """

    # Load the model to equalize
    model = tf.keras.applications.resnet50.ResNet50(weights=None, classes=10)

    # replace any ReLU6 layers with ReLU
    model_for_cle, _ = replace_relu6_with_relu(model)

    # pick potential pairs of conv and bn ops for fold
    layer_pairs = get_example_layer_pairs_resnet50_for_folding(model_for_cle)

    # fold given layers
    fold_given_batch_norms(model_for_cle, layer_pairs=layer_pairs)

    # Cross Layer Scaling
    # Create a list of consecutive conv layers to be equalized
    consecutive_layer_list = get_consecutive_layer_list_from_resnet50_for_scaling(model_for_cle)

    # invoke api to perform scaling on given list of cls pairs
    scaling_factor_list = CrossLayerScaling.scale_cls_sets(consecutive_layer_list)

    # get info from bn fold and cross layer scaling in format required for high bias fold
    folded_pairs, cls_set_info_list = format_info_for_high_bias_fold(layer_pairs,
                                                                     consecutive_layer_list,
                                                                     scaling_factor_list)

    HighBiasFold.bias_fold(cls_set_info_list, folded_pairs)
    return model_for_cle


def get_example_layer_pairs_resnet50_for_folding(model: tf.keras.Model):
    """
    Function to pick example conv-batchnorm layer pairs for folding.
    :param model: Keras model containing conv batchnorm pairs to fold
    :return: pairs of conv and batchnorm layers for batch norm folding in Resnet50 model.
    """

    conv_op_1 = model.layers[2]
    bn_op_1 = model.layers[3]

    conv_op_2 = model.layers[7]
    bn_op_2 = model.layers[8]

    conv_op_3 = model.layers[10]
    bn_op_3 = model.layers[11]

    # make a layer pair list with potential the conv op and bn_op pair along with a flag
    # to indicate if given bn op can be folded upstream or downstream.
    # example of two pairs of conv and bn op  shown below
    layer_pairs = [(conv_op_1, bn_op_1, True),
                   (conv_op_2, bn_op_2, True),
                   (conv_op_3, bn_op_3, True)]

    return layer_pairs


def get_consecutive_layer_list_from_resnet50_for_scaling(model: tf.keras.Model):
    """
    helper function to pick example consecutive layer list for scaling.
    :param model: tf.keras.Model
    :return: sample layers for scaling as consecutive_layer_list from Resnet50 model
    """
    conv_op_1 = model.layers[2]
    conv_op_2 = model.layers[7]
    conv_op_3 = model.layers[10]

    consecutive_layer_list = [(conv_op_1, conv_op_2), (conv_op_2, conv_op_3)]
    return consecutive_layer_list


def format_info_for_high_bias_fold(layer_pairs, consecutive_layer_list, scaling_factor_list):
    """
    Helper function that formats data from cross layer scaling and bn fold for usage by high bias fold
    :param layer_pairs: info obtained after batchnorm fold
    :param consecutive_layer_list: info obtained after cross layer scaling
    :param scaling_factor_list: scaling params corresponding to consecutive_layer_list
    :return: data formatted for high bias fold
    """

    # convert info after batch norm fold and cross layer scaling for usage by high bias fold api
    folded_pairs = []
    for (conv_op, bn_op_with_meta, _fold_upstream_flag) in layer_pairs:
        folded_pairs.append((conv_op, bn_op_with_meta.op))

    # List that hold a boolean for if there were relu activations between layers of each cross layer scaling set
    is_relu_activation_in_cls_sets = []
    # Note the user is expected to fill in this list manually

    # Convert to a list of cls-set-info elements
    cls_set_info_list = CrossLayerScaling.create_cls_set_info_list(consecutive_layer_list,
                                                                   scaling_factor_list,
                                                                   is_relu_activation_in_cls_sets)

    return folded_pairs, cls_set_info_list
