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
import tensorflow as tf
from packaging import version

from aimet_common.bias_correction import ConvBnPatternHandler
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.graph_searcher import GraphSearcher
from aimet_tensorflow.keras.connectedgraph import ConnectedGraph as KerasConnectedGraph


def test_connected_graph_resnet50_keras():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        keras_model = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3))
        connected_graph_from_keras_model = KerasConnectedGraph(keras_model)

        ops_from_keras_model = connected_graph_from_keras_model.get_all_ops()
        products_from_keras_model = connected_graph_from_keras_model.get_all_products()

        # Approx number of ops
        assert len(ops_from_keras_model) > 180
        # Approx number of products
        assert len(products_from_keras_model) > 450
        assert connected_graph_from_keras_model._split_count == 16

        split_product = products_from_keras_model["Split_0__to__multiple_ops"]
        assert ops_from_keras_model["Conv_6"] in split_product.consumers
        assert ops_from_keras_model["Conv_12"] in split_product.consumers

        add_op = ops_from_keras_model["Add_16"]
        assert ops_from_keras_model["BatchNormalization_14"] in add_op.input_ops
        assert ops_from_keras_model["BatchNormalization_15"] in add_op.input_ops
        assert products_from_keras_model["BatchNormalization_14_to_Add_16"] in add_op.inputs
        assert products_from_keras_model["BatchNormalization_15_to_Add_16"] in add_op.inputs


def test_graph_searcher_functionality_keras():
    if version.parse(tf.version.VERSION) >= version.parse("2.00"):
        keras_model = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3))
        connected_graph_from_keras_model = KerasConnectedGraph(keras_model)

        patterns_with_callbacks = []
        layer_select_handler = ConvBnPatternHandler()
        conv_types = ['Conv1d', 'Conv', 'ConvTranspose']
        linear_types = ['Gemm']

        for op_type in conv_types + linear_types:
            patterns_with_callbacks.append(PatternType(pattern=['BatchNormalization', op_type],
                                                       action=layer_select_handler))
            patterns_with_callbacks.append(PatternType(pattern=[op_type, 'BatchNormalization'],
                                                       action=layer_select_handler))

        graph_searcher = GraphSearcher(connected_graph_from_keras_model, patterns_with_callbacks)

        graph_searcher.find_all_patterns_in_graph_apply_actions()
        conv_linear_bn_info_dict = layer_select_handler.get_conv_linear_bn_info_dict()
        assert len(conv_linear_bn_info_dict) == 53
