# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022-2023, Qualcomm Innovation Center, Inc. All rights reserved.
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

import numpy as np
import copy
import torch
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession
from onnx import numpy_helper

from aimet_common.cross_layer_equalization import GraphSearchUtils
from aimet_onnx.meta.connectedgraph import ConnectedGraph, WEIGHT_INDEX
from aimet_onnx.cross_layer_equalization import get_ordered_list_of_conv_modules, \
    cls_supported_layer_types, cls_supported_activation_types, CrossLayerScaling, HighBiasFold, equalize_model
from aimet_onnx.utils import ParamUtils, replace_relu6_with_relu
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight

from models import models_for_tests


class TestCLS:
    def test_graph_search_utils_single_residual_model(self):
        model = models_for_tests.single_residual_model()
        connected_graph = ConnectedGraph(model)
        ordered_module_list = get_ordered_list_of_conv_modules(connected_graph.starting_ops)
        graph_search_utils = GraphSearchUtils(connected_graph, ordered_module_list, cls_supported_layer_types, cls_supported_activation_types)
        ordered_layer_groups = graph_search_utils.find_layer_groups_to_scale()[0]
        ordered_layer_groups_names = [op.dotted_name for op in ordered_layer_groups]
        assert ordered_layer_groups_names == ['/conv2/Conv', '/conv3/Conv']

    def test_find_cls_sets_depthwise_model(self):
        model = models_for_tests.depthwise_conv_model()

        connected_graph = ConnectedGraph(model)
        ordered_module_list = get_ordered_list_of_conv_modules(connected_graph.starting_ops)
        graph_search_utils = GraphSearchUtils(connected_graph, ordered_module_list, cls_supported_layer_types,
                                              cls_supported_activation_types)

        ordered_layer_groups = graph_search_utils.find_layer_groups_to_scale()[0]
        # Find cls sets from the layer groups
        cls_sets = graph_search_utils.convert_layer_group_to_cls_sets(ordered_layer_groups)
        cls_sets_names = []
        for cls_set in cls_sets:
            cls_sets_name = tuple([op.dotted_name for op in cls_set])
            cls_sets_names.append(cls_sets_name)
        assert cls_sets_names == [('/model/model.0/model.0.0/Conv', '/model/model.1/model.1.0/Conv', '/model/model.1/model.1.3/Conv'),
                                  ('/model/model.1/model.1.3/Conv', '/model/model.2/model.2.0/Conv', '/model/model.2/model.2.3/Conv'),
                                  ('/model/model.2/model.2.3/Conv', '/model/model.3/model.3.0/Conv', '/model/model.3/model.3.3/Conv'),
                                  ('/model/model.3/model.3.3/Conv', '/model/model.4/model.4.0/Conv', '/model/model.4/model.4.3/Conv'),
                                  ('/model/model.4/model.4.3/Conv', '/model/model.5/model.5.0/Conv', '/model/model.5/model.5.3/Conv'),
                                  ('/model/model.5/model.5.3/Conv', '/model/model.6/model.6.0/Conv', '/model/model.6/model.6.3/Conv'),
                                  ('/model/model.6/model.6.3/Conv', '/model/model.7/model.7.0/Conv', '/model/model.7/model.7.3/Conv'),
                                  ('/model/model.7/model.7.3/Conv', '/model/model.8/model.8.0/Conv', '/model/model.8/model.8.3/Conv')]

    def test_find_cls_sets_resnet_model(self):
        model = models_for_tests.single_residual_model()
        connected_graph = ConnectedGraph(model)
        ordered_module_list = get_ordered_list_of_conv_modules(connected_graph.starting_ops)
        graph_search_utils = GraphSearchUtils(connected_graph, ordered_module_list, cls_supported_layer_types,
                                              cls_supported_activation_types)

        ordered_layer_groups = graph_search_utils.find_layer_groups_to_scale()[0]
        # Find cls sets from the layer groups
        cls_sets = graph_search_utils.convert_layer_group_to_cls_sets(ordered_layer_groups)
        cls_sets_names = []
        for cls_set in cls_sets:
            cls_sets_name = tuple([op.dotted_name for op in cls_set])
            cls_sets_names.append(cls_sets_name)
        assert cls_sets_names == [('/conv2/Conv', '/conv3/Conv')]

    def test_scale_model_residual(self):
        model = models_for_tests.single_residual_model()
        input_shape = (1, 3, 32, 32)
        test_data = np.random.randn(*input_shape).astype(np.float32)
        session = _build_session(model)
        output_before_cls = session.run(None, {'input': test_data})
        cls = CrossLayerScaling(model)
        cls_set_info = cls.scale_model()
        session = _build_session(model)
        output_after_cls = session.run(None, {'input': test_data})
        assert np.allclose(output_after_cls, output_before_cls, rtol=1e-2, atol=1e-5)
        conv_3 = cls_set_info[0].cls_pair_info_list[0].layer1.get_module()
        conv_5 = cls_set_info[0].cls_pair_info_list[0].layer2.get_module()
        weight_3 = numpy_helper.to_array(ParamUtils.get_param(model.model, conv_3, WEIGHT_INDEX))
        weight_5 = numpy_helper.to_array(ParamUtils.get_param(model.model, conv_5, WEIGHT_INDEX))
        assert np.allclose(np.amax(np.abs(weight_3), axis=(1, 2, 3)), np.amax(np.abs(weight_5), axis=(0, 2, 3)))

    def test_scale_model_tranposed_conv(self):
        model = models_for_tests.transposed_conv_model_without_bn()
        input_shape = (10,10,4,4)
        test_data = np.random.randn(*input_shape).astype(np.float32)
        session = _build_session(model)
        output_before_cls = session.run(None, {'input': test_data})
        cls = CrossLayerScaling(model)
        cls_set_info = cls.scale_model()
        session = _build_session(model)
        output_after_cls = session.run(None, {'input': test_data})
        assert np.allclose(output_after_cls, output_before_cls, rtol=1e-2, atol=1e-5)
        conv_3 = cls_set_info[0].cls_pair_info_list[0].layer1.get_module()
        conv_5 = cls_set_info[0].cls_pair_info_list[0].layer2.get_module()
        weight_3 = numpy_helper.to_array(ParamUtils.get_param(model.model, conv_3, WEIGHT_INDEX))
        weight_5 = numpy_helper.to_array(ParamUtils.get_param(model.model, conv_5, WEIGHT_INDEX))
        assert np.allclose(np.amax(np.abs(weight_3), axis=(0, 2, 3)), np.amax(np.abs(weight_5), axis=(1, 2, 3)))

    def test_scale_model_depthwise(self):
        model = models_for_tests.depthwise_conv_model()
        input_shape = (1, 3, 224, 224)
        test_data = np.random.randn(*input_shape).astype(np.float32)
        session = _build_session(model)
        output_before_cls = session.run(None, {'input': test_data})
        cls = CrossLayerScaling(model)
        cls_set_infos = cls.scale_model()
        session = _build_session(model)
        output_after_cls = session.run(None, {'input': test_data})
        assert np.allclose(output_after_cls, output_before_cls, rtol=1e-2, atol=1e-5)
        assert len(cls_set_infos) == 8

    def test_cle(self):
        np.random.seed(0)
        model = models_for_tests.my_model_with_bns()
        fold_all_batch_norms_to_weight(model.model)
        input_shape = (2, 10, 24, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)
        session = _build_session(model)
        output_before_cle = session.run(None, {'input': test_data})
        equalize_model(model)
        session = _build_session(model)
        output_after_cle = session.run(None, {'input': test_data})
        assert np.allclose(output_after_cle, output_before_cle, rtol=1e-2, atol=1e-5)

    def test_cle_conv1D_model(self):
        x = torch.randn((2, 10, 24))
        model = models_for_tests.BNAfterConv1d()
        models_for_tests.initialize_bn_params(model)
        model = models_for_tests._convert_to_onnx_no_fold(model, x)
        input_shape = (2, 10, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)
        session = _build_session(model)
        output_before_cle = session.run(None, {'input': test_data})
        equalize_model(model)
        output_after_cle = session.run(None, {'input': test_data})
        assert np.allclose(output_after_cle, output_before_cle, rtol=1e-2, atol=1e-5)

    def test_cle_transpose1D_model(self):
        x = torch.randn((2, 10, 24))
        model = models_for_tests.BNAfterConvTranspose1d()
        models_for_tests.initialize_bn_params(model)
        model = models_for_tests._convert_to_onnx_no_fold(model, x)
        input_shape = (2, 10, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)
        session = _build_session(model)
        output_before_cle = session.run(None, {'input': test_data})
        equalize_model(model)
        output_after_cle = session.run(None, {'input': test_data})
        assert np.allclose(output_after_cle, output_before_cle, rtol=1e-2, atol=1e-5)

    def test_cls_squeezenet(self, tmp_path):
        model = models_for_tests.squeezenet1_0(tmp_path)
        cls = CrossLayerScaling(model)
        cls_set_infos = cls.scale_model()
        # Squeezenet1_0 doesn't have any scalable sets
        assert not cls_set_infos


class TestHighBiasFold:
    """ Test methods for HighBiasFold"""

    def test_find_high_bias_fold(self):
        model_onnx = models_for_tests.my_model_with_bns()

        input_shape = (2, 10, 24, 24)
        test_data = np.random.randn(*input_shape).astype(np.float32)

        # Equalize ONNX
        conv_bn_pairs, bn_conv_pairs = fold_all_batch_norms_to_weight(model_onnx.model)
        bn_dict = {}

        replace_relu6_with_relu(model_onnx)

        convs = []
        for conv_bn in conv_bn_pairs:
            bn_dict[conv_bn[0].name] = conv_bn[1]
            convs.append(conv_bn[0])

        for bn_conv in bn_conv_pairs:
            bn_dict[bn_conv[1].name] = bn_conv[0]

        bias1 = copy.deepcopy(numpy_helper.to_array(ParamUtils.get_param(model_onnx.model, convs[1], 1)))

        cls = CrossLayerScaling(model_onnx)
        cls_set_info = cls.scale_model()
        cls_session = _build_session(model_onnx)
        hbf = HighBiasFold(model_onnx)
        hbf.bias_fold(cls_set_info, bn_dict)

        bias_new = numpy_helper.to_array(ParamUtils.get_param(model_onnx.model, convs[1], 1))

        assert not np.allclose(bias_new, bias1, rtol=1e-2, atol=1e-5)


def _build_session(model):
    """
    Build and return onnxruntime inference session
    :param providers: providers to execute onnxruntime
    """
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    session = InferenceSession(
        path_or_bytes=model.model.SerializeToString(),
        sess_options=sess_options,
        providers=['CPUExecutionProvider'],
    )
    return session