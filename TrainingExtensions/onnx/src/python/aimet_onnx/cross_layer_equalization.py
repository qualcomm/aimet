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

""" Cross Layer Equalization

Some terminology for this code.
CLS set: Set of layers (2 or 3) that can be used for cross-layer scaling
Layer groups: Groups of layers that are immediately connected and can be decomposed further into CLS sets
"""

from typing import Tuple, List, Union
import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from packaging import version

from aimet_common.utils import AimetLogger
from aimet_common.connected_graph.connectedgraph import get_ordered_ops
from aimet_common.cross_layer_equalization import GraphSearchUtils, CrossLayerScaling as CLS, ClsSetInfo, \
    HighBiasFold as HBF
from aimet_common import libpymo

from aimet_onnx.meta.connectedgraph import ConnectedGraph, WEIGHT_INDEX, BIAS_INDEX
from aimet_onnx.meta.operations import Op
from aimet_onnx.utils import transpose_tensor, ParamUtils, get_node_attribute, replace_relu6_with_relu
from aimet_onnx.batch_norm_fold import BNLayer, fold_all_batch_norms_to_weight

# pylint: disable=no-name-in-module, ungrouped-imports
if version.parse(onnx.__version__) >= version.parse("1.14.0"):
    from onnx import NodeProto, ModelProto
else:
    from onnx.onnx_pb import NodeProto, ModelProto

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

ClsSet = Union[Tuple['Conv', 'Conv'],
               Tuple['Conv', 'Conv', 'Conv']]
ScaleFactor = Union[np.ndarray, Tuple[np.ndarray]]
cls_supported_layer_types = ['Conv', 'ConvTranspose']
cls_supported_activation_types = ['Relu', 'PRelu']


def get_ordered_list_of_conv_modules(list_of_starting_ops: List) -> List:
    """
    Finds order of nodes in graph
    :param list_of_starting_ops: list of starting ops for the model
    :return: List of names in graph in order
    """
    module_list = get_ordered_ops(list_of_starting_ops)
    module_list = [[module.dotted_name, module] for module in module_list if module.type in cls_supported_layer_types]
    return module_list


class CrossLayerScaling(CLS):
    """
    Scales a model's layers to equalize the weights between consecutive layers
    """
    def __init__(self, model: ModelProto):
        """
        :param model: ONNX model
        """
        super().__init__()
        self._model = model

    def scale_model(self) -> List[ClsSetInfo]:
        """
        Uses cross-layer scaling to scale all applicable layers in the given model

        :param model: Model to scale
        :return: CLS information for each CLS set
        """
        # Find layer groups
        connected_graph = ConnectedGraph(self._model)
        ordered_module_list = get_ordered_list_of_conv_modules(connected_graph.starting_ops)
        graph_search = GraphSearchUtils(connected_graph, ordered_module_list, cls_supported_layer_types,
                                        cls_supported_activation_types)
        layer_groups = graph_search.find_layer_groups_to_scale()

        # Find cls sets from the layer groups
        cls_sets = []
        for layer_group in layer_groups:
            cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
            cls_sets += cls_set

        # Scale the CLS sets
        scale_factors = self.scale_cls_sets(cls_sets)

        # Find if there were relu activations between layers of each cls set
        is_relu_activation_in_cls_sets = graph_search.is_relu_activation_present_in_cls_sets(cls_sets)

        # Convert to a list of cls-set-info elements
        cls_set_info_list = CrossLayerScaling.create_cls_set_info_list(cls_sets, scale_factors,
                                                                       is_relu_activation_in_cls_sets)

        return cls_set_info_list

    def _populate_libpymo_params(self, module: NodeProto,
                                 layer_param: libpymo.EqualizationParams):
        """
        Populates libpymo weight parameter
        """
        weight = ParamUtils.get_param(self._model.model, module, WEIGHT_INDEX)
        groups = get_node_attribute(module, "group")

        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if module.op_type == "ConvTranspose" and groups == 1:
            weight = transpose_tensor(weight, (1, 0, 2, 3))

        layer_param.weight = numpy_helper.to_array(weight).reshape(-1)
        weight_shape = get_weight_dimensions(np.array(weight.dims))
        layer_param.weightShape = weight_shape

    def _pack_params_for_conv(self,
                              cls_set,
                              prev_layer_params: libpymo.EqualizationParams,
                              curr_layer_params: libpymo.EqualizationParams):
        """
        Prepare and pack data structure for previous and current layer in given cls set.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        self._populate_libpymo_params(cls_set[0].get_module(), prev_layer_params)
        self._populate_libpymo_params(cls_set[1].get_module(), curr_layer_params)

        cls_set_0_bias = ParamUtils.get_param(self._model.model, cls_set[0].get_module(), BIAS_INDEX)
        if cls_set_0_bias is not None:
            prev_layer_params.bias = numpy_helper.to_array(cls_set_0_bias).reshape(-1)
        else:
            prev_layer_params.isBiasNone = True

    def _update_weight_for_layer_from_libpymo_obj(self, layer_param: libpymo.EqualizationParams,
                                                  module: NodeProto):
        """
        Update weight parameter from libpymo object
        """
        weight = ParamUtils.get_param(self._model.model, module, WEIGHT_INDEX)
        weight.raw_data = np.asarray(layer_param.weight, dtype=np.float32).tobytes()
        groups = get_node_attribute(module, "group")
        # Transpose weight back to original configuration
        if module.op_type == "ConvTranspose" and groups == 1:
            weight = transpose_tensor(weight, (1, 0, 2, 3))

        weight_param = ParamUtils.get_param(self._model.model, module, WEIGHT_INDEX)
        weight_param.raw_data = weight.raw_data

    def _update_params_for_conv(self,
                                cls_set,
                                prev_layer_params: libpymo.EqualizationParams,
                                curr_layer_params: libpymo.EqualizationParams):
        """
        Update weight and biases for cls set using updated data structures.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        self._update_weight_for_layer_from_libpymo_obj(prev_layer_params, cls_set[0].get_module())
        self._update_weight_for_layer_from_libpymo_obj(curr_layer_params, cls_set[1].get_module())

        if not prev_layer_params.isBiasNone:
            bias_param = ParamUtils.get_param(self._model.model, cls_set[0].get_module(),
                                              BIAS_INDEX)
            bias_param.raw_data = np.asarray(prev_layer_params.bias, dtype=np.float32).tobytes()

    def _pack_params_for_depthwise_conv(self, cls_set,
                                        prev_layer_params: libpymo.EqualizationParams,
                                        curr_layer_params: libpymo.EqualizationParams,
                                        next_layer_params: libpymo.EqualizationParams):
        """
        Prepare and pack data structure for previous, current and next layer in given cls set.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        :param next_layer_params: Data structure holding weight and bias for next layer in cls set.
        """

        self._populate_libpymo_params(cls_set[0].get_module(), prev_layer_params)

        assert cls_set[1].groups > 1

        weight = ParamUtils.get_param(self._model.model, cls_set[1].get_module(), WEIGHT_INDEX)
        curr_layer_params.weight = numpy_helper.to_array(weight).reshape(-1)
        curr_layer_params.weightShape = np.array(weight.dims)

        self._populate_libpymo_params(cls_set[2].get_module(), next_layer_params)

        cls_set_0_bias = ParamUtils.get_param(self._model.model, cls_set[0].get_module(), BIAS_INDEX)
        if cls_set_0_bias is not None:
            prev_layer_params.bias = numpy_helper.to_array(cls_set_0_bias).reshape(-1)
        else:
            prev_layer_params.isBiasNone = True

        cls_set_1_bias = ParamUtils.get_param(self._model.model, cls_set[1].get_module(), BIAS_INDEX)
        if cls_set_1_bias is not None:
            curr_layer_params.bias = numpy_helper.to_array(cls_set_1_bias).reshape(-1)
        else:
            curr_layer_params.isBiasNone = True

    def _update_params_for_depthwise_conv(self, cls_set,
                                          prev_layer_params: libpymo.EqualizationParams,
                                          curr_layer_params: libpymo.EqualizationParams,
                                          next_layer_params: libpymo.EqualizationParams):
        """
        Update weight and biases for cls set using updated data structures.

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        :param next_layer_params: Data structure holding weight and bias for next layer in cls set.
        """
        self._update_weight_for_layer_from_libpymo_obj(prev_layer_params, cls_set[0].get_module())
        self._update_weight_for_layer_from_libpymo_obj(curr_layer_params, cls_set[1].get_module())
        self._update_weight_for_layer_from_libpymo_obj(next_layer_params, cls_set[2].get_module())

        if not prev_layer_params.isBiasNone:
            bias_param = ParamUtils.get_param(self._model.model, cls_set[0].get_module(),
                                              BIAS_INDEX)
            bias_param.raw_data = np.asarray(prev_layer_params.bias, dtype=np.float32).tobytes()

        if not curr_layer_params.isBiasNone:
            bias_param = ParamUtils.get_param(self._model.model, cls_set[1].get_module(),
                                              BIAS_INDEX)
            bias_param.raw_data = np.asarray(curr_layer_params.bias, dtype=np.float32).tobytes()


class HighBiasFold(HBF):
    """
    Code to apply the high-bias-fold technique to a model
    """
    def __init__(self, model: ModelProto):
        self._model = model

    def _check_if_bias_is_none(self, layer: Op) -> bool:
        """ Returns if bias is a None for a layer. True if bias is None"""
        bias = ParamUtils.get_param(self._model.model, layer.get_module(), BIAS_INDEX)
        return not bias

    def _populate_bn_params_in_libpymo_obj(self, prev_layer_bn_params: libpymo.BNParamsHighBiasFold,
                                           bn_layer: BNLayer):
        """
        Populates BatchNorm params in the libpymo object
        :param prev_layer_bn_params: Data structure to pack batch norm parameter
        :param bn_layer: BatchNorm layer
        """
        prev_layer_bn_params.gamma = bn_layer.gamma
        prev_layer_bn_params.beta = bn_layer.beta

    def _pack_previous_and_current_layer_params(self, cls_pair_info: ClsSetInfo.ClsSetLayerPairInfo,
                                                prev_layer_params: libpymo.LayerParams,
                                                curr_layer_params: libpymo.LayerParams):
        """
        Helper method to pack information of previous and current layer.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param prev_layer_params: Data structure to pack previous layer parameters.
        :param curr_layer_params: Data structure to pack current layer parameters.
        """
        prev_layer_params.activationIsRelu = cls_pair_info.relu_activation_between_layers

        bias = ParamUtils.get_param(self._model.model, cls_pair_info.layer1.get_module(),
                                    BIAS_INDEX)

        prev_layer_params.bias = numpy_helper.to_array(bias).reshape(-1)

        module = cls_pair_info.layer2.get_module()
        weight = ParamUtils.get_param(self._model.model, module, WEIGHT_INDEX)
        bias = ParamUtils.get_param(self._model.model, module, BIAS_INDEX)

        groups = get_node_attribute(module, "group")

        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if module.op_type == "ConvTranspose" and groups == 1:
            weight = transpose_tensor(weight, (1, 0, 2, 3))

        curr_layer_params.bias = numpy_helper.to_array(bias).reshape(-1)
        curr_layer_params.weight = numpy_helper.to_array(weight).reshape(-1)
        curr_layer_params.weightShape = get_weight_dimensions(np.array(weight.dims))

    def _update_bias_for_layer_from_libpymo_obj(self, layer_param: libpymo.LayerParams,
                                                module: NodeProto):
        """
        Update bias parameter from libpymo object
        """
        bias = ParamUtils.get_param(self._model.model, module, BIAS_INDEX)

        bias.raw_data = np.asarray(layer_param.bias, dtype=np.float32).tobytes()

    def _update_previous_and_current_layer_bias(self, cls_pair_info: ClsSetInfo.ClsSetLayerPairInfo,
                                                prev_layer_params: libpymo.LayerParams,
                                                curr_layer_params: libpymo.LayerParams):
        """
        Update biases for previous and current layer.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param prev_layer_params: Data structure holding weight and bias for previous layer in cls set.
        :param curr_layer_params: Data structure holding weight and bias for current layer in cls set.
        """
        self._update_bias_for_layer_from_libpymo_obj(prev_layer_params, cls_pair_info.layer1.get_module())
        self._update_bias_for_layer_from_libpymo_obj(curr_layer_params, cls_pair_info.layer2.get_module())


def get_weight_dimensions(weight_shape: np.array) -> np.array:
    """
    Returns a length 4 weight shape
    :param weight_shape: shape of the weight tensor
    """
    dims = len(weight_shape)
    if dims == 4:
        return weight_shape
    return np.append(weight_shape, [1 for _ in range(4 - dims)]).astype(int)


def equalize_model(model: ModelProto):
    """
    High-level API to perform Cross-Layer Equalization (CLE) on the given model. The model is equalized in place.

    :param model: Model to equalize
    """
    if not isinstance(model, ONNXModel):
        model = ONNXModel(model)
    conv_bn_pairs, bn_conv_pairs = fold_all_batch_norms_to_weight(model)

    replace_relu6_with_relu(model)

    bn_dict = {}
    for conv_bn in conv_bn_pairs:
        bn_dict[conv_bn[0].name] = conv_bn[1]

    for bn_conv in bn_conv_pairs:
        bn_dict[bn_conv[1].name] = bn_conv[0]

    # perform cross-layer scaling on applicable layer sets
    cls = CrossLayerScaling(model)
    cls_set_info = cls.scale_model()

    # high-bias fold
    hbf = HighBiasFold(model)
    hbf.bias_fold(cls_set_info, bn_dict)
