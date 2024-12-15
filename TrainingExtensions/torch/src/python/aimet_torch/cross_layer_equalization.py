# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2019-2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=too-many-lines

from typing import Tuple, List, Union, Dict
import numpy as np
import torch

from aimet_common.utils import AimetLogger
from aimet_common.cross_layer_equalization import ClsLayerType, ClsSetInfo, ClsImpl, HbfImpl
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.utils import (get_device, get_ordered_list_of_modules, create_rand_tensors_given_shapes,
                               place_model)

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.CrosslayerEqualization)

ClsSet = Union[Tuple[torch.nn.Conv2d, torch.nn.Conv2d],
               Tuple[torch.nn.Conv2d, torch.nn.Conv2d, torch.nn.Conv2d]]

ClsSupportedLayer = Union[torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d]

ScaleFactor = Union[np.ndarray, Tuple[np.ndarray]]

cls_supported_layers = (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Conv1d, torch.nn.ConvTranspose1d)
cls_supported_activations = (torch.nn.ReLU, torch.nn.PReLU)


def get_ordered_list_of_conv_modules(model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple]) -> List:
    """
    Finds order of nodes in graph
    :param model: model
    :param dummy_input: Dummy input to the model. Used to parse model graph.
    :return: List of names in graph in order
    """
    module_list = get_ordered_list_of_modules(model, dummy_input)
    module_list = [[name, module] for name, module in module_list if isinstance(module, cls_supported_layers)]
    return module_list


class GraphSearchUtils:
    """
    Code to search a model graph to find nodes to use for cross-layer-scaling and high-bias-fold
    """

    def __init__(self, model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]],
                 dummy_input: Union[torch.Tensor, List[torch.Tensor]] = None):
        """

        :param model: PyTorch model.
        :param input_shapes: Input shape for the model (can be one or multiple inputs)
        :param dummy_input: Dummy input to the model. Used to parse model graph. dummy_input is expected to be placed
         on the same device as model.
        """
        if dummy_input is None:
            inp_tensor_list = tuple(utils.create_rand_tensors_given_shapes(input_shapes, get_device(model)))
        else:
            inp_tensor_list = dummy_input
        self._connected_graph = ConnectedGraph(model, inp_tensor_list)
        self._ordered_module_list = get_ordered_list_of_conv_modules(model, inp_tensor_list)


    @staticmethod
    def find_downstream_layer_groups_to_scale(op, layer_groups, current_group=None, visited_nodes=None):
        """
        Recursive function to find cls layer groups downstream from a given op
        :param op: Starting op to search from
        :param layer_groups: Running list of layer groups
        :param current_group: Running current layer group
        :param visited_nodes: Running list of visited nodes (to short-circuit recursion)
        :return: None
        """

        if not visited_nodes:
            visited_nodes = []
        if not current_group:
            current_group = []

        if op in visited_nodes:
            return
        visited_nodes.append(op)
        # print("Visiting node: {}".format(op.dotted_name))

        # If current node is Conv2D, add to the current group
        if op.model_module and isinstance(op.model_module.get_module(), cls_supported_layers):
            current_group.append(op.model_module.get_module())

        # Terminating condition for current group
        if not op.model_module or not isinstance(op.model_module.get_module(),
                                                 cls_supported_layers + cls_supported_activations):

            if (len(current_group) > 1) and (current_group not in layer_groups):
                layer_groups.append(current_group)
            current_group = []

        if op.output:
            for consumer in op.output.consumers:
                GraphSearchUtils.find_downstream_layer_groups_to_scale(consumer, layer_groups,
                                                                       current_group, visited_nodes)

        # Reached a leaf.. See if the current group has something to grab
        if (len(current_group) > 1) and (current_group not in layer_groups):
            layer_groups.append(current_group)

    @staticmethod
    def convert_layer_group_to_cls_sets(layer_group):
        """
        Helper function to convert a layer group to a list of cls sets
        :param layer_group: Given layer group to generate cls sets
        :return: List of cls sets

        Supported layer combinations for CLS are:
        1. Conv + Conv
        2. DepthwiseConv + Conv
        3. Conv + DepthwiseConv + Conv

        Can be rewritten as,
        Conv
            -> Conv
            -> DepthwiseConv
                -> Conv
        DepthwiseConv
            -> Conv

        If a combination is partially supported, the cls_set is completely omitted and restarted from the next
        supported layer
        For example: Consider Conv + DepthwiseConv + Depthwise(unsupported)
        - Since Depthwise(unsupported) is the last layer encountered, we need to omit all the three layers and restart
        the cls sets from the next supported layer.

        """

        # pylint: disable=too-many-branches
        def convert_to_cls_layer_type(layer: ClsSupportedLayer) -> Tuple[ClsLayerType, ClsSupportedLayer]:
            """
            Given the layer, check if its supported in CLS
            :param layer: layer to check
            :return: Tuple of ClsLayerType and the layer
            """
            if layer.groups == 1:
                layer_type = ClsLayerType.Conv
            elif layer.groups == layer.in_channels and layer.in_channels == layer.out_channels:
                # depthwiseConv layer with depth multiplier = 1
                layer_type = ClsLayerType.DepthwiseConv
            else:
                layer_type = ClsLayerType.Unsupported
            return layer_type, layer

        def get_next_layer() -> Union[Tuple[ClsLayerType, Union[ClsSupportedLayer, None]]]:
            """
            :return: Tuple of ClsLayerType and the next layer in layer_group
            """
            if not layer_group:
                return ClsLayerType.Unsupported, None
            layer = layer_group.pop(0)
            return convert_to_cls_layer_type(layer)

        cls_sets = []

        first_layer_to_scale = (ClsLayerType.Unsupported, None)
        while layer_group:
            while layer_group and first_layer_to_scale[0] is ClsLayerType.Unsupported:
                first_layer_to_scale = get_next_layer()
                if first_layer_to_scale[0] is ClsLayerType.Unsupported:
                    logger.info('Layer %s is not supported. Ignoring for cls', first_layer_to_scale[1])

            second_layer_to_scale = get_next_layer()
            if first_layer_to_scale[0] == ClsLayerType.Conv:
                if second_layer_to_scale[0] == ClsLayerType.Conv:
                    cls_sets.append((first_layer_to_scale[1], second_layer_to_scale[1]))
                    first_layer_to_scale = second_layer_to_scale
                elif second_layer_to_scale[0] == ClsLayerType.DepthwiseConv:
                    if layer_group:
                        # do not pop third layer yet, determine its type and then pop it
                        third_layer_to_scale = convert_to_cls_layer_type(layer_group[0])
                        if third_layer_to_scale[0] == ClsLayerType.Conv:
                            cls_sets.append(
                                (first_layer_to_scale[1], second_layer_to_scale[1], third_layer_to_scale[1]))
                            # adding third_layer_to_scale for the next round of CLS set determination
                            first_layer_to_scale = get_next_layer()
                        else:
                            # unsupported combination encountered
                            first_layer_to_scale = second_layer_to_scale
                else:
                    logger.info('Layer %s is not supported. Ignoring for cls', second_layer_to_scale[1])
                    first_layer_to_scale = (ClsLayerType.Unsupported, None)
            elif first_layer_to_scale[0] == ClsLayerType.DepthwiseConv:
                if second_layer_to_scale[0] == ClsLayerType.Conv:
                    cls_sets.append((first_layer_to_scale[1], second_layer_to_scale[1]))
                first_layer_to_scale = second_layer_to_scale
            else:
                logger.info('Layer %s is not supported. Ignoring for cls', first_layer_to_scale[1])
                first_layer_to_scale = second_layer_to_scale

        return cls_sets

    def find_layer_groups_to_scale(self) -> List[List[torch.nn.Conv2d]]:
        """
        :return: List of groups of layers. Each group can be independently equalized
        """

        # Find the input node(s) in the graph
        input_nodes = []
        for op in self._connected_graph.get_all_ops().values():
            if op.inputs and op.inputs[0].is_model_input:
                input_nodes.append(op)

        layer_groups = []
        for op in input_nodes:
            self.find_downstream_layer_groups_to_scale(op, layer_groups)

        # Sort the layer groups in order of occurrence in the model
        ordered_layer_groups = []
        for _, module in self._ordered_module_list:
            for layer_group in layer_groups:
                if layer_group[0] is module:
                    ordered_layer_groups.append(layer_group)

        return ordered_layer_groups

    @staticmethod
    def does_module_have_relu_activation(connected_graph: ConnectedGraph, module: torch.nn.Module) -> bool:
        """
        Finds if a given module has a ReLU activation
        :param connected_graph: Reference to ConnectedGraph instance
        :param module: PyTorch module to find activation for
        :return: True if module has a relu activation
        """

        for op in connected_graph.get_all_ops().values():

            if op.model_module and op.model_module.get_module() is module:
                assert len(op.output.consumers) == 1
                is_relu_activation = isinstance(op.output.consumers[0].model_module.get_module(),
                                                (torch.nn.ReLU, torch.nn.PReLU))
                return is_relu_activation

        return False

    def is_relu_activation_present_in_cls_sets(self, cls_sets: List[ClsSet]):
        """
        :param cls_sets: CLS sets to find relu activations in
        :return: List of groups of layers. Each group can be independently equalized
        """

        is_relu_activation_in_cls_sets = []
        for cls_set in cls_sets:

            # We need to check activation functions for all layers but the last one in the set
            # Because we are only interested in checking activation functions between the layers we will scale
            cls_set = cls_set[:-1]

            is_relu_activation_in_cls_set = ()
            for module in cls_set:
                is_relu_activation_in_cls_set += (self.does_module_have_relu_activation(self._connected_graph,
                                                                                        module), )

            if len(is_relu_activation_in_cls_set) == 1:
                is_relu_activation_in_cls_set = is_relu_activation_in_cls_set[0]

            is_relu_activation_in_cls_sets.append(is_relu_activation_in_cls_set)

        return is_relu_activation_in_cls_sets


class CrossLayerScaling:
    """
    Code to apply the cross-layer-scaling technique to a model
    """

    @staticmethod
    def scale_cls_sets(cls_sets: List[ClsSet]) -> List[ScaleFactor]:
        """
        Scale multiple CLS sets

        :param cls_sets: List of CLS sets
        :return: Scaling factors calculated and applied for each CLS set in order
        """
        scale_factor_list = []
        for cls_set in cls_sets:
            scale_factor = CrossLayerScaling.scale_cls_set(cls_set)
            scale_factor_list.append(scale_factor)
        return scale_factor_list

    @staticmethod
    def scale_cls_set(cls_set: ClsSet) -> ScaleFactor:
        """
        Scale a CLS set
        :param cls_set: Either a pair or regular conv layers or a triplet of depthwise separable layers
        :return: Scaling factor calculated and applied
        """
        if len(cls_set) == 3:
            scale_factor = CrossLayerScaling.scale_cls_set_with_depthwise_layers(cls_set)
        else:
            scale_factor = CrossLayerScaling.scale_cls_set_with_conv_layers(cls_set)

        return scale_factor

    @classmethod
    def scale_cls_set_with_conv_layers(cls, cls_set: ClsSet) -> np.ndarray:
        """
        API to invoke equalize layer params (update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized
        :return: Scaling factor S_12 for each conv layer pair: numpy array
        """
        on_gpu = False
        for module in cls_set:
            if not isinstance(module, cls_supported_layers):
                raise ValueError(f"Only Conv or Transposed Conv layers are supported for cross layer equalization."
                                 f" Layer class {str(module.__class__)} is not supported.")
            if module.weight.is_cuda:
                on_gpu = True
                module.cpu()

        cls_impl = PythonClsImpl()
        scaling_factor = cls_impl.scale_cls_set_with_conv_layers(cls_set)

        if on_gpu:
            for module in cls_set:
                module.to(device="cuda")

        return scaling_factor

    @classmethod
    def scale_cls_set_with_depthwise_layers(cls, cls_set: ClsSet) -> [np.ndarray, np.ndarray]:
        """
        API to invoke equalize layer params for depth wise separable layers(update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers whose weights and biases need to be equalized.
                        Second Conv layer is a depth-wise conv and third conv layer is point-wise conv
        :return: Scaling factors S_12 and S_23 : numpy arrays
        """
        on_gpu = False
        for module in cls_set:
            if not isinstance(module, cls_supported_layers):
                raise ValueError(f"Only Conv or Transposed Conv layers are supported for cross layer equalization."
                                 f" Layer class {str(module.__class__)} is not supported.")
            if module.weight.is_cuda:
                on_gpu = True
                module.cpu()

        cls_impl = PythonClsImpl()
        scaling_factors = cls_impl.scale_cls_set_with_depthwise_layers(cls_set)

        if on_gpu:
            for module in cls_set:
                module.to(device="cuda")

        return scaling_factors

    @staticmethod
    def create_cls_set_info_list(cls_sets: List[ClsSet], scale_factors: List[ScaleFactor],
                                 is_relu_activation_in_cls_sets):
        """
        Binds information from there separate lists into one [ClsInfoSet] data-structure
        :param cls_sets: List of CLS sets
        :param scale_factors: Scale-factors for each cls-set
        :param is_relu_activation_in_cls_sets: Information if there is relu activation in each cls-set
        :return: List of ClsSetInfo
        """
        cls_set_info_list = []
        assert len(cls_sets) == len(scale_factors) == len(is_relu_activation_in_cls_sets)

        for index, cls_set in enumerate(cls_sets):

            if isinstance(scale_factors[index], tuple):
                # If we are dealing with a triplet of layers, then we should have 2 scale factors and 2 relu flags
                # Assert that this is true
                assert len(cls_set) == 3
                assert len(scale_factors[index]) == len(is_relu_activation_in_cls_sets[index]) == 2

                cls_pair_1 = ClsSetInfo.ClsSetLayerPairInfo(cls_set[0], cls_set[1], scale_factors[index][0],
                                                            is_relu_activation_in_cls_sets[index][0])
                cls_pair_2 = ClsSetInfo.ClsSetLayerPairInfo(cls_set[1], cls_set[2], scale_factors[index][1],
                                                            is_relu_activation_in_cls_sets[index][1])

                cls_set_info = ClsSetInfo(cls_pair_1, cls_pair_2)

            else:
                cls_pair = ClsSetInfo.ClsSetLayerPairInfo(cls_set[0], cls_set[1], scale_factors[index],
                                                          is_relu_activation_in_cls_sets[index])

                cls_set_info = ClsSetInfo(cls_pair)

            cls_set_info_list.append(cls_set_info)

        return cls_set_info_list

    @staticmethod
    def scale_model(model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]] = None,
                    dummy_input: Union[torch.Tensor, List[torch.Tensor]] = None) -> List[ClsSetInfo]:
        """
        Uses cross-layer scaling to scale all applicable layers in the given model

        :param model: Model to scale
        :param input_shapes: Input shape for the model (can be one or multiple inputs)
        :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors. dummy_input will be
         placed on CPU if not already.
        :return: CLS information for each CLS set
        """
        if isinstance(model, torch.nn.DataParallel):
            return CrossLayerScaling.scale_model(model.module, input_shapes, dummy_input=dummy_input)

        # The use of input_shapes will be removed in the future release. It is maintained now for backward compatibility.
        if input_shapes and dummy_input is None:
            dummy_input = create_rand_tensors_given_shapes(input_shapes, torch.device('cpu'))
        if input_shapes is None and dummy_input is None:
            raise ValueError("Both input_shapes and dummy_input can't be None")

        # Place model and dummy input on the cpu.
        with place_model(model, torch.device("cpu")):
            dummy_input = utils.change_tensor_device_placement(dummy_input, device=torch.device('cpu'))

            # Find layer groups
            graph_search = GraphSearchUtils(model, input_shapes, dummy_input=dummy_input)
            layer_groups = graph_search.find_layer_groups_to_scale()

            # Find cls sets from the layer groups
            cls_sets = []
            for layer_group in layer_groups:
                cls_set = GraphSearchUtils.convert_layer_group_to_cls_sets(layer_group)
                cls_sets += cls_set

            # Scale the CLS sets
            scale_factors = CrossLayerScaling.scale_cls_sets(cls_sets)

            # Find if there were relu activations between layers of each cls set
            is_relu_activation_in_cls_sets = graph_search.is_relu_activation_present_in_cls_sets(cls_sets)

            # Convert to a list of cls-set-info elements
            cls_set_info_list = CrossLayerScaling.create_cls_set_info_list(cls_sets, scale_factors,
                                                                           is_relu_activation_in_cls_sets)
        return cls_set_info_list


class PythonClsImpl(ClsImpl):
    """
    This class implements the CLS algorithm using Python version while following the base Implementation interface.
    """
    def scale_cls_set_with_depthwise_layers(self, cls_set) -> [np.ndarray, np.ndarray]:
        """
        API to invoke equalize layer params for depth wise separable layers(update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers whose weights and biases need to be equalized.
                        Second Conv layer is a depth-wise conv and third conv layer is point-wise conv
        :return: Scaling factors S_12 and S_23 : numpy arrays
        """
        weight_0 = self._prepare_params(cls_set[0])
        assert cls_set[1].groups > 1
        weight_1 = self._prepare_params(cls_set[1])
        weight_2 = self._prepare_params(cls_set[2])
        weight_0 = weight_0.numpy()
        weight_1 = weight_1.numpy()
        weight_2 = weight_2.numpy()

        bias_0 = None
        if cls_set[0].bias is not None:
            bias_0 = cls_set[0].bias.detach().cpu().numpy()
        bias_1 = None
        if cls_set[1].bias is not None:
            bias_1 = cls_set[1].bias.detach().cpu().numpy()

        # compute scaling factors and folded parameters.
        s_12, s_23 = self.compute_scaling_params_for_depthwise_conv(weight_0, weight_1, weight_2)
        _weight_0, _weight_1, _weight_2, _bias_0, _bias_1 = (
            self.fold_scaling_params_for_depthwise_conv(weight_0, weight_1, weight_2, bias_0, bias_1, s_12, s_23))

        with torch.no_grad():
            self._restore_params(cls_set[0], torch.from_numpy(_weight_0))
            self._restore_params(cls_set[1], torch.from_numpy(_weight_1))
            self._restore_params(cls_set[2], torch.from_numpy(_weight_2))

            if cls_set[0].bias is not None:
                cls_set[0].bias.copy_(torch.from_numpy(_bias_0).reshape_as(cls_set[0].bias)).to(device=cls_set[0].bias.device,
                                                                                                dtype=cls_set[0].bias.dtype)
            if cls_set[1].bias is not None:
                cls_set[1].bias.copy_(torch.from_numpy(_bias_1).reshape_as(cls_set[1].bias)).to(device=cls_set[1].bias.device,
                                                                                                dtype=cls_set[1].bias.dtype)
        return s_12, s_23

    def scale_cls_set_with_conv_layers(self, cls_set) -> np.ndarray:
        """
        API to invoke equalize layer params for regular conv layers (update for weights and bias is in place)

        :param cls_set: Consecutive Conv layers Tuple whose weights and biases need to be equalized
        :return: Scaling factor S_12 for each conv layer pair: numpy array
        """
        weight_0 = self._prepare_params(cls_set[0])
        weight_1 = self._prepare_params(cls_set[1])
        weight_0 = weight_0.numpy()
        weight_1 = weight_1.numpy()

        bias_0 = None
        if cls_set[0].bias is not None:
            bias_0 = cls_set[0].bias.detach().cpu().numpy()

        # compute scaling factors and folded parameters.
        scale_factor = self.compute_scaling_params_for_conv(weight_0, weight_1)
        _weight_0, _weight_1, _bias_0 = (
            self.fold_scaling_params_for_conv(weight_0, weight_1, bias_0, scale_factor))

        with torch.no_grad():
            self._restore_params(cls_set[0], torch.from_numpy(_weight_0))
            self._restore_params(cls_set[1], torch.from_numpy(_weight_1))
            if cls_set[0].bias is not None:
                cls_set[0].bias.copy_(torch.from_numpy(_bias_0).reshape_as(cls_set[0].bias)).to(device=cls_set[0].bias.device,
                                                                                                dtype=cls_set[0].bias.dtype)
        return scale_factor

    @staticmethod
    def _transpose_tensor(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """
        During preparation:
        For TransposeConv2d, Transpose tensor in the common format [Noc, Nin, Kh, Kw].
        For TransposeConv1d, Transpose tensor in common format [Noc, Nin, K].

        During restoration:
        For TransposeConv2d, Transpose tensor in the original format [Nin, Noc, Kh, Kw].
        For TransposeConv1d, Transpose tensor in back in original format [Nin, Noc, K].

        :param module: Module.
        :param tensor: Input tensor.
        :return: Output tensor.
        """
        if isinstance(module, torch.nn.ConvTranspose2d) and module.groups == 1:
            tensor = tensor.permute(1, 0, 2, 3).contiguous()

        if isinstance(module, torch.nn.ConvTranspose1d) and module.groups == 1:
            tensor = tensor.permute(1, 0, 2).contiguous()
        return tensor

    @staticmethod
    def _make_4d_tensor(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """
        Return 4 dimensional tensor by adding a dimension on the end if the tensor is not 4d.

        :param module: Module.
        :param tensor: Input tensor.
        :return: Output tensor.
        """
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            assert len(tensor.shape) == 3, "Module should have 3d weight tensor."
            tensor = torch.unsqueeze(tensor, dim=-1)
        return tensor

    def _prepare_params(self, module: torch.nn.Module) -> torch.Tensor:
        """
        Prepare weight parameters for CLS.

        :param module: PyTorch module.
        :return: Prepared weight.
        """
        weight = module.weight.detach().cpu()
        weight = self._transpose_tensor(module, weight)
        weight = self._make_4d_tensor(module, weight)
        return weight

    def _restore_params(self, module: torch.nn.Module, tensor: torch.Tensor):
        """
        Restore the weight parameters.

        :param module: PyTorch module.
        :param tensor: updated parameters.
        """
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            tensor = torch.squeeze(tensor, dim=-1)

        _weight_0 = self._transpose_tensor(module, tensor)
        module.weight.copy_(_weight_0.reshape_as(module.weight)).to(device=module.weight.device,
                                                                    dtype=module.weight.dtype)


class HighBiasFold:
    """
    Code to apply the high-bias-fold technique to a model
    """

    ActivationIsReluForFirstModule = bool
    ScaleForFirstModule = np.ndarray

    @classmethod
    def bias_fold(cls, cls_set_info_list: List[ClsSetInfo],
                  bn_layers: Dict[Union[torch.nn.Conv2d, torch.nn.ConvTranspose2d], torch.nn.BatchNorm2d]):
        """
        Folds bias values greater than 3 * sigma to next layer's bias

        :param cls_set_info_list: List of info elements for each cls set
        :param bn_layers: Key: Conv/Linear layer Value: Corresponding folded BN layer
        :return: None
        """
        if not bn_layers:
            logger.info('High Bias folding is not supported for models without BatchNorm Layers')
            return

        for cls_set_info in cls_set_info_list:
            for cls_pair_info in cls_set_info.cls_pair_info_list:

                if (cls_pair_info.layer1.bias is None) or (cls_pair_info.layer2.bias is None) or \
                        (cls_pair_info.layer1 not in bn_layers):
                    continue

                hbf_impl = PythonHbfImpl()
                hbf_impl.bias_fold(cls_pair_info, bn_layers)


class PythonHbfImpl(HbfImpl):
    """
    This class implements the HBF algorithm using python version while following the base Implementation interface.
    """
    # pylint: disable=no-self-use
    def bias_fold(self, cls_pair_info, bn_layers):
        """
        Bias fold implementation using python version.

        :param cls_pair_info: Layer pairs that were scaled using CLS and related information.
        :param bn_layers: Dictionary with Key being Conv/Linear layer and value being corresponding folded BN layer.
        """
        weight = cls_pair_info.layer2.weight.detach().cpu()
        if isinstance(cls_pair_info.layer2, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            weight = torch.unsqueeze(weight, dim=-1)
        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        if isinstance(cls_pair_info.layer2, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)) and \
                cls_pair_info.layer2.groups == 1:
            weight = weight.permute(1, 0, 2, 3)
        weight = weight.numpy()

        activation_is_relu = cls_pair_info.relu_activation_between_layers

        beta = bn_layers[cls_pair_info.layer1].bias.detach().cpu().numpy() / cls_pair_info.scale_factor
        gamma = bn_layers[cls_pair_info.layer1].weight.detach().cpu().numpy() / cls_pair_info.scale_factor

        bias_prev_layer = cls_pair_info.layer1.bias.detach().cpu().numpy()
        bias_curr_layer = cls_pair_info.layer2.bias.detach().cpu().numpy()

        # Absorb high biases
        _bias_prev_layer, _bias_curr_layer = (
            self._absorb_bias(activation_is_relu, beta, gamma, weight, bias_curr_layer, bias_prev_layer))

        with torch.no_grad():
            cls_pair_info.layer1.bias.copy_(torch.from_numpy(_bias_prev_layer).reshape_as(cls_pair_info.layer1.bias)).to(
                device=cls_pair_info.layer1.bias.device, dtype=cls_pair_info.layer1.bias.dtype)
            cls_pair_info.layer2.bias.copy_(torch.from_numpy(_bias_curr_layer).reshape_as(cls_pair_info.layer2.bias)).to(
                device=cls_pair_info.layer2.bias.device, dtype=cls_pair_info.layer2.bias.dtype)

def equalize_model(model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]] = None,
                   dummy_input: Union[torch.Tensor, Tuple] = None):
    """
    High-level API to perform Cross-Layer Equalization (CLE) on the given model. The model is equalized in place.

    :param model: Model to equalize
    :param input_shapes: Shape of the input (can be a tuple or a list of tuples if multiple inputs)
    :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors. dummy_input will be
     placed on CPU if not already.
    """
    if isinstance(model, torch.nn.DataParallel):
        equalize_model(model.module, input_shapes, dummy_input)
    else:
        # The use of input_shapes will be removed in the future release. It is maintained now for backward compatibility.
        if input_shapes and dummy_input is None:
            dummy_input = create_rand_tensors_given_shapes(input_shapes, torch.device('cpu'))
        if input_shapes is None and dummy_input is None:
            raise ValueError("Both input_shapes and dummy_input can't be None")

        # Place model and dummy input on the cpu.
        with place_model(model, torch.device('cpu')):
            dummy_input = utils.change_tensor_device_placement(dummy_input, device=torch.device('cpu'))

            # fold batchnorm layers and perform CLE on the folded model.
            folded_pairs = fold_all_batch_norms(model, input_shapes, dummy_input=dummy_input)
            equalize_bn_folded_model(model, input_shapes, folded_pairs,
                                     dummy_input=dummy_input)


def equalize_bn_folded_model(model: torch.nn.Module,
                             input_shapes: Union[Tuple, List[Tuple]],
                             folded_pairs: List[Tuple[torch.nn.Module, torch.nn.BatchNorm2d]],
                             dummy_input: Union[torch.Tensor, Tuple] = None
                             ):
    """
    Perform Cross-Layer Scaling (CLS) and High Bias Folding (HBF) on a batchnorm-folded model.
    The model is equalized in place.

    :param model: Batchnorm-folded model to equalize
    :param input_shapes: Shape of the input (can be a tuple or a list of tuples if multiple inputs)
    :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors. dummy_input will be
     placed on CPU if not already.
    :param folded_pairs: List of pairs of folded layers
    """
    if isinstance(model, torch.nn.DataParallel):
        equalize_bn_folded_model(model.module, input_shapes, folded_pairs, dummy_input=dummy_input)
    else:
        bn_dict = {}
        for conv_bn in folded_pairs:
            bn_dict[conv_bn[0]] = conv_bn[1]

        with place_model(model, torch.device('cpu')):
            # replace any ReLU6 layers with ReLU
            utils.replace_modules_of_type1_with_type2(model, torch.nn.ReLU6, torch.nn.ReLU)

            # perform cross-layer scaling on applicable layer sets
            cls_set_info_list = CrossLayerScaling.scale_model(model, input_shapes, dummy_input=dummy_input)

            # high-bias fold
            HighBiasFold.bias_fold(cls_set_info_list, bn_dict)
