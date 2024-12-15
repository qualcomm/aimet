# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
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
""" Batchnorm folding base """

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict, Iterable, Set, Any
import torch
import torch.nn
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d

from aimet_common.batch_norm_fold import batch_norm_fold, expand_shape_to_4d
from aimet_common.bias_correction import ConvBnPatternHandler, CONV_OP_TYPES, LINEAR_OP_TYPES, BN_OP_TYPES
from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.utils import AimetLogger

# pylint: disable=unused-import
from aimet_torch.defs import PassThroughOp
from aimet_torch import utils
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch._base.quantsim import _QuantizationSimModelInterface, _QuantizedModuleProtocol

_logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.BatchNormFolding)

LayerType = Union[
    torch.nn.Linear,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.ConvTranspose2d,
]
_supported_layers = LayerType.__args__

BatchNormType = Union[BatchNorm1d, BatchNorm2d]
_supported_batchnorms = BatchNormType.__args__

class _BatchNormFoldingNotSupported(RuntimeError):
    pass

class BatchNormFoldBase(ABC):
    """Handles batch norm folding logic"""
    @staticmethod
    def _call_batch_norm_fold(weight: torch.Tensor,
                                bias: torch.Tensor,
                                bn: Union[BatchNorm1d, BatchNorm2d],
                                fold_backward: bool):
        """
        BN fold without calling C++ APIs.

        :param weight: conv/linear weight
        :param bias: conv/linear bias
        :param bn: Batch Norm layer
        :param fold_backward: True if BatchNorm comes after Conv/Linear layer
        """
        with torch.no_grad():
            gamma = bn.weight.detach().cpu().numpy()
            beta = bn.bias.detach().cpu().numpy()
            mu = bn.running_mean.detach().cpu().numpy()
            sigma = torch.sqrt(bn.running_var + bn.eps).detach().cpu().numpy()

            _weight = weight.detach().cpu().numpy()
            _bias = bias.detach().cpu().numpy()

            _4d_shape = expand_shape_to_4d(_weight.shape)
            _weight, _bias = batch_norm_fold(_weight.reshape(_4d_shape), _bias, gamma, beta, mu, sigma, fold_backward)

            bias.copy_(torch.from_numpy(_bias).reshape_as(bias)).to(device=bias.device, dtype=bias.dtype)
            weight.copy_(torch.from_numpy(_weight).reshape_as(weight)).to(device=weight.device, dtype=weight.dtype)

    @classmethod
    def _fold_to_weight(cls, conv_linear: LayerType, bn: BatchNormType, fold_backward: bool):
        """
        Fold BatchNorm into the weight and bias of the given layer.

        :param conv_linear: Conv or linear layer to fold BN into.
        :param bn: BatchNorm to fold.
        """
        # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
        # However depthwise conv layers are always N, 1, H, W whether transposed-conv or not, so no need to transpose
        if isinstance(conv_linear, torch.nn.ConvTranspose2d) and conv_linear.groups == 1:
            conv_linear.weight.data = conv_linear.weight.data.permute(1, 0, 2, 3)

        if conv_linear.bias is None:
            out_channels = conv_linear.out_features if isinstance(conv_linear, torch.nn.Linear)\
                        else conv_linear.out_channels
            bias = torch.zeros(out_channels,
                            device=conv_linear.weight.device,
                            dtype=conv_linear.weight.dtype)
            conv_linear.bias = torch.nn.Parameter(bias)


        cls._call_batch_norm_fold(conv_linear.weight, conv_linear.bias, bn, fold_backward=fold_backward)

        # Transpose weight back to N, C, H, W for transposed Conv2D, for non-depthwise layers
        if isinstance(conv_linear, torch.nn.ConvTranspose2d) and conv_linear.groups == 1:
            conv_linear.weight.data = conv_linear.weight.data.permute(1, 0, 2, 3)


    @classmethod
    def fold_given_batch_norms(cls, model, layer_pairs):
        """
        Fold a given set of batch_norm layers into conv layers

        :param model: Model
        :param layer_pairs: Pairs of conv and batch_norm layers to use for folding
        :return: None
        """
        # pylint: disable=protected-access
        conv_bn_pairs = []
        bn_conv_pairs = []

        for x, y in layer_pairs:
            if cls._is_batchnorm(x):
                assert cls._is_conv_linear(y)
                bn = x
                conv = y
                bn_conv_pairs.append((bn, conv))
            else:
                assert cls._is_conv_linear(x)
                assert cls._is_batchnorm(y)
                conv = x
                bn = y
                conv_bn_pairs.append((conv, bn))

        cls._fold_given_batch_norms(model, conv_bn_pairs, bn_conv_pairs)

    @classmethod
    def _is_batchnorm(cls, module: torch.nn.Module) -> bool:
        return isinstance(module, _supported_batchnorms)

    @classmethod
    def _is_conv_linear(cls, module: torch.nn.Module) -> bool:
        return isinstance(module, _supported_layers)

    @classmethod
    def fold_all_batch_norms_to_weight(
            cls,
            model: torch.nn.Module,
            input_shapes: Union[Tuple, List[Tuple]],
            dummy_input: Union[torch.Tensor, Tuple] = None
    ) -> List[Tuple[LayerType, BatchNormType]]:
        """
        Fold all batch_norm layers in a model into the weight of the corresponding conv layers

        :param model: Model
        :param input_shapes: Input shapes for the model (can be one or multiple inputs)
        :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors
        :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
        """
        if isinstance(model, torch.nn.DataParallel):
            return cls.fold_all_batch_norms_to_weight(model.module, input_shapes, dummy_input)
        device = utils.get_device(model)
        if dummy_input is None:
            inp_tensor_list = utils.create_rand_tensors_given_shapes(input_shapes, device)
        else:
            inp_tensor_list = dummy_input
        connected_graph = ConnectedGraph(model, inp_tensor_list)

        conv_bn_pairs, bn_conv_pairs, bn_to_fold = cls._find_all_batch_norms_to_fold(connected_graph)

        cls._fold_given_batch_norms(model, conv_bn_pairs, bn_conv_pairs)

        # Convert the standalone BNs which are not folded
        bn_converted = cls.convert_standalone_batchnorms(model, inp_tensor_list, bn_to_fold)
        _logger.debug("Total %d standalone BatchNorms' weights got converted", len(bn_converted))
        return conv_bn_pairs + [(conv, bn) for bn, conv in bn_conv_pairs]

    @classmethod
    def convert_standalone_batchnorms(cls,
                                      model: torch.nn.Module,
                                      dummy_input: Union[torch.Tensor, Tuple],
                                      folded_bn: set) -> List[Tuple[Any, BatchNorm2d]]:
        """
        Convert the weights of all the standalone batchnorms of a model which didn't get folded.
        :param model: torch model for which batch norm folding is being performed
        :param dummy_input: dummy input for the model
        :param folded_bn: list of BNs which got folded
        :return: List of tuple(name, bn_module) whose weights got converted
        """

        module_list = utils.get_ordered_list_of_modules(model, dummy_input)
        bn_converted = []
        for name, module in module_list:
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)) and module not in folded_bn:
                cls.convert_batchnorm_parameters(model, module)
                _logger.debug("%s weights got converted", name)
                bn_converted.append((name, module))
        return bn_converted


    @staticmethod
    def convert_batchnorm_parameters(model: torch.nn.Module, bn: Union[torch.nn.BatchNorm1d, torch.nn.BatchNorm2d]):
        """
        To convert the weight of a batchnorm such that it becomes in the format y = weights*input + bias
        :param model: torch model for which batch norm folding is being performed
        :param bn: BatchNorm module whose weights needs to be converted
        """
        with utils.in_eval_mode(model), torch.no_grad():
            gamma = bn.weight
            beta = bn.bias
            running_mean = bn.running_mean
            inv_sigma = torch.rsqrt(bn.running_var + bn.eps)

            weight = gamma * inv_sigma
            bias = beta - running_mean * weight

            # Update the values
            bn.eps = 0
            bn.track_running_stats = False
            bn.weight.copy_(weight.clone().detach())
            bn.bias.copy_(bias.clone().detach())
            bn.running_mean = torch.zeros(bn.running_mean.shape, device=bn.running_mean.device, dtype=bn.running_mean.dtype)
            bn.running_var = torch.ones(bn.running_var.shape, device=bn.running_var.device, dtype=bn.running_var.dtype)

    @classmethod
    def find_all_conv_bn_with_activation(cls, model: torch.nn.Module, input_shape: Tuple) -> Dict:
        """
        Uses searcher to find preceding and next bn layers for a conv/linear layer
        :param model: PyTorch model
        :param input_shape: shape of input to the model
        :return: dictionary of conv/linear layers with associated bn op / activation info
        """
        device = utils.get_device(model)
        inp_tensor_list = utils.create_rand_tensors_given_shapes(input_shape, device)
        connected_graph = ConnectedGraph(model, inp_tensor_list)
        return cls.find_all_conv_bn_with_activation_in_graph(connected_graph)

    @classmethod
    def find_all_conv_bn_with_activation_in_graph(cls, connected_graph: ConnectedGraph) -> Dict:
        """
        Uses searcher to find preceding and next bn layers for a conv/linear layer
        :param connected_graph: ConnectedGraph object.
        :return: dictionary of conv/linear layers with associated bn op / activation info
        """

        # initialize all patterns to be matched and associated call back functions
        patterns_with_callbacks = []
        layer_select_handler = ConvBnPatternHandler()
        conv_types = ['Conv1d', 'Conv', 'ConvTranspose']
        linear_types = ['Gemm']

        for op_type in conv_types + linear_types:
            patterns_with_callbacks.append(PatternType(pattern=['BatchNormalization', op_type],
                                                    action=layer_select_handler))
            patterns_with_callbacks.append(PatternType(pattern=[op_type, 'BatchNormalization'],
                                                    action=layer_select_handler))
        patterns_with_callbacks.append(PatternType(pattern=['Conv3d', 'BatchNorm3d'], action=layer_select_handler))
        patterns_with_callbacks.append(PatternType(pattern=['BatchNorm3d', 'Conv3d'], action=layer_select_handler))

        # create graph searcher instance with connected graph and patterns to search
        graph_searcher = GraphSearcher(connected_graph, patterns_with_callbacks)

        # get all conv/linear and bn info
        graph_searcher.find_all_patterns_in_graph_apply_actions()
        convs_bn_activation_dict = layer_select_handler.get_conv_linear_bn_info_dict()

        return convs_bn_activation_dict

    @classmethod
    def find_standalone_batchnorm_ops(cls, connected_graph: ConnectedGraph)->set:
        """
        Find all batchnorms ops can not be folded.
        :param connected_graph: Connected graph associated with the model.
        :return stand_alone_bn_ops: Set of batchnorm ops can not be folded.
        """
        _, _, bn_picked_for_folding = cls._find_foldable_bn_pair_and_bn_picked_for_folding(connected_graph)
        bn_ops = {op for op in connected_graph.get_all_ops().values() if op.type in BN_OP_TYPES}
        stand_alone_bn_ops = bn_ops - bn_picked_for_folding

        return stand_alone_bn_ops

    @staticmethod
    def _delete_bn_from_model(model: torch.nn.Module, bn_layer_list: Iterable[BatchNormType]):
        utils.replace_modules_with_instances_of_new_type(model, bn_layer_list, torch.nn.Identity)

    @classmethod
    def _find_all_batch_norms_to_fold(cls, connected_graph: ConnectedGraph) -> Tuple[
            List[Tuple[LayerType, BatchNormType]], List[Tuple[BatchNormType, LayerType]]]:
        """
        Find all possible batch norm layers that can be folded. And returns a list of pairs such that (bn, layer)
        means bn will be forward-folded into layer and (layer, bn) means bn will be backward-folded into layer
        :param connected_graph: Connected graph associated with the model.
        :return: A list of (layer, bn) pairs and a list of (bn, layer) pairs,
                where `bn` can be folded into to `layer`.
        """
        conv_bn_pairs, bn_conv_pairs, bn_to_fold = cls._find_foldable_bn_pair_and_bn_picked_for_folding(connected_graph)
        return conv_bn_pairs, bn_conv_pairs, bn_to_fold

    @classmethod
    def find_all_batch_norms_to_fold(cls, model, input_shapes, dummy_input: Union[torch.Tensor, Tuple] = None):
        """
        Find all possible batch norm layers that can be folded. And returns a list of pairs such that (bn, layer)
        means bn will be forward-folded into layer and (layer, bn) means bn will be backward-folded into layer
        :param model: Model to search
        :param input_shapes: Input shapes to use for the model (can be one or multiple inputs)
        :param dummy_input: A dummy input to the model. Can be a Tensor or a Tuple of Tensors
        :return: List of pairs of bn and layers to fold bn into
        """
        device = utils.get_device(model)
        if dummy_input is not None:
            connected_graph = ConnectedGraph(model, dummy_input)
        else:
            device = utils.get_device(model)
            inp_tensor_list = utils.create_rand_tensors_given_shapes(input_shapes, device)
            connected_graph = ConnectedGraph(model, inp_tensor_list)

        conv_bn_pairs, bn_conv_pairs, _ = cls._find_all_batch_norms_to_fold(connected_graph)
        return conv_bn_pairs + bn_conv_pairs


    @staticmethod
    def _is_valid_bn_fold(conv: LayerType, fold_backward: bool) -> bool:
        """
        Determine if a given layer can successfully absorb a BatchNorm given the layer type and parameters
        :param conv: The Conv/Linear layer to fold a BatchNorm into.
        :param fold_backward: True if BatchNorm comes after Conv/Linear layer
        :return: True if a BatchNorm layer can be folded without causing output error.
        """
        valid = True
        if not fold_backward:
            # Cannot fold BN -> Conv with padding. AIMET does not support forward folding to grouped or DW Conv
            if isinstance(conv, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Conv3d)):
                valid &= all(item == 0 for item in conv.padding)
                valid &= conv.groups == 1
            # AIMET does not support forward folding to ConvTranspose
            elif isinstance(conv, torch.nn.ConvTranspose2d):
                valid = False
        else:
            # AIMET does not support backwards folding to grouped ConvTranspose
            if isinstance(conv, torch.nn.ConvTranspose2d):
                valid &= conv.groups in (1, conv.in_channels)
        return valid

    @classmethod
    def _find_foldable_bn_pair_and_bn_picked_for_folding(cls, connected_graph: ConnectedGraph) -> Tuple[
            List[Tuple[LayerType, BatchNormType]], List[Tuple[BatchNormType, LayerType]], Set]:
        """
        Find all possible batch norm layers that can be folded. And returns a list of pairs such that (bn, layer)
        means bn will be forward-folded into layer and (layer, bn) means bn will be backward-folded into layer
        :param connected_graph: Connected graph associated with the model.
        :return: A list of (layer, bn) pairs and a list of (bn, layer) pairs,
                where `bn` can be folded into to `layer`.
                A set of bn ops which can be folded in to immediate convs.
        """
        conv_linear_bn_activation_info_dict = cls.find_all_conv_bn_with_activation_in_graph(connected_graph)

        # To mark BN's already picked for backward folding
        bn_picked_for_folding = set()

        _conv_linear_optypes = CONV_OP_TYPES + LINEAR_OP_TYPES
        ordered_conv_fc_modules = [op.get_module() for op in connected_graph.ordered_ops if op.type in _conv_linear_optypes]

        conv_bn_pairs = []
        # Backward fold is given priority over Forward fold
        for module in ordered_conv_fc_modules:
            if module in conv_linear_bn_activation_info_dict.keys() and cls._is_valid_bn_fold(module, True):
                bn_info = conv_linear_bn_activation_info_dict[module]
                # print(bn_info)
                if bn_info.output_bn and bn_info.output_bn not in bn_picked_for_folding:
                    conv_bn_pairs.append((module, bn_info.output_bn.get_module()))
                    bn_picked_for_folding.add(bn_info.output_bn)

        bn_conv_pairs = []
        for module in ordered_conv_fc_modules:
            if module in conv_linear_bn_activation_info_dict.keys() and cls._is_valid_bn_fold(module, False):
                bn_info = conv_linear_bn_activation_info_dict[module]
                if bn_info.input_bn and bn_info.input_bn not in bn_picked_for_folding:
                    bn_conv_pairs.append((bn_info.input_bn.get_module(), module))
                    bn_picked_for_folding.add(bn_info.input_bn)

        return conv_bn_pairs, bn_conv_pairs, bn_picked_for_folding

    @classmethod
    @abstractmethod
    def fold_all_batch_norms_to_scale(
            cls, sim: _QuantizationSimModelInterface,
    ) -> List[Tuple[_QuantizedModuleProtocol, _QuantizedModuleProtocol]]:
        """
        Fold all batch_norm layers in a model into the quantization scale parameter
        of the corresponding conv layers

        :param sim: QuantizationSimModel
        :return: A list of pairs of layers [(Conv/Linear, BN layer that got folded)]
        """

    @classmethod
    @abstractmethod
    def _fold_given_batch_norms(cls, model,
                                conv_bn_pairs: Iterable[Tuple[torch.nn.Module, torch.nn.Module]],
                                bn_conv_pairs: Iterable[Tuple[torch.nn.Module, torch.nn.Module]]):
        """
        Fold a given set of batch_norm layers into conv layers

        :param model: Model
        :param conv_bn_pairs: List of (conv, bn) pairs to fold
        :param bn_conv_pairs: List of (bn, conv) pairs to fold
        :return: None
        """

    @classmethod
    @abstractmethod
    def _fold_to_scale(cls, conv_wrapper: _QuantizedModuleProtocol, bn_wrapper: _QuantizedModuleProtocol):
        """
        Fold BatchNorm into the scale and bias of the given layer.

        :param conv_wrapper: _QuantizedModuleProtocol that wraps conv or linear layer.
        :param bn_wrapper: _QuantizedModuleProtocol that wraps bn.
        """
