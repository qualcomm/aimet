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

""" Code to perform bias correction for layers """

from typing import Callable, Tuple, List, Union, Dict
import copy
import torch
import torch.nn
import numpy as np

from aimet_common.graph_pattern_matcher import PatternType
from aimet_common.graph_searcher import GraphSearcher
from aimet_common.utils import AimetLogger
from aimet_common.bias_correction import (
    ConvBnInfoType,
    ConvBnPatternHandler,
    analytical_bias_correction,
    empirical_bias_correction
)
from aimet_common.defs import ActivationType

from aimet_torch import utils
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch._base.quantsim import _QuantizedModuleProtocol, QuantParams
from aimet_torch.meta.connectedgraph import ConnectedGraph
from aimet_torch.save_utils import SaveUtils
from aimet_torch.utils import get_ordered_lists_of_conv_fc
from aimet_torch.v2.nn.base import BaseQuantizationMixin

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.Quant)

class StopForwardException(Exception):
    """ Dummy exception to early-terminate forward-pass """


def forward_pass(model: torch.nn.Module, batch: torch.Tensor):
    """
    forward pass depending model allocation on CPU / GPU till StopForwardException
    :param model: model
    :param batch: batch
    :return: Nothing
    """
    # first check if the model is on GPU or not
    if utils.is_model_on_gpu(model):
        batch = batch.cuda()
    try:
        with utils.in_eval_mode(model), torch.no_grad():
            _ = model(batch)
    except StopForwardException:
        pass


def get_quantized_dequantized_weight(layer: torch.nn.Module) -> torch.Tensor:
    """
    Gets quantized dequantized weights of a layer
    :param layer: Conv/FC layer
    :return: quantized dequantized weights
    """
    # pylint: disable=protected-access
    weight_tensor = layer.weight
    weight_quantizer = layer.param_quantizers['weight']

    # v2 quantizer
    if callable(weight_quantizer):
        return weight_quantizer(weight_tensor)

    # v1 quantizer
    return weight_quantizer.quantize_dequantize(weight_tensor, weight_quantizer.round_mode)


def register_fwd_hook_for_layer(layer: torch.nn.Module, hook: Callable) -> torch.utils.hooks.RemovableHandle:
    """
    register forward hook for given layer
    :param layer: layer
    :param hook: hook function
    :return: hook handle
    """
    hook_handle = layer.register_forward_hook(hook)
    return hook_handle


def get_output_data(layer: torch.nn.Module, model: torch.nn.Module, images_in_one_batch: torch.Tensor) -> np.ndarray:
    """
    Function to get output values of a layer
    :param layer: layer
    :param model: model
    :param images_in_one_batch
    :return: list of output of layer for all batches of images
    """
    def _hook_to_collect_output_data(_, __, out_data):
        """
        hook to collect output data
        """
        out_data = utils.to_numpy(out_data)
        orig_layer_out_data.append(out_data)
        raise StopForwardException

    hook_handles = list()
    orig_layer_out_data = list()

    # register forward hooks
    hook_handles.append(register_fwd_hook_for_layer(layer, _hook_to_collect_output_data))

    # forward pass for 1 batch for model
    forward_pass(model, images_in_one_batch)
    output_data = np.vstack(orig_layer_out_data)

    # delete list entries used for hooks
    del orig_layer_out_data[:]

    # remove hook handles
    for hook_handle in hook_handles:
        hook_handle.remove()

    return output_data


def call_empirical_correct_bias(layer: torch.nn.Module,
                                reference_outputs: np.ndarray,
                                quantized_outputs: np.ndarray):
    """
    Empirical bias correction using python.

    :param layer:
    :param reference_outputs:
    :param quantized_outputs:
    """
    bias = layer.bias.detach().cpu().numpy()
    _bias = empirical_bias_correction(reference_outputs, quantized_outputs, bias)

    # update the bias param.
    with torch.no_grad():
        layer.bias.copy_(torch.from_numpy(_bias).reshape_as(layer.bias)).to(device=layer.bias.device,
                                                                            dtype=layer.bias.dtype)


def call_analytical_correct_bias(layer: torch.nn.Module,
                                 bn: Union[torch.nn.BatchNorm2d, None],
                                 activation_type: Union[ActivationType, None]):
    """
    Analytical bias correction using python.

    :param layer: Layer to be corrected
    :param bn: Input BN to layer
    :param activation_type: Input activation to layer
    """
    quant_dequant_weight = get_quantized_dequantized_weight(layer)
    layer = layer.get_original_module()

    # Transpose weights to C, N, H, W from N, C, H, W since axis are flipped for transposed conv
    weight_tensor = layer.weight
    if isinstance(layer, torch.nn.ConvTranspose2d) and layer.groups == 1:
        weight_tensor = weight_tensor.permute(1, 0, 2, 3)
        quant_dequant_weight = quant_dequant_weight.permute(1, 0, 2, 3)

    quant_dequant_weight = quant_dequant_weight.detach().cpu().numpy()
    weight_tensor = weight_tensor.detach().cpu().numpy()
    bias_tensor = layer.bias.detach().cpu().numpy()

    # Assigning activation to No Activation
    activation = ActivationType.no_activation
    if bn is None:
        shape = weight_tensor.shape[1]
        gamma = np.ones(shape)
        beta = np.zeros(shape)
    else:
        gamma = bn.get_module().weight.detach().cpu().numpy()
        beta = bn.get_module().bias.detach().cpu().numpy()

        if activation_type == ActivationType.relu:
            activation = ActivationType.relu
        elif activation_type == ActivationType.relu6:
            activation = ActivationType.relu6

    _bias = analytical_bias_correction(weight_tensor, quant_dequant_weight, bias_tensor, beta, gamma, activation)

    # update the bias param.
    with torch.no_grad():
        layer.bias.copy_(torch.from_numpy(_bias).reshape_as(layer.bias)).to(device=layer.bias.device,
                                                                            dtype=layer.bias.dtype)


def correct_bias(model: torch.nn.Module, quant_params: QuantParams,
                 num_quant_samples: int, data_loader, num_bias_correct_samples: int,
                 conv_bn_dict: Union[Dict[torch.nn.Module, ConvBnInfoType], None] = None,
                 perform_only_empirical_bias_corr: bool = True,
                 layers_to_ignore: List[torch.nn.Module] = None):
    """
    Corrects bias for each Conv layer of model (unless ignored). A combination of Analytical and Empirical Bias
    Correction is used i.e. all the layers which can be corrected using Analytical Bias Correction are corrected
    using Analytical Bias Correction and remaining layers are corrected using Empirical method.

    Returns an in-place corrected floating point model

    :param model: Model to be corrected
    :param quant_params: Named tuple for quantization simulation for bias correction
    :param num_quant_samples: number of samples of images to pass through quantization sim for bias correction.
    :param data_loader: data loader for the model
    :param num_bias_correct_samples: number of samples for Bias correction
    :param conv_bn_dict: Dict of conv and bn with information related to activation. If None, the function calc it
    :param perform_only_empirical_bias_corr: Default True. If true will perform only empirical Bias Corr for all layers
           irrespective of the fact that layer is eligible for Analytical Bias Corr.
    :param layers_to_ignore: list of layer names for which we need to skip bias correction.
    """
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements, too-many-nested-blocks, no-else-continue
    if layers_to_ignore is None:
        layers_to_ignore = []

    # Find batch size and shape of input tensor
    batch_size, input_shape = utils.get_input_shape_batch_size(data_loader)

    # Rounding up number of samples to batch size
    n_batches_bias_correction = int(np.ceil(num_bias_correct_samples / batch_size))
    n_batches_quantization = int(np.ceil(num_quant_samples / batch_size))

    data_loader_n_samples_bias_corr = utils.IterFirstX(data_loader, n_batches_bias_correction)
    data_loader_n_samples_quant = utils.IterFirstX(data_loader, n_batches_quantization)

    # TODO: Remove wrapper function
    # Create a wrapping function for data loader for quantization
    def pass_data_through_model(model, early_stopping_iterations=None, use_cuda=False):
        # pylint: disable=unused-argument
        # forward pass for given number of batches for model
        for (images_in_one_batch, *_) in data_loader_n_samples_quant:
            forward_pass(model, images_in_one_batch)

    ordered_conv_linear_nodes = get_ordered_lists_of_conv_fc(model, input_shape)

    if conv_bn_dict is None:
        conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape)

    # Create a copy of the model as reference model
    model_copy = copy.deepcopy(model)

    # Add bias for all the layers whose bias is None
    for _, module in ordered_conv_linear_nodes:
        if module.bias is None:
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                output_size = module.out_channels
            elif isinstance(module, torch.nn.Linear):
                output_size = module.out_features
            module.bias = torch.nn.Parameter(torch.zeros(output_size))
            module.bias.data = module.bias.data.to(device=module.weight.device)

    # Quantize full model
    dummy_tensors = utils.create_rand_tensors_given_shapes(input_shape, utils.get_device(model))
    q = QuantizationSimModel(model=model, quant_scheme=quant_params.quant_scheme,
                             rounding_mode=quant_params.round_mode,
                             default_output_bw=quant_params.act_bw,
                             default_param_bw=quant_params.weight_bw,
                             in_place=True,
                             dummy_input=dummy_tensors, config_file=quant_params.config_file)

    # make sure  model got updated in-place before we use it for bc updates
    assert q.model is model

    def disable_output_quantizers(qmodule):
        # v2 quantized module
        if isinstance(qmodule, BaseQuantizationMixin):
            qmodule._remove_output_quantizers() # pylint: disable=protected-access
            return

        # v1 QcQuantizeWrapper
        for qtzr in qmodule.output_quantizers:
            qtzr.enabled = False

    # updates to skip_output_activation and layers_to_ignore
    for _, module in model.named_modules():
        # Skip all layer's output quantization
        if isinstance(module, _QuantizedModuleProtocol): # pylint: disable=protected-access
            disable_output_quantizers(module)

    q.compute_encodings(pass_data_through_model, None)

    # For first conv layer, perform analytical bc if perform_only_empirical_bias_corr is set to False
    # and layer is not marked to be ignored during bc.
    if not perform_only_empirical_bias_corr:
        module_name, module = ordered_conv_linear_nodes[0]
        if module not in layers_to_ignore:
            logger.info('Correcting layer %s using Analytical Bias Correction', module_name)
            quantize_layer = utils.get_layer_by_name(model, module_name)
            call_analytical_correct_bias(quantize_layer, None, None)
            logger.info('Corrected bias for the layer')
            ordered_conv_linear_nodes.pop(0)

    for module_name, module in ordered_conv_linear_nodes:
        # Ignore all layers which are skipped by user
        if module in layers_to_ignore:
            continue
        else:
            # Analytical Bias Correction is only done for Conv layers
            reference_layer = utils.get_layer_by_name(model_copy, module_name)
            quantize_layer = utils.get_layer_by_name(model, module_name)

            if module in conv_bn_dict.keys():
                bn_layer_info = conv_bn_dict[module]
                if perform_only_empirical_bias_corr or bn_layer_info is None or bn_layer_info.input_bn is None:
                    # Get output from quantized model and reference model
                    reference_outputs = []
                    quantized_outputs = []
                    for images_in_one_batch, *_ in data_loader_n_samples_bias_corr:
                        reference_output_batch = get_output_data(reference_layer, model_copy, images_in_one_batch)
                        quantized_model_output_batch = get_output_data(quantize_layer, model, images_in_one_batch)

                        if isinstance(reference_layer, torch.nn.Linear):
                            extended_shape = np.concatenate((reference_output_batch.shape, np.array([1, 1])))
                            reference_output_batch = reference_output_batch.reshape(extended_shape)
                            quantized_model_output_batch = quantized_model_output_batch.reshape(extended_shape)

                        reference_outputs.append(reference_output_batch)
                        quantized_outputs.append(quantized_model_output_batch)

                    reference_outputs = np.concatenate(reference_outputs)
                    quantized_outputs = np.concatenate(quantized_outputs)
                    logger.info('Correcting layer %s using Empirical Bias Correction', module_name)
                    call_empirical_correct_bias(module, reference_outputs, quantized_outputs)
                else:
                    logger.info('Correcting layer %s using Analytical Bias Correction', module_name)
                    call_analytical_correct_bias(quantize_layer, bn_layer_info.input_bn,
                                                 bn_layer_info.in_activation_type)
                logger.info('Corrected bias for the layer')

    SaveUtils.remove_quantization_wrappers(model)

    logger.info('Completed bias correction')


def find_all_conv_bn_with_activation(model: torch.nn.Module, input_shape: Tuple) -> Dict:
    """
    Uses searcher to find preceding and next bn layers for a conv/linear layer
    :param model: PyTorch model
    :param input_shape: shape of input to the model
    :return: dictionary of conv/linear layers with associated bn op / activation info
    """

    activation_types = ['Relu', 'Clip']

    # initialize all patterns to be matched and associated call back functions
    patterns_with_callbacks = []
    layer_select_handler = ConvBnPatternHandler()
    patterns_with_callbacks.append(PatternType(pattern=['BatchNormalization', 'Conv'],
                                               action=layer_select_handler))

    patterns_with_callbacks.append(PatternType(pattern=['BatchNormalization', 'ConvTranspose'],
                                               action=layer_select_handler))

    patterns_with_callbacks.append(PatternType(pattern=['Conv'],
                                               action=layer_select_handler))

    patterns_with_callbacks.append(PatternType(pattern=['Gemm'],
                                               action=layer_select_handler))

    for activation in activation_types:
        patterns_with_callbacks.append(PatternType(pattern=['BatchNormalization', activation, 'Conv'],
                                                   action=layer_select_handler))

        patterns_with_callbacks.append(PatternType(pattern=['BatchNormalization', activation, 'ConvTranspose'],
                                                   action=layer_select_handler))

    device = utils.get_device(model)
    connected_graph = ConnectedGraph(model, (torch.rand(input_shape).to(device),))

    # create graph searcher instance with connected graph and patterns to search
    graph_searcher = GraphSearcher(connected_graph, patterns_with_callbacks)

    # get all conv/linear and bn info
    graph_searcher.find_all_patterns_in_graph_apply_actions()
    convs_bn_activation_dict = layer_select_handler.get_conv_linear_bn_info_dict()

    return convs_bn_activation_dict
