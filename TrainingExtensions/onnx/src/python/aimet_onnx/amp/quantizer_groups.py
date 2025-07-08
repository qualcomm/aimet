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

"""Find quantizer groups in a model"""

import itertools
from typing import Dict, Tuple, List
from collections import defaultdict
from dataclasses import dataclass, field

from aimet_common.connected_graph.operation import Op

from aimet_common.amp.utils import CANDIDATE_WITH_DTYPE

from aimet_common.amp.quantizer_groups import (
    QuantizerGroupBase,
    get_supported_candidates_for_quantizers,
    compute_baseline_candidate_options,
)
from aimet_common.utils import AimetLogger

from aimet_onnx.meta.connectedgraph import ConnectedGraph, Product
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.qc_quantize_op import QcQuantizeOp

logger = AimetLogger.get_area_logger(AimetLogger.LogAreas.MixedPrecision)


@dataclass(frozen=True)
class QuantizerGroup(QuantizerGroupBase):
    """
    Group of modules and quantizers
    """

    parameter_quantizers: Tuple[str, ...] = field(default_factory=tuple)
    activation_quantizers: Tuple[str, ...] = field(default_factory=tuple)

    def get_candidate(self, name_to_quantizer_dict: Dict) -> CANDIDATE_WITH_DTYPE:
        """
        Gets Activation & parameter bitwidth

        :param name_to_quantizer_dict: Gets module from module name
        :return: Tuple of Activation, parameter bitwidth and data type
        """
        activation_bw, parameter_bw = None, None
        activation_dtype, param_dtype = None, None

        for quantizer in self.get_activation_quantizers(name_to_quantizer_dict):
            activation_bw = quantizer.bitwidth
            activation_dtype = quantizer.data_type
            break

        for quantizer in self.get_param_quantizers(name_to_quantizer_dict):
            if quantizer.enabled:
                parameter_bw = quantizer.bitwidth
                param_dtype = quantizer.data_type
                break

        return (activation_bw, activation_dtype), (parameter_bw, param_dtype)

    def set_quantizers_to_candidate(
        self, name_to_quantizer_dict: Dict, candidate: CANDIDATE_WITH_DTYPE
    ):
        """
        Sets a quantizer group to a given candidate bitwidth

        :param name_to_quantizer_dict: Gets module from module name
        :param candidate: candidate with act and param bw and data types
        """
        (activation_bw, activation_dtype), (param_bw, param_dtype) = candidate

        for quantizer in self.get_activation_quantizers(name_to_quantizer_dict):
            quantizer.set_bitwidth(activation_bw)
            quantizer.data_type = activation_dtype

        for quantizer in self.get_param_quantizers(name_to_quantizer_dict):
            quantizer.set_bitwidth(param_bw)
            quantizer.data_type = param_dtype

    def to_list(self) -> List[Tuple[str, str]]:
        """
        Converts quantizer group to a list

        :return: List containing input/output quantizers & weight quantizers
        """
        return list(
            itertools.chain(
                (
                    ("activation", module_name)
                    for module_name in self.activation_quantizers
                ),
                (("weight", module_name) for module_name in self.parameter_quantizers),
            )
        )

    def get_active_quantizers(self, name_to_quantizer_dict) -> List[QcQuantizeOp]:
        """
        Find all active tensor quantizers associated with this quantizer group

        :param name_to_quantizer_dict: Gets module from module name
        :return: List of active quantizers
        """
        quantizers = self.get_activation_quantizers(
            name_to_quantizer_dict
        ) + self.get_param_quantizers(name_to_quantizer_dict)
        return [quantizer for quantizer in quantizers if quantizer.enabled]

    def get_activation_quantizers(self, name_to_quantizer_dict):
        """
        Gets activation quantizers

        :param name_to_quantizer_dict: Gets module from module name

        :return List of activation quantizers
        """
        result = []
        for module_name in self.activation_quantizers:
            quantizer = name_to_quantizer_dict[module_name]
            result.append(quantizer)
        return result

    def get_param_quantizers(self, name_to_quantizer_dict):
        """
        Gets parameter quantizers

        :param name_to_quantizer_dict: Gets module from module name

        :return List of parameter quantizers
        """
        result = []
        for module_name in self.parameter_quantizers:
            quantizer = name_to_quantizer_dict[module_name]
            result.append(quantizer)
        return result


op_types_to_ignore = ["Reshape", "branch", "Gather", "Unsqueeze", "Pad", "Transpose"]
ops_not_to_traverse = ["Shape"]


def find_quantizer_group(sim: QuantizationSimModel):
    """
    Create quantizer groups following these rules:

    1) All quantized tensors exist in exactly 1 quantizer group
    2) A parameter's quantizer group contains all other tensors which feed into all ops that the parameter feeds into
    3) Any quantizer group must not be decomposable into multiple quantizer groups that still follow rules 1 and 2

    Note that two activations feeding into the same binary op would not fall into the same quantizer group using the above
    definition, while an activation and parameter feeding into a binary op would be in the same group
    """
    quantized_tensors = {
        name for name, quantizer in sim.qc_quantize_op_dict.items() if quantizer.enabled
    }
    visited_tensors = set()
    quantizer_groups = []

    for tensor_name in quantized_tensors:
        # Avoid re-creating duplicate quantizer groups
        if tensor_name in visited_tensors:
            continue

        # Get all tensors belonging to the same group
        # TODO: Derive op_types_to_ignore from config file
        related_tensors = _get_related_quantizers(
            tensor_name, quantized_tensors, sim.connected_graph, op_types_to_ignore
        )

        visited_tensors |= related_tensors

        # Use ConnectedGraph to determine which tensors are parameters vs. activations
        parameters = {
            tensor
            for tensor in related_tensors
            if sim.connected_graph.get_all_products()[tensor].is_parm
        }
        activations = related_tensors - parameters

        quantizer_group = QuantizerGroup(
            tuple(sorted(parameters)), tuple(sorted(activations))
        )
        logger.debug("Quantizer Group added: %s", quantizer_group)
        quantizer_groups.append(quantizer_group)

    return sim.qc_quantize_op_dict, quantizer_groups


def _get_related_quantizers(
    tensor: str,
    quantized_tensors: set[str],
    connected_graph: ConnectedGraph,
    pass_through_op_types: List[str],
):
    """
    Get all tensors for which the valid configurations depend on the configuration of `tensor`.

    Dependant tensors are all inputs to all ops that consume `tensor`, and all tensors which depend on these tensors.
    """
    tensor_queue = [tensor]
    related_quantized_tensors = set()
    visited_ops = set()

    while tensor_queue:
        name = tensor_queue.pop(0)
        product: Product = connected_graph.get_all_products()[name]

        # Find all ops that consume this tensor
        consumers = _get_tensor_consumers(product, pass_through_op_types)

        # Ignore already-visited ops
        consumers -= visited_ops
        visited_ops |= consumers

        # For any consumer which has a quantized parameter, add all inputs to the quantizer group
        input_tensors = {name}
        for op in consumers:
            if any(name in quantized_tensors for name in op.parameters.keys()):
                input_tensors |= set(
                    t.name for t in _get_op_input_tensors(op, pass_through_op_types)
                )

        # Only look at quantized tensors which we haven't visited
        input_tensors = (input_tensors & quantized_tensors) - related_quantized_tensors
        related_quantized_tensors |= input_tensors

        # Add newly found tensors to tensor_queue
        for item in input_tensors:
            if item not in tensor_queue:
                tensor_queue.append(item)

    return related_quantized_tensors


def _get_op_input_tensors(op: Op, pass_through_op_types: List[str]) -> List[Product]:
    """Get all input tensors to `op`, traversing through ops of type `pass_through_op_types`"""
    inputs = []
    for inp in op.inputs:
        # Pass through ops which don't have output quantizers if necessary
        while inp.producer and inp.producer.type in pass_through_op_types:
            inp = inp.producer.inputs[0]
        inputs.append(inp)

    return inputs


def _get_tensor_consumers(product: Product, pass_through_op_types: List[str]):
    """Get all consumers of `product`, traversing through ops of type `pass_through_op_types`"""
    consumers = set()
    for consumer in product.consumers:
        if consumer.type in pass_through_op_types:
            consumers |= {
                op
                for output in consumer.outputs
                for op in _get_tensor_consumers(output, pass_through_op_types)
            }
        else:
            consumers.add(consumer)

    return consumers


def find_supported_candidates(
    quantizer_groups: List[QuantizerGroup],
    amp_candidates: List[CANDIDATE_WITH_DTYPE],
    supported_kernels: Dict,
    quantizer_to_op_type: Dict,
    use_all_amp_candidates: bool,
) -> Tuple[Dict, List]:
    """
    Computes 1. a list of supported candidates per Quantizer and 2. List of candidate options for max_candidate
    :param quantizer_groups: List of quantizer groups computed for the given model
    :param amp_candidates: List of candidates specified by the user to be used for the AMP algorithm
    :param supported_kernels: Dict of supported kernels for a given op/defaults specified in the config file
    :param quantizer_to_op_type: Dict of quantizers to onnx op type
    :param use_all_amp_candidates: Boolean value representing whether the unsupported candidates in the
    "candidates" list need to be considered for creating the output lists. If set to True, all the AMP candidates are
    directly used for all the Quantizers, else the candidates per Quantizers are computed.
    """

    quantizers_with_supported_candidates = defaultdict(list)

    for quantizer_group in quantizer_groups:
        quantizers = sorted(
            set(
                itertools.chain(
                    quantizer_group.activation_quantizers,
                    quantizer_group.parameter_quantizers,
                )
            )
        )

        supported_kernels_for_quantizers = get_supported_candidates_for_quantizers(
            quantizers,
            quantizer_to_op_type,
            supported_kernels,
            amp_candidates,
            use_all_amp_candidates,
        )

        quantizers_with_supported_candidates[quantizer_group] = (
            supported_kernels_for_quantizers.copy()
        )

    max_candidate_options = compute_baseline_candidate_options(
        quantizers_with_supported_candidates, amp_candidates, use_all_amp_candidates
    )

    return quantizers_with_supported_candidates, max_candidate_options
