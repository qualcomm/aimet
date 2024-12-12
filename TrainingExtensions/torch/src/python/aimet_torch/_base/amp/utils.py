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
""" Utilities for mixed precision feature in aimet_torch.v1 """

from typing import Union, Tuple, List, Dict
from collections import defaultdict
import torch

from aimet_common.amp.utils import CANDIDATE_WITH_DTYPE, get_effective_bitwidth
from aimet_torch.cost_calculator import CostCalculator
from aimet_torch.layer_database import LayerDatabase
from aimet_torch._base.amp.quantizer_groups import QuantizerGroup
from aimet_torch.v2.nn.base import BaseQuantizationMixin


def find_bit_ops_reduction(quantizer_group: QuantizerGroup, mac_dict: Dict, max_candidate: CANDIDATE_WITH_DTYPE, candidate: CANDIDATE_WITH_DTYPE) -> int:
    """
    Find bit ops reduction when Bitwidth changes from max_candidate to candidate
    :param quantizer_group: Quantizer group for which we want to find bit ops reduction
    :param mac_dict: Dictionary mapping modules to mac counts
    :param max_candidate: Default starting bitwidth
    :param candidate: Current BW candidate
    :return: Bit ops reduction
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    quantizer_group_dict = create_quantizer_module_dict(quantizer_group)
    bit_ops_reduction = 0

    if len(max_candidate) == 1:
        (act_bw_max, act_max_dtype), = max_candidate
        param_bw_max, param_max_dtype = None, None
    else:
        (act_bw_max, act_max_dtype), (param_bw_max, param_max_dtype) = max_candidate

    if len(candidate) == 1:
        (act_bw, act_dtype), = candidate
        param_bw, param_dtype = None, None
    else:
        (act_bw, act_dtype), (param_bw, param_dtype) = candidate

    act_effective_bw_max = get_effective_bitwidth(act_max_dtype, act_bw_max)
    if param_bw_max is not None:
        param_effective_bw_max = get_effective_bitwidth(param_max_dtype, param_bw_max)
    act_effective_bw = get_effective_bitwidth(act_dtype, act_bw)
    if param_bw is not None:
        param_effective_bw = get_effective_bitwidth(param_dtype, param_bw)

    if 'output' in quantizer_group_dict and 'weight' in quantizer_group_dict:
        for module_name in quantizer_group_dict['weight']:
            if module_name in mac_dict:
                if param_bw_max is not None and param_bw is not None:
                    bit_ops_reduction = bit_ops_reduction - mac_dict[module_name] * act_effective_bw * param_effective_bw + \
                                        mac_dict[module_name] * act_effective_bw_max * param_effective_bw_max
                else:
                    bit_ops_reduction = bit_ops_reduction - mac_dict[module_name] * act_effective_bw + \
                                        mac_dict[module_name] * act_effective_bw_max
    elif 'weight' in quantizer_group_dict:
        for module_name in quantizer_group_dict['weight']:
            if module_name in mac_dict:
                if param_bw_max is not None and param_bw is not None:
                    bit_ops_reduction = bit_ops_reduction - mac_dict[module_name] * act_effective_bw_max \
                                        * param_effective_bw + mac_dict[module_name] * act_effective_bw_max \
                                        * param_effective_bw_max
                else:
                    bit_ops_reduction = bit_ops_reduction - mac_dict[module_name] * act_effective_bw_max \
                                        + mac_dict[module_name] * act_effective_bw_max

    if 'input' in quantizer_group_dict:
        for module_name in quantizer_group_dict['input']:
            if module_name in mac_dict:
                if param_bw_max is not None and param_bw is not None:
                    bit_ops_reduction = bit_ops_reduction - mac_dict[module_name] * act_effective_bw * \
                                        param_effective_bw_max + mac_dict[module_name] * act_effective_bw_max * \
                                        param_effective_bw_max
                else:
                    bit_ops_reduction = bit_ops_reduction - mac_dict[module_name] * act_effective_bw \
                                        + mac_dict[module_name] * act_effective_bw_max
    return bit_ops_reduction


def create_mac_dict(model: torch.nn.Module, dummy_input: Union[torch.Tensor, Tuple]) -> Dict[str, int]:
    """
    Create a dictionary mapping compressible modules (Conv and Linear) to mac counts
    :param model: Torch model to evaluate
    :param dummy_input: Dummy input to the model
    """
    is_v2_model = any(isinstance(module, BaseQuantizationMixin) for module in model.modules())

    layer_db = LayerDatabase(model, dummy_input)
    mac_dict = {}
    for compressible_layer in layer_db.get_compressible_layers().values():
        if is_v2_model:
            module_name =  compressible_layer.name
        else:
            # Should truncate "._module_to_wrap" suffix from module name
            assert compressible_layer.name.endswith('._module_to_wrap')
            module_name = compressible_layer.name[:-len('._module_to_wrap')]
        mac_dict[module_name] = CostCalculator.compute_layer_cost(compressible_layer).mac
    return mac_dict


def create_quantizer_module_dict(quantizer_group: QuantizerGroup) -> Dict:
    """
    Creates quantizer module dictionary from a quantizer group
    :param quantizer_group:
    :return: Dict of quantizer
    """
    quantizer_module_dict = defaultdict(list)
    for module_name in quantizer_group.get_input_quantizer_modules():
        quantizer_module_dict['input'].append(module_name)
    for module_name in quantizer_group.output_quantizers:
        quantizer_module_dict['output'].append(module_name)
    for module_name in quantizer_group.parameter_quantizers:
        quantizer_module_dict['weight'].append(module_name)

    return quantizer_module_dict


def calculate_running_bit_ops(mac_dict: Dict[str, int],
                              quantizer_group: QuantizerGroup,
                              module_bitwidth_dict: Dict[str, List],
                              max_candidate: CANDIDATE_WITH_DTYPE,
                              new_candidate: CANDIDATE_WITH_DTYPE,
                              running_bit_ops: int) -> int:
    """
    Returns new running bit ops given previous running bit ops value and the current quantizer to change bitwidth of
    :param mac_dict: Dictionary mapping modules to mac count of the module (only Conv and Linear modules are present in
    the dictionary)
    :param quantizer_group: TensorQuantizer to change bitwidth of
    :param module_bitwidth_dict: Dictionary mapping modules to tuples of quantizer types and most recently used
    bitwidths for each quantizer type (only tracks input and weight quantizers)
    :param max_candidate: Maximum bitwidth and data type for the TensorQuantizer
    :param new_candidate: New bitwidth and data type for the TensorQuantizer
    :param running_bit_ops: previous running bit ops count
    """
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _calculate_bit_ops(quantizer_type: str, running_bit_ops: int, module_name: str) -> int:
        """
        Helper function to compute bit ops for weight or inputs feeding into an op
        """
        if module_name in mac_dict:
            if module_name in module_bitwidth_dict.keys():
                # If an input or weight quantizer of the module was quantized previously, there will be an entry in
                # the dictionary.  Then, we simply need to update the corresponding index (input or weight) with
                # the new bitwidth.

                # Subtract the previous bitops count for this module (will add on the new bitops count for this
                # module later, taking new bitwidth value into account)
                if module_bitwidth_dict[module_name][weight_index] is not None:
                    running_bit_ops -= (module_bitwidth_dict[module_name][input_index] *
                                        module_bitwidth_dict[module_name][weight_index] *
                                        mac_dict[module_name])
                else:
                    running_bit_ops -= (module_bitwidth_dict[module_name][input_index] *
                                        mac_dict[module_name])
                if quantizer_type == 'input':
                    module_bitwidth_dict[module_name][input_index] = act_effective_bw
                else:
                    module_bitwidth_dict[module_name][weight_index] = param_effective_bw
            else:
                # New entry in module_bitwidth_dict needs to be made.  One index will be default_bitwidth, and the
                # other index will be the new bitwidth we are currently using.

                # Subtract the previous bitops count for this module (will add on the new bitops count for this
                # module later, taking new bitwidth value into account)
                if param_effective_bw_max is not None:
                    running_bit_ops -= (act_effective_bw_max * param_effective_bw_max) * mac_dict[module_name]
                else:
                    running_bit_ops -= act_effective_bw_max * mac_dict[module_name]
                if quantizer_type == 'input':
                    module_bitwidth_dict[module_name] = [act_effective_bw, param_effective_bw_max]
                else:
                    module_bitwidth_dict[module_name] = [act_effective_bw_max, param_effective_bw]
            # Add new bitops count to the running bit ops value, taking into account the updated bitwidths for
            # input and weight quantizers.
            if module_bitwidth_dict[module_name][weight_index] is not None:
                running_bit_ops += (module_bitwidth_dict[module_name][input_index] *
                                    module_bitwidth_dict[module_name][weight_index] *
                                    mac_dict[module_name])
            else:
                running_bit_ops += (module_bitwidth_dict[module_name][input_index] *
                                    mac_dict[module_name])
        return running_bit_ops

    input_index = 0
    weight_index = 1

    quantizer_module_dict = create_quantizer_module_dict(quantizer_group)

    if len(max_candidate) == 1:
        (activation_bw_max, act_max_dtype), = max_candidate
        param_bw_max, param_max_dtype = None, None
    else:
        (activation_bw_max, act_max_dtype), (param_bw_max, param_max_dtype) = max_candidate

    if len(new_candidate) == 1:
        (activation_bw_new, act_dtype), = new_candidate
        param_bw_new, param_dtype = None, None
    else:
        (activation_bw_new, act_dtype), (param_bw_new, param_dtype) = new_candidate

    act_effective_bw_max = get_effective_bitwidth(act_max_dtype, activation_bw_max)
    if param_bw_max is not None:
        param_effective_bw_max = get_effective_bitwidth(param_max_dtype, param_bw_max)
    else:
        param_effective_bw_max = None
    act_effective_bw = get_effective_bitwidth(act_dtype, activation_bw_new)
    if param_bw_new is not None:
        param_effective_bw = get_effective_bitwidth(param_dtype, param_bw_new)
    else:
        param_effective_bw = None

    if 'output' in quantizer_module_dict and 'weight' in quantizer_module_dict:
        for module_name in quantizer_module_dict['weight']:
            # Output of previous op is the input to this op
            running_bit_ops = _calculate_bit_ops('input', running_bit_ops, module_name)

    if 'input' in quantizer_module_dict:
        for module_name in quantizer_module_dict['input']:
            running_bit_ops = _calculate_bit_ops('input', running_bit_ops, module_name)

    if 'weight' in quantizer_module_dict:
        for module_name in quantizer_module_dict['weight']:
            running_bit_ops = _calculate_bit_ops('weight', running_bit_ops, module_name)

    return running_bit_ops
