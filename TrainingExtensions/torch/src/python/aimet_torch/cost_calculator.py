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

"""Per layer cost calculator specialized for PyTorch"""

from functools import reduce
from typing import List

import torch

from aimet_common.cost_calculator import CostCalculator as GenericCostCalculator, Cost
from aimet_torch.layer_database import Layer
from aimet_torch._base.nn.modules.custom import Multiply

SUPPORTED_LAYER_TYPES = [torch.nn.modules.conv.Conv2d,
                         torch.nn.modules.linear.Linear,
                         torch.nn.modules.activation.Sigmoid,
                         torch.nn.modules.pooling.AvgPool2d,
                         Multiply]

class CostCalculator(GenericCostCalculator):
    """ This is a specialized implementation of CostCalculator for PyTorch"""

    @staticmethod
    def compute_layer_cost(layer: Layer):
        """
        Computes per layer cost. This is a specialized function for PyTorch.

        :param layer: Attributes for a layer
        :return: Cost of the layer
        """

        assert isinstance(layer.module, tuple(SUPPORTED_LAYER_TYPES)), "Unsupported layer type encountered"

        weight_dim = list(layer.weight_shape)
        if len(weight_dim) > 1:
            additional_act_dim = [layer.output_shape[-1], layer.output_shape[-2]]
            mem_cost = reduce(lambda x, y: x * y, weight_dim)
            mac_dim = weight_dim + additional_act_dim
            mac_cost = reduce(lambda x, y: x * y, mac_dim)

        else:  # For Ops w/o weights (or single dim), take the max b/w input and output size to calculate cost
            output_mac_cost = reduce(lambda x, y: x * y, layer.output_shape)
            input_mac_cost = 0
            if layer.input_shape:
                if not isinstance(layer.input_shape, List):
                    layer.input_shape = [layer.input_shape]
                input_mac_cost = max([reduce(lambda x, y: x * y, i_input) for i_input in layer.input_shape])

            mac_cost = max([output_mac_cost, input_mac_cost])
            mem_cost = mac_cost

        return Cost(mem_cost, mac_cost)
