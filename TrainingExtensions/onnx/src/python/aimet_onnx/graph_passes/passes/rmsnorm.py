# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
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
# pylint: disable=missing-module-docstring

from aimet_common.connected_graph.operation import Op
from aimet_onnx.graph_passes.graph_pass import SupergroupGraphPass
from aimet_onnx.graph_passes.pass_registry import register_pass
from aimet_onnx.graph_passes.utils import check_consecutive_ops, match_pow_2_pattern, is_constant_scalar
from aimet_onnx.utils import ModelProto

@register_pass("RMSNormalization")
class RMSNormalization(SupergroupGraphPass):
    """
    Disable output quantizers for RMSNormalization intermediate ops:
    
    RMSNormalization(x) = x / Sqrt(E(x**2) + ε) * γ

    Expected graph:
    Version 1: With x * div ( 1 / denominator )
                x
            +---+---+
            |       |
    Mul or Pow(x, 2)|
            |       |
        ReduceMean  |
            |       |
            Add     |
            |       |
            Sqrt    |
        1   |       |
        +-- Div     |
            |       |
            +---+---+ 
                Mul
                |
                Mul (if elementwise_affine=True)

    Version 2: With x * div ( 1 / denominator )
                x
            +---+---+
            |       |
            |       Mul or Pow(x, 2)
            |       |
            |       ReduceMean
            |       |
            |       Add
            |       |
            |       Sqrt
            |       |
            +---+---+ 
                Div
                |
                Mul (if elementwise_affine=True)
    """

    # pylint: disable=too-many-branches, too-many-return-statements
    def match_pattern(self, op: Op, model: ModelProto):
        """
        Match LayerNormalization pattern and collect ops to disable output quantizers
        """
        # Match Mul(x, x) or Pow(x, 2)
        match = match_pow_2_pattern(op, model)
        if not match or len(op.output_ops) != 1:
            return False

        # Sqrt(E(Pow(x, 2)) + ε)
        match, denominator_ops = check_consecutive_ops(op.output_ops[0], ["ReduceMean", "Add", "Sqrt", "Div"], validate_last_op_consumers=False)
        if not match:
            return False

        all_ops = [ op ] + denominator_ops
        div_op = all_ops[-1]

        # Div pattern 1: x * (1 / Sqrt(E(Pow(x, 2)) + ε))
        if is_constant_scalar(model, div_op.inputs[0], 1) and len(div_op.output_ops) == 1:
            mul_op = div_op.output_ops[0]
            # Mul input order can be anything.
            input_names = { input.name for input in mul_op.inputs }
            expected_inputs = { op.inputs[0].name, div_op.outputs[0].name }
            if mul_op.type != "Mul" or input_names != expected_inputs:
                return False
            all_ops.append(mul_op)

        # Div pattern 2: x / Sqrt(E(Pow(x, 2)) + ε)
        elif div_op.inputs[0] != op.inputs[0]:
            return False

        # Check if weights are present
        elementwise_affine = False
        if len(all_ops[-1].output_ops) == 1 and all_ops[-1].output_ops[0].type == "Mul":
            elementwise_affine = True
            # Weights are present
            all_ops.append(all_ops[-1].output_ops[0])

        # Disable output quantizers for all the intermediate outputs
        self.disable_output_quantizers(all_ops[:-1])
        # Disable all constant quantizers except weights
        self.disable_const_quantizers(all_ops[:-1] if elementwise_affine else all_ops)
        return True
