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
from aimet_onnx.graph_passes.utils import check_consecutive_ops, get_op_from_outputs, match_and_get_next_op

@register_pass("LayerNormalization")
class LayerNormalization(SupergroupGraphPass):
    """
    Disable output quantizers for LayerNormalization intermediate ops:
    
    LayerNormalization(x) = (x - E(x)) / Sqrt(Var(x) + ε) * γ + β

    Expected graph:
                x
            +---+---+
            |       |
        ReduceMean  |
            |       |
            +---+---+     
                Sub
            +---+---+
            |       |
            Pow     |
            |       |
        ReduceMean  | 
            |       |
            Add     |
            |       |
            Sqrt    |
            +---+---+ 
                Div
                |
                Mul (if affine_transform=True)
                |
                Add (if bias=True)
    """

    # pylint: disable=too-many-branches, too-many-return-statements
    def match_pattern(self, op:Op):
        """
        Match LayerNormalization pattern and collect ops to disable output quantizers
        """
        # E[x]
        sub_1 = match_and_get_next_op(op, "ReduceMean")

        # x - E[x]
        if sub_1 is None or sub_1.type != "Sub" or len(sub_1.output_ops) != 2 or sub_1.inputs[0] != op.inputs[0]:
            return False

        pow_1 = get_op_from_outputs(sub_1, "Pow")
        div_1 = get_op_from_outputs(sub_1, "Div")
        if pow_1 is None or div_1 is None:
            return False

        # Sqrt(Var(x) + ε)
        match, denominator_ops = check_consecutive_ops(pow_1, ["Pow", "ReduceMean", "Add", "Sqrt"])
        if not match:
            return False

        # (x - E(x)) / Sqrt(Var(x) + ε)
        if div_1.inputs[0].producer != sub_1 or div_1.inputs[1].producer != denominator_ops[-1]:
            return False

        # Collect quantizers to disable.
        all_ops = [ op, sub_1 ] + denominator_ops + [ div_1 ]
        self.disable_output_quantizers(op_list=all_ops[:-1])
        self.disable_const_quantizers(op_list=all_ops)

        # LayerNormalization pattern has been matched.
        # Check if affine_transform is set.
        # (x - E(x)) / Sqrt(Var(x) + ε) * γ
        match, div_mul_ops = check_consecutive_ops(div_1, ["Div", "Mul"])
        if not match:
            return True

        # NOTE: keep weights quantized
        self.disable_output_quantizers(op_list=[div_1])
        self.disable_const_quantizers(op_list=[div_1])

        # (x - E(x)) / Sqrt(Var(x) + ε) * γ + β
        match, mul_add_ops = check_consecutive_ops(div_mul_ops[-1], ["Mul", "Add"], validate_last_op_consumers=False)
        if not match:
            return True

        # NOTE: skip bias quantization
        self.disable_output_quantizers(op_list=mul_add_ops[:1])
        self.disable_const_quantizers(op_list=mul_add_ops[-1:])
        return True
