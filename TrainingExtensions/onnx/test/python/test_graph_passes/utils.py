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
from aimet_onnx.meta.operations import Op
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from typing import Dict, List


def assert_on_const_quantizers(ops: List[Op], qc_quantize_op_dict: Dict[str, QcQuantizeOp], enabled: bool = False):
    """
    Assert on all constant inputs for provided list of ops with given condition

    Args:
        ops (List[Op]): List of ops to check constant inputs
        qc_quantize_op_dict (Dict[str, QcQuantizeOp]): Global dictionary of quantizer names to Quantize op.
        enabled (bool, optional): Condition to check. Defaults to False.
    """
    for op in ops:
        for op_input in op.inputs:
            if op_input.is_const and op_input.name in qc_quantize_op_dict:
                assert qc_quantize_op_dict[op_input.name].enabled == enabled

def assert_on_output_quantizers(ops: List[Op], qc_quantize_op_dict: Dict[str, QcQuantizeOp], enabled: bool = False):
    """
    Assert on all output quantizers for provided list of ops with given condition

    Args:
        ops (List[Op]): List of ops to check output quantizers
        qc_quantize_op_dict (Dict[str, QcQuantizeOp]): Global dictionary of quantizer names to Quantize op.
        enabled (bool, optional): Condition to check. Defaults to False.
    """
    for op in ops:
        for output in op.outputs:
            if output.name in qc_quantize_op_dict:
                assert qc_quantize_op_dict[output.name].enabled == enabled
