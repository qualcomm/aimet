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
from aimet_onnx.meta.product import Product
from aimet_onnx.utils import ParamUtils, ModelProto

from onnx import numpy_helper
import numpy as np
from typing import List, Tuple, Optional, Union


def _get_numpy_array(model: ModelProto, param_name: str) -> Optional[np.ndarray]:
    """
    returns param value as a numpy array from model if present, otherwise None.

    Args:
        model (ModelProto): source Model 
        param_name (str): parameter name to fetch value for

    Returns:
        Optional[nd.array]: returns nd.array if parameter exists. Otherwise, None.
    """
    return numpy_helper.to_array(ParamUtils.get_param_by_name(model, param_name))

def is_constant_scalar(model: ModelProto, op_input: Product, expected_value: Union[int | float]) -> bool:
    """
    Returns True if provided input is constant with scalar value equal to expected value.

    Args:
        model (ModelProto): source Model
        op_input (Product): input to check for constant value for
        expected_value (Union[int | float]): expected value

    Returns:
        bool: returns True if op_input is constant with same scalar value.
    """
    if not op_input.is_const:
        return False

    value = _get_numpy_array(model, op_input.name)
    return value.ndim == 0 and value == expected_value

def match_pow_2_pattern(op: Op, model: ModelProto) -> bool:
    """
    Check if Op is equivalent to pow(x, 2)

    Args:
        op (Op): Op to check for
        model (ModelProto): source model

    Returns:
        bool: Return True if Op is either pow(x, 2) or mul(x, x)
    """
    if op.type == "Mul":
        return op.inputs[0] == op.inputs[1]
    if op.type == "Pow":
        return is_constant_scalar(model, op.inputs[1], 2)
    return False

def match_and_get_next_op(op: Op, op_type: str) -> Op:
    """
    Checks if input op and op_type matches and has exact one output_op.
    If so, returns consumer Op. Otherwise, None

    Args:
        op (Op): Input Op to check for op_type and output_op from.
        op_type (str): op_type to check.

    Returns:
        Op: Output of input op if matches constraint. Otherwise, None.
    """
    if op.type != op_type or len(op.output_ops) != 1:
        return None

    return op.output_ops[0]

def check_consecutive_ops(op: Op, op_type_list: List[str], validate_last_op_consumers: bool=True) -> Tuple[bool, List[Op]]:
    """
    Check for chain of Ops with provided Op type list

    Args:
        op (Op): Starting op
        op_type_list (List[str]): List of Op type to check for
        validate_last_op_consumers (bool): Validate last op for number of outputs if True

    Returns:
        Tuple[bool, List[Op]]: Returns Tuple [True if Op type matches else False, List of corresponding Ops]
    """
    ops = []
    for op_type in op_type_list[:-1]:
        num_output_ops = len(op.output_ops)
        # Return False if
        #  - Op types does not match
        #  - Current op has multiple consumers
        if op.type != op_type or num_output_ops != 1:
            return False, ops
        ops.append(op)
        op = op.output_ops[0]

    # Validate last op
    if op.type != op_type_list[-1]:
        return False, ops

    if validate_last_op_consumers and len(op.output_ops) > 1:
        return False, ops

    # Last op matches the constraint as well
    ops.append(op)
    return True, ops


def get_op_from_outputs(op: Op, output_op_type: str) -> Op:
    """
    Return Op from op output_ops that matches given type.
    Useful utility in case of branching to query Op of interest.

    Args:
        op (Op): Source op to request consumer Op from.
        output_op_type (str): Op type to return from consumers of op.

    Returns:
        Op: Output Op of source op with type output_op_type. Returns None if output_op_type is not present.
    """
    for output in op.output_ops:
        if output.type == output_op_type:
            return output
    return None

def get_output_names(op_list: List[Op]) -> List[str]:
    """
    Returns list of output names for all provided ops

    Args:
        op_list (List(Op)): Provided list of ops.

    Returns:
        List[str]: List of output names for provided ops
    """
    output_names = [ output.name for op in op_list for output in op.outputs ]
    return output_names


def get_const_input_names(op_list: List[Op]) -> List[str]:
    """
    Returns list of constant input names for provided ops

    Args:
        op_list (List(Op)): Provided list of ops

    Returns:
        List[str]: List of constant input names for provided ops
    """
    const_input_names = [ input.name for op in op_list for input in op.inputs if input.is_const ]
    return const_input_names
