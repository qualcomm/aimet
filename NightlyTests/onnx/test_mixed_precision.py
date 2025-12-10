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

"""Test top level mixed precision api"""

import tempfile
import pytest
import unittest
import unittest.mock
import numpy as np
from test_models import resnet18
from aimet_onnx.mixed_precision import choose_mixed_precision
from aimet_onnx.common.defs import QuantizationDataType, CallbackFunc
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.defs import DataLoader
from aimet_onnx.amp import utils as mixed_precision_utils
from aimet_onnx.amp.mixed_precision_algo import EvalCallbackFactory
from aimet_onnx.amp.quantizer_groups import find_quantizer_group
from aimet_onnx.common.amp.utils import AMPSearchAlgo, calculate_starting_bit_ops


class TestMixedPrecision:
    """Test case for mixed precision api"""

    @pytest.mark.cuda
    def test_quantize_with_mixed_precision(self):
        """Test top level quantize_with_mixed_precision api"""
        model = resnet18()
        default_bitwidth = 16
        input_shape = (1, 3, 224, 224)
        # ((activation bitwidth, activation data type), (param bitwidth, param data type))
        candidates = [
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
        ]
        allowed_accuracy_drop = 0.10015

        # Quantize the model to default bitwidth
        sim = QuantizationSimModel(
            model,
            default_param_bw=default_bitwidth,
            default_activation_bw=default_bitwidth,
        )
        sim.compute_encodings(
            forward_pass_callback, forward_pass_callback_args=input_shape
        )
        eval_callback = CallbackFunc(
            eval_function(num_candidates=len(candidates)), None
        )
        fp32_accuracy = eval_callback.func(model, None)
        forward_pass_call_back = CallbackFunc(forward_pass_callback, input_shape)

        with tempfile.TemporaryDirectory() as tempdir:
            pareto_front_list = choose_mixed_precision(
                sim,
                candidates,
                eval_callback,
                eval_callback,
                allowed_accuracy_drop,
                tempdir,
                True,
                forward_pass_call_back,
                amp_search_algo=AMPSearchAlgo.BruteForce,
            )

            assert len(pareto_front_list) == 66, (
                "Length of the pareto front list is not equal to the expected value: 24"
            )
            # Test that final eval score is still within allowable accuracy range
            _, eval_score, _, _ = pareto_front_list[-1]
            assert fp32_accuracy - eval_score < 0.1

    @pytest.mark.cuda
    @pytest.mark.parametrize(
        "search_algo",
        (AMPSearchAlgo.BruteForce, AMPSearchAlgo.Interpolation, AMPSearchAlgo.Binary),
    )
    def test_quantize_with_mixed_precision_v2(self, search_algo):
        """Test top level quantize_with_mixed_precision api"""

        model = resnet18()
        default_bitwidth = 16
        input_shape = (1, 3, 224, 224)
        # ((activation bitwidth, activation data type), (param bitwidth, param data type))
        candidates = [
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
        ]
        allowed_accuracy_drop = None

        # Quantize the model to default bitwidth
        sim = QuantizationSimModel(
            model,
            default_param_bw=default_bitwidth,
            default_activation_bw=default_bitwidth,
        )
        sim.compute_encodings(
            forward_pass_callback, forward_pass_callback_args=input_shape
        )

        with tempfile.TemporaryDirectory() as tempdir:
            np.random.seed(0)
            dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

            class _Dataloader(DataLoader):
                def __init__(self):
                    super(_Dataloader, self).__init__(dummy_input, 32, 1)

                def __iter__(self):
                    yield dummy_input

            data_loader = _Dataloader()

            # Use SQNR eval funcion for phase 1
            eval_callback_phase1 = EvalCallbackFactory(data_loader).sqnr(sim)

            args = (dummy_input, candidates, sim)
            # Use full eval function for phase 2
            eval_callback_phase2 = CallbackFunc(eval_function_v2, args)

            fp32_accuracy = 1.0

            forward_pass_call_back = CallbackFunc(forward_pass_callback, input_shape)

            pareto_front_list = choose_mixed_precision(
                sim,
                candidates,
                eval_callback_phase1,
                eval_callback_phase2,
                allowed_accuracy_drop,
                tempdir,
                True,
                forward_pass_call_back,
                search_algo,
            )
            assert pareto_front_list

            eval_score = eval_function_v2(sim.model, args)

            # Check pareto curve contains the final eval score
            pareto_eval_scores = [
                eval_score for _, eval_score, _, _ in pareto_front_list
            ]
            assert eval_score in pareto_eval_scores


def forward_pass_callback(session, inp_shape):
    """Call mnist_evaluate setting use_cuda to True, iterations=5"""
    inputs = np.random.rand(*inp_shape).astype(np.float32)
    in_tensor = {"input": inputs}
    session.run(None, in_tensor)


def eval_function_v2(model, args):
    """
    Returns eval score in [0, 1] range.
    NOTE: In this example, we use relative bitops as the eval score to simulate
          a model whose eval scores are proportional to bitops.
          Also assumed W16A16 is equal to fp32 accuracy
    """
    dummy_input, candidates, sim = args

    is_fp32 = True
    for quantizer in sim.qc_quantize_op_dict.values():
        if quantizer is not None and quantizer.enabled:
            is_fp32 = False
            break
    if is_fp32:
        # FP32 model's eval score is the same as W16A16
        return 1.0

    mac_dict = mixed_precision_utils.create_mac_dict(sim)

    max_bitops = 0
    for candidate in candidates:
        bitops = calculate_starting_bit_ops(mac_dict, candidate)
        max_bitops = max(max_bitops, bitops)

    current_bitops = 0
    qg_dict, quantizer_groups = find_quantizer_group(sim)
    weight_name_to_op_name_dict = {}
    for op_name, op in sim.connected_graph.get_all_ops().items():
        if op.parameters:
            for product, values in op.parameters.values():
                if values == "weight":
                    weight_name_to_op_name_dict[product.name] = op_name
    for quantizer_group in quantizer_groups:
        if (
            quantizer_group.activation_quantizers
            and quantizer_group.parameter_quantizers
        ):
            (act_bw, _), (param_bw, _) = quantizer_group.get_candidate(qg_dict)
            param_quantizer_name = quantizer_group.parameter_quantizers[0]
            mac = mac_dict[weight_name_to_op_name_dict[param_quantizer_name]]
            current_bitops += mac * act_bw * param_bw

    assert current_bitops <= max_bitops
    return current_bitops / max_bitops


def eval_function(num_candidates=3):
    eval_scores = []
    first_val = 0.71
    for i in range(0, 1000):
        if i <= num_candidates + 1:
            val = first_val - 0.1 * i
        else:
            val = first_val - 0.0001 * i
        eval_scores.append(val)
    mock_eval = unittest.mock.MagicMock()
    mock_eval.side_effect = eval_scores
    return mock_eval
