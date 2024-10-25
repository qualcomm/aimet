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

""" Test top level mixed precision api """

import os
import pathlib
import tempfile

import pytest
import unittest
import unittest.mock

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import json

from aimet_torch.mixed_precision import choose_mixed_precision
from aimet_common.defs import QuantizationDataType, CallbackFunc
from aimet_torch.v1.quantsim import QuantizationSimModel
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v1.batch_norm_fold import fold_all_batch_norms
from aimet_torch.amp import utils as mixed_precision_utils
from aimet_torch.amp.mixed_precision_algo import EvalCallbackFactory
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper
from aimet_common.amp.utils import AMPSearchAlgo, calculate_starting_bit_ops

def get_config(config_dir: str):
    quantsim_config = {
        "defaults":
            {
                "ops":
                    {
                        "is_output_quantized": "True"
                    },
                "params":
                    {
                        "is_quantized": "True",
                        "is_symmetric": "True"
                    },
                "per_channel_quantization": "True",
                "strict_symmetric": "False",
                "unsigned_symmetric": "False"
            },

        "params":
            {
                "bias":
                    {
                        "is_quantized": "False"
                    }
            },

        "op_type":
            {
                "Dropout":
                    {
                        "is_output_quantized": "False"
                    },
                "Reshape":
                    {
                        "is_output_quantized": "False"
                    },
                "Gemm":
                    {
                        "per_channel_quantization": "False"
                    },
                "MaxPool":
                    {
                        "is_output_quantized": "False"
                    },
                "Sigmoid":
                    {
                        "encoding_constraints":
                            {
                                "min": 0.0,
                                "max": 1.0
                            }
                    },
                "Softmax":
                    {
                        "encoding_constraints":
                            {
                                "min": 0.0,
                                "max": 1.0
                            }
                    },
                "Transpose":
                    {
                        "is_output_quantized": "False"
                    }
            },

        "supergroups":
            [
                {
                    "op_list": ["Add", "Relu"]
                },
                {
                    "op_list": ["Conv", "Clip"]
                },
                {
                    "op_list": ["Conv", "HardSwish"]
                },
                {
                    "op_list": ["Conv", "PRelu"]
                },
                {
                    "op_list": ["Conv", "Relu"]
                },
                {
                    "op_list": ["ConvTranspose", "Relu"]
                },
                {
                    "op_list": ["Gemm", "Relu"]
                }
            ],

        "model_input":
            {
                "is_input_quantized": "True"
            },

        "model_output":
            {}
    }

    with open(os.path.join(config_dir, 'quantsim_config.json'), 'w') as f:
        json.dump(quantsim_config, f)
    return os.path.join(config_dir, 'quantsim_config.json')


class TestMixedPrecision:
    """ Test case for mixed precision api """


    @pytest.mark.cuda
    def test_quantize_with_mixed_precision(self):
        """ Test top level quantize_with_mixed_precision api """

        torch.manual_seed(0)

        model = torch.load('data/mnist_trained_on_GPU.pth')
        input_shape = (1, 1, 28, 28)
        dummy_input = torch.randn(1, 1, 28, 28).cuda()
        default_bitwidth = 16
        # ((activation bitwidth, activation data type), (param bitwidth, param data type))
        candidates = [((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
                     ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
                     ((8, QuantizationDataType.int), (8, QuantizationDataType.int))]
        allowed_accuracy_drop = None

        # Quantize the model to default bitwidth
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, default_param_bw=default_bitwidth, default_output_bw=default_bitwidth,
                                       dummy_input=dummy_input, config_file=get_config(tempdir))
            sim.compute_encodings(forward_pass_callback, forward_pass_callback_args=input_shape)
            eval_callback = CallbackFunc(eval_function(num_candidates=len(candidates)), None)
            fp32_accuracy = eval_callback.func(model, None)
            forward_pass_call_back = CallbackFunc(forward_pass_callback, input_shape)

            pareto_front_list = choose_mixed_precision(sim, dummy_input, candidates, eval_callback, eval_callback,
                                                       allowed_accuracy_drop, tempdir, True, forward_pass_call_back,
                                                       amp_search_algo=AMPSearchAlgo.BruteForce)

            assert len(pareto_front_list) == 18
            # Test that final eval score is still within allowable accuracy range
            _, eval_score, _, _ = pareto_front_list[-1]
            assert fp32_accuracy - eval_score < 0.01

    @pytest.mark.cuda
    def test_quantize_with_mixed_precision_fp16_1(self):
        """
        Test top level quantize_with_mixed_precision api and export functionality
        - The allowed accuracy drop is set to a very low value, and hence the resultant sim.model would only have
        float data_type used for all layers. This is done to verify the encodings file generated for the fp16 model
        through amp. The encodings should only contain dtype and bitwidth as per spec. The pareto_front_list would
        also have no entries
        """

        torch.manual_seed(0)

        model = torch.load('data/mnist_trained_on_GPU.pth')
        input_shape = (1, 1, 28, 28)
        dummy_input = torch.randn(1, 1, 28, 28).cuda()
        default_bitwidth = 16
        # ((activation bitwidth, activation data type), (param bitwidth, param data type))
        candidates = [((16, QuantizationDataType.int), (32, QuantizationDataType.int)),
                      ((16, QuantizationDataType.float), (16, QuantizationDataType.float))]
        allowed_accuracy_drop = None

        # Quantize the model to default bitwidth
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, default_param_bw=default_bitwidth, default_output_bw=default_bitwidth,
                                       dummy_input=dummy_input, config_file=get_config(tempdir))
            sim.compute_encodings(forward_pass_callback, forward_pass_callback_args=input_shape)
            eval_callback = CallbackFunc(eval_function(num_candidates=len(candidates)), None)
            forward_pass_call_back = CallbackFunc(forward_pass_callback, input_shape)

            pareto_front_list = choose_mixed_precision(sim, dummy_input, candidates, eval_callback, eval_callback,
                                                       allowed_accuracy_drop, tempdir, True, forward_pass_call_back,
                                                       amp_search_algo=AMPSearchAlgo.BruteForce)

            sim.export(tempdir, 'test_quantize_with_mixed_precision_fp16_1', torch.randn(1, 1, 28, 28))

            assert len(pareto_front_list) == 9

            with open(os.path.join(tempdir, 'test_quantize_with_mixed_precision_fp16_1.encodings'), "r") as encodings_file:
                encodings = json.load(encodings_file)

            assert len(encodings['activation_encodings'].keys()) == 8
            assert len(encodings['param_encodings'].keys()) == 4

            for name in encodings['activation_encodings']:
                layer_encoding_dict = encodings['activation_encodings'][name][0]
                assert len(layer_encoding_dict) == 2
                assert layer_encoding_dict['dtype'] == 'float'
                assert layer_encoding_dict['bitwidth'] == 16

            for name in encodings['param_encodings']:
                layer_encoding_dict = encodings['param_encodings'][name][0]
                assert len(layer_encoding_dict) == 2
                assert layer_encoding_dict['dtype'] == 'float'
                assert layer_encoding_dict['bitwidth'] == 16


    @pytest.mark.cuda
    def test_quantize_with_mixed_precision_fp16_2(self):
        """
        Test top level choose_mixed_precision API.
        This test creates a QuantSim model, calls the choose_mixed_precision with the created QuantSim model and a list
        of candidates (both float and int), and compares eval score of the pareto_front_list[-1] with the
        allowed_accuracy_drop
        """

        torch.manual_seed(0)

        model = torch.load('data/mnist_trained_on_GPU.pth')
        input_shape = (1, 1, 28, 28)
        dummy_input = torch.randn(1, 1, 28, 28).cuda()
        default_bitwidth = 16
        # ((activation bitwidth, activation data type), (param bitwidth, param data type))
        candidates = [((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
                      ((8, QuantizationDataType.int), (16, QuantizationDataType.int)),
                      ((8, QuantizationDataType.int), (8, QuantizationDataType.int))]
        allowed_accuracy_drop = None

        # Quantize the model to default bitwidth
        with tempfile.TemporaryDirectory() as tempdir:
            sim = QuantizationSimModel(model, default_param_bw=default_bitwidth, default_output_bw=default_bitwidth,
                                       dummy_input=dummy_input, config_file=get_config(tempdir))
            sim.compute_encodings(forward_pass_callback, forward_pass_callback_args=input_shape)
            eval_callback = CallbackFunc(eval_function(num_candidates=len(candidates)), None)
            fp32_accuracy = eval_callback.func(model, None)
            forward_pass_call_back = CallbackFunc(forward_pass_callback, input_shape)

            pareto_front_list = choose_mixed_precision(sim, dummy_input, candidates, eval_callback, eval_callback,
                                                       allowed_accuracy_drop, tempdir, True, forward_pass_call_back,
                                                       amp_search_algo=AMPSearchAlgo.BruteForce)

            assert len(pareto_front_list) == 18
            # Test that final eval score is still within allowable accuracy range
            _, eval_score, _, _ = pareto_front_list[-1]
            assert fp32_accuracy - eval_score < 0.01


    @pytest.mark.cuda
    def test_quantize_with_mixed_precision_fp16_3(self):
        """
        Test top level choose_mixed_precision API by comparing int and fp16 behavior.
        This test creates two QuantSim models - one with default_data_type set to int and the other to float,
        calls the choose_mixed_precision with the created QuantSim models and a list
        of candidates containing both float and int candidates, and compares relative_bit_ops, quantizer_groups, and
        candidates generated from both the int and fp16 models.
        This lets us confirm that the behavior is the same irrespective of whether the QuantSim model is created
        initially with fp16 or with int data type.
        """

        torch.manual_seed(0)

        model = torch.load('data/mnist_trained_on_GPU.pth')
        input_shape = (1, 1, 28, 28)
        dummy_input = torch.randn(1, 1, 28, 28).cuda()
        default_bitwidth = 16

        # ((activation bitwidth, activation data type), (param bitwidth, param data type))
        candidates = [((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
                      ((8, QuantizationDataType.int), (16, QuantizationDataType.int)),
                      ((8, QuantizationDataType.int), (8, QuantizationDataType.int))]
        allowed_accuracy_drop = None

        # Quantize the model to default bitwidth
        with tempfile.TemporaryDirectory() as tempdir:
            sim_fp16 = QuantizationSimModel(model, default_param_bw=default_bitwidth, default_output_bw=default_bitwidth,
                                            dummy_input=dummy_input, default_data_type=QuantizationDataType.float,
                                            config_file=get_config(tempdir))
            sim_fp16.compute_encodings(forward_pass_callback, forward_pass_callback_args=input_shape)
            eval_callback = CallbackFunc(eval_function(num_candidates=len(candidates)), None)
            fp32_accuracy_fp16 = eval_callback.func(sim_fp16.model, None)
            forward_pass_call_back = CallbackFunc(forward_pass_callback, input_shape)

            pareto_front_list_fp16 = choose_mixed_precision(sim_fp16, dummy_input, candidates, eval_callback, eval_callback,
                                                            allowed_accuracy_drop, tempdir, True, forward_pass_call_back,
                                                            amp_search_algo=AMPSearchAlgo.BruteForce)

            # Test that final eval score is still within allowable accuracy range
            _, eval_score, _, _ = pareto_front_list_fp16[-1]
            assert fp32_accuracy_fp16 - eval_score < 0.01


            sim_int = QuantizationSimModel(model, default_param_bw=default_bitwidth, default_output_bw=default_bitwidth,
                                            dummy_input=dummy_input, default_data_type=QuantizationDataType.int,
                                            config_file=get_config(tempdir))
            sim_int.compute_encodings(forward_pass_callback, forward_pass_callback_args=input_shape)
            eval_callback = CallbackFunc(eval_function(num_candidates=len(candidates)), None)
            fp32_accuracy_int = eval_callback.func(sim_int.model, None)

            pareto_front_list_int = choose_mixed_precision(sim_int, dummy_input, candidates, eval_callback, eval_callback,
                                                           allowed_accuracy_drop, tempdir, True,
                                                           forward_pass_call_back, amp_search_algo=AMPSearchAlgo.BruteForce)

            # Test that final eval score is still within allowable accuracy range
            _, eval_score, _, _ = pareto_front_list_int[-1]
            assert fp32_accuracy_int - eval_score < 0.01

            # we expect the same behavior for both the QuantSimModel started with float and the int default_data_type
            assert len(pareto_front_list_int) == len(pareto_front_list_fp16),\
                    "Length of the int pareto front list is not equal to the fp16 pareto front list"

            # check if pareto front list generated for both int QuantSimModel and fp16 QuantSimModel are the same
            # (relative_bit_ops, eval_score, quantizer_group, candidate)
            for i in range(0, len(pareto_front_list_int)):
                relative_bit_ops_fp16, _, quantizer_group_fp16, candidate_fp16 = pareto_front_list_fp16[i]
                relative_bit_ops_int, _, quantizer_group_int, candidate_int = pareto_front_list_int[i]
                assert relative_bit_ops_fp16 == relative_bit_ops_int,\
                        "relative bit ops differ between int and fp16 pareto lists"
                assert candidate_fp16 == candidate_int,\
                        "candidates differ between int and fp16 pareto lists"
                assert quantizer_group_fp16 == quantizer_group_int,\
                        "quantizer groups differ between int and fp16 pareto lists"


    def test_dummy(self):
        # pytest has a 'feature' that returns an error code when all tests for a given suite are not selected
        # to be executed
        # So adding a dummy test to satisfy pytest
        pass

    @pytest.mark.cuda
    @pytest.mark.parametrize(
        "search_algo",
        (AMPSearchAlgo.BruteForce, AMPSearchAlgo.Interpolation, AMPSearchAlgo.Binary)
    )
    def test_quantize_with_mixed_precision_v2(self, search_algo):
        """ Test top level quantize_with_mixed_precision api """
        torch.manual_seed(0)

        model = resnet18().cuda()
        model = prepare_model(model)

        input_shape = (1, 3, 224, 224)
        dummy_input = torch.rand(input_shape).cuda()

        fold_all_batch_norms(model, input_shape)

        default_bitwidth = 16
        # ((activation bitwidth, activation data type), (param bitwidth, param data type))
        candidates = [((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
                     ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
                     ((8, QuantizationDataType.int), (8, QuantizationDataType.int))]
        allowed_accuracy_drop = 0.1

        # Quantize the model to default bitwidth
        sim = QuantizationSimModel(model, default_param_bw=default_bitwidth,
                                   default_output_bw=default_bitwidth,
                                   dummy_input=dummy_input)
        sim.compute_encodings(forward_pass_callback, forward_pass_callback_args=input_shape)

        class _Dataset(Dataset):
            def __getitem__(self, _):
                return dummy_input[0,:].clone()

            def __len__(self):
                return 1

        data_loader = DataLoader(_Dataset())

        # Use SQNR eval funcion for phase 1
        eval_callback_phase1 = EvalCallbackFactory(data_loader).sqnr()

        args = (dummy_input, candidates)
        # Use full eval function for phase 2
        eval_callback_phase2 = CallbackFunc(eval_function_v2, args)

        fp32_accuracy = eval_callback_phase2.func(model, args)

        forward_pass_call_back = CallbackFunc(forward_pass_callback, input_shape)

        with tempfile.TemporaryDirectory() as tempdir:
            pareto_front_list = choose_mixed_precision(sim, dummy_input, candidates,
                                                       eval_callback_phase1, eval_callback_phase2,
                                                       allowed_accuracy_drop, tempdir, True,
                                                       forward_pass_call_back, search_algo)
            assert pareto_front_list

            eval_score = eval_function_v2(sim.model, args)

            # Check pareto curve contains the final eval score
            pareto_eval_scores = [eval_score for _, eval_score, _, _ in pareto_front_list]
            assert eval_score in pareto_eval_scores

            # Check final eval score is within tolerable range
            assert fp32_accuracy - eval_score < 0.1


def forward_pass_callback(model, inp_shape):
    """ Call mnist_evaluate setting use_cuda to True, iterations=5 """

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(inp_shape).cuda())
    return output


def eval_function_v2(model, args):
    """
    Returns eval score in [0, 1] range.
    NOTE: In this example, we use relative bitops as the eval score to simulate
          a model whose eval scores are proportional to bitops.
          Also assumed W16A16 is equal to fp32 accuracy
    """
    is_fp32 = True
    for _, module in model.named_modules():
        if isinstance(module, QcQuantizeWrapper):
            quantizer = module.param_quantizers.get("weight")
            if quantizer is not None and quantizer.enabled:
                is_fp32 = False
                break
    if is_fp32:
        # FP32 model's eval score is the same as W16A16
        return 1.0

    dummy_input, candidates = args
    mac_dict = mixed_precision_utils.create_mac_dict(model, dummy_input)

    max_bitops = 0
    for candidate in candidates:
        bitops = calculate_starting_bit_ops(mac_dict, candidate)
        max_bitops = max(max_bitops, bitops)

    current_bitops = 0

    for name, module in model.named_modules():
        if name not in mac_dict:
            continue
        mac = mac_dict[name]

        if not module.param_quantizers["weight"].enabled:
            # FP32 model's eval score is the same as W16A16
            param_bw = 16
            output_bw = 16
        else:
            param_bw = module.param_quantizers["weight"].bitwidth
            output_bw = module.output_quantizers[0].bitwidth

        current_bitops += mac * output_bw * param_bw

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
