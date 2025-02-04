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

import tempfile
import pytest
import itertools
import json
import unittest.mock
import numpy as np
import os
import shutil
import torch
from packaging import version

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo, _compute_sqnr, EvalCallbackFactory
from aimet_onnx.defs import DataLoader

from aimet_common.defs import QuantizationDataType, CallbackFunc
from aimet_common.amp.mixed_precision_algo import interpolation_search, brute_force_search, binary_search
from aimet_common.amp.utils import calculate_starting_bit_ops

from .models.test_models import single_residual_model

INPUT_SHAPE = (1, 3, 32, 32)

def forward_fn(session, _):
    np.random.seed(0)
    test_data = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    session.run(None, {'input': test_data})


@pytest.fixture
def forward_pass_callback():
    return CallbackFunc(forward_fn, func_callback_args=None)


W8A8 = ((8, QuantizationDataType.int), (8, QuantizationDataType.int))
W8A16 = ((16, QuantizationDataType.int), (8, QuantizationDataType.int))
W16A8 = ((8, QuantizationDataType.int), (16, QuantizationDataType.int))
W16A16 = ((16, QuantizationDataType.int), (16, QuantizationDataType.int))


# Lookup table that maps (quantizer_group_1, quantizer_group_2) -> eval_score
# NOTE: This lookup table mocks a model with following characteristics
# 1. Quantizer group 2 is more sensitive than 1
# 2. Activation quantizers are more sensitive then weight quantizers
phase1_eval_score_lookup_table = {
    ("fp32", W8A8): 0.85,
    ("fp32", W16A8): 0.9,
    ("fp32", W8A16): 0.91,

    (W8A8, "fp32"): 0.92,
    (W16A8, "fp32"): 0.93,
    (W8A16, "fp32"): 0.94,

    ("fp32", "fp32"): 1.0,
}

# Lookup table that maps (quantizer_group_1, quantizer_group_2) -> eval_score
# NOTE: This lookup table mocks a model with following characteristics
# 1. Quantizer group 2 is more sensitive than 1
# 2. Activation quantizers are more sensitive then weight quantizers
phase2_eval_score_lookup_table = {
    (W8A8,   W8A8): 0.8,
    (W8A16,  W8A8): 0.81,
    (W16A8,  W8A8): 0.82,
    (W16A16, W8A8): 0.83,

    (W8A8,   W16A8): 0.84,
    (W16A8,  W16A8): 0.85,
    (W8A16,  W16A8): 0.86,
    (W16A16, W16A8): 0.87,

    (W8A8,   W8A16): 0.88,
    (W16A8,  W8A16): 0.89,
    (W8A16,  W8A16): 0.90,
    (W16A16, W8A16): 0.91,

    (W8A8,   W16A16): 0.92,
    (W16A8,  W16A16): 0.93,
    (W8A16,  W16A16): 0.94,
    (W16A16, W16A16): 0.95,

    ("fp32", "fp32"): 1.0,
}

def eval_func(model, args):
    eval_score_lookup_table, sim = args
    # quantizer group 1
    input_quantizer = sim.qc_quantize_op_dict['input']

    conv0_param_quantizer = list(sim.qc_quantize_op_dict.values())[0]
    if input_quantizer.enabled and conv0_param_quantizer.enabled:
        quantizer_1 = (
            (input_quantizer.bitwidth, QuantizationDataType.int),
            (conv0_param_quantizer.bitwidth, QuantizationDataType.int)
        )
    else:
        quantizer_1 = "fp32"

    # quantizer group 2
    fc_weight_quantizer = sim.qc_quantize_op_dict['fc.weight']
    fc_output_quantizer = sim.qc_quantize_op_dict['/avgpool/AveragePool_output_0']
    if fc_weight_quantizer.enabled and fc_output_quantizer.enabled:
        quantizer_2 = (
            (fc_output_quantizer.bitwidth, QuantizationDataType.int),
            (fc_weight_quantizer.bitwidth, QuantizationDataType.int)
        )
    else:
        quantizer_2 = "fp32"

    key = (quantizer_1, quantizer_2)
    return eval_score_lookup_table[key]


@pytest.fixture
def eval_callback_phase1(sim):
    return CallbackFunc(eval_func, [phase1_eval_score_lookup_table, sim])


@pytest.fixture
def eval_callback_phase2(sim):
    return CallbackFunc(eval_func, [phase2_eval_score_lookup_table, sim])


@pytest.fixture
def candidates():
    # ((activation bitwidth, activation data type), (param bitwidth, param data type))
    return [W16A16, W8A16, W16A8]

@pytest.fixture
def model():
    return single_residual_model()

@pytest.fixture
def sim(model):
    # Quantize the model to default bitwidth
    sim = QuantizationSimModel(model)
    return sim

@pytest.fixture
def sim_supported_kernel():
    model = single_residual_model()
    quantsim_config = {
        "defaults": {
            "ops": {
                "is_output_quantized": "True"
            },
            "params": {
                "is_quantized": "True"
            },
            "supported_kernels": [
                {
                    "activation": {
                        "bitwidth": 16,
                        "dtype": "int"
                    },
                    "param": {
                        "bitwidth": 16,
                        "dtype": "int"
                    }
                },
                {
                    "activation": {
                        "bitwidth": 16,
                        "dtype": "float"
                    },
                    "param": {
                        "bitwidth": 16,
                        "dtype": "float"
                    }
                }
            ]
        },
        "params": {
            "bias": {
                "is_quantized": "False"
            }
        },
        "op_type": {},
        "supergroups": [
        ],
        "model_input": {
            "is_input_quantized": "True"
        },
        "model_output": {}
    }

    with tempfile.TemporaryDirectory() as tempdir:
        with open(os.path.join(tempdir, 'quantsim_config.json'), 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model, config_file=os.path.join(tempdir, 'quantsim_config.json'))
        return sim


@pytest.fixture
def results_dir():
    with tempfile.TemporaryDirectory() as tempdir:
        os.makedirs(os.path.join(tempdir, ".cache"))
        yield tempdir
        shutil.rmtree(tempdir)

def _get_quantizer_name_op_type(sim):
    """
    Get quantizer_name -> Connected_graph_op type map.
    """
    quantizer_to_op_type = {}
    for cg_op in sim.connected_graph.ordered_ops:
        input_quantizers, output_quantizers, param_quantizers = sim.get_op_quantizers(cg_op)
        print(input_quantizers, output_quantizers, param_quantizers)
        if input_quantizers:
            for inp_qtz in input_quantizers:
                for quantizer_name, quantizer in sim.qc_quantize_op_dict.items():
                    if inp_qtz == quantizer:
                        quantizer_to_op_type[quantizer_name] = [cg_op.type]
        if output_quantizers:
            for out_qtz in output_quantizers:
                for quantizer_name, quantizer in sim.qc_quantize_op_dict.items():
                    if out_qtz == quantizer:
                        quantizer_to_op_type[quantizer_name] = [cg_op.type]
        if param_quantizers:
            for param_qtz in param_quantizers.values():
                for quantizer_name, quantizer in sim.qc_quantize_op_dict.items():
                    if param_qtz == quantizer:
                        quantizer_to_op_type[quantizer_name] = [cg_op.type]
    return quantizer_to_op_type


class TestAMPv1:
    def test_phase1(self, sim, candidates, forward_pass_callback, eval_callback_phase1, results_dir):
        algo = GreedyMixedPrecisionAlgo(sim, candidates, eval_callback_phase1, unittest.mock.MagicMock(),
                                        results_dir, True, forward_pass_callback)
        algo.set_baseline()

        candidate = algo.quantizer_groups[0].get_candidate(algo._module_name_dict)
        # Check if quantizer group is set to maximum bitwidth
        assert algo.baseline_candidate == candidate

        active_quantizers = {
            quantizer_group: quantizer_group.get_active_quantizers(algo._module_name_dict)
            for quantizer_group in algo.quantizer_groups
        }

        call_count = 0

        def assert_only_one_quantizer_group_enabled(*args, **kwargs):
            nonlocal call_count

            found_quantizer_groups = []
            for quantizer_group in algo.quantizer_groups:
                if quantizer_group.get_active_quantizers(algo._module_name_dict):
                    found_quantizer_groups.append(quantizer_group.get_active_quantizers(algo._module_name_dict))

            if call_count < len(algo.quantizer_groups) * (len(candidates) - 1):
                # During phase 1 loop, only one quantizer group can be activated at a time
                # TODO: the below commented out assert should the one to check. However, current logic for finding
                # quantizer groups does not correctly identify that Concat -> Gemm should not be a quantizer group.
                # This results in Gemm's weight showing up as an independent quantizer group, as well as paired with
                # AveragePool as a second quantizer group.
                # assert len(found_quantizer_groups) == 1
                # As a result of fc.weight being its own quantizer group, we run into another issue where there are no
                # quantizers to disable associated with fc.weight, since its quantizer was already disabled when
                # processing the earlier AveragePool_12 -> fc quantizer group. Thus, len(found_quantizer_group) = 0 when
                # dealing with fc.weight standalone quantizer group.
                assert len(found_quantizer_groups) <= 2
                # The below check should be removed when the above bug is fixed. The check is to make sure that the only
                # case when found_quantizer_groups has more than one entry is the known issue.
                if len(found_quantizer_groups) == 2:
                    avgpool_output_name = algo._sim.connected_graph.get_all_ops()['AveragePool_12'].get_module().output[0]
                    avgpool_quantizer = algo._sim.qc_quantize_op_dict[avgpool_output_name]
                    fc_weight_quantizer = algo._sim.qc_quantize_op_dict['fc.weight']
                    assert [fc_weight_quantizer] in found_quantizer_groups
                    assert [avgpool_quantizer, fc_weight_quantizer] in found_quantizer_groups

            call_count += 1

        with unittest.mock.patch(
                'aimet_onnx.amp.mixed_precision_algo.EvalCallbackFactory.sqnr',
                side_effect=assert_only_one_quantizer_group_enabled
        ):
            accuracy_list = algo._create_and_save_accuracy_list(algo.baseline_candidate)

        # All the active quantizers should be still active
        for quantizer_group in algo.quantizer_groups:
            assert active_quantizers[quantizer_group] == \
                   quantizer_group.get_active_quantizers(algo._module_name_dict)

        assert len(accuracy_list) == 20
        # Check if accuracy list is in descending order
        assert accuracy_list[0][2] >= accuracy_list[1][2]
        assert accuracy_list[1][2] >= accuracy_list[2][2]
        assert accuracy_list[2][2] >= accuracy_list[3][2]
        assert accuracy_list[3][2] >= accuracy_list[4][2]
        assert accuracy_list[4][2] >= accuracy_list[5][2]

    def test_phase2_brute_force(self, sim, candidates, forward_pass_callback,
                                eval_callback_phase1, eval_callback_phase2, results_dir):

        allowed_accuracy_drop = 0.12
        algo = GreedyMixedPrecisionAlgo(sim, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)
        algo.set_baseline()
        pareto_front_list = self._run_phase2(algo, allowed_accuracy_drop, brute_force_search)

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == 4

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, [phase2_eval_score_lookup_table, sim])
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

        # Test 3. Check bitops
        starting_bit_ops = calculate_starting_bit_ops(
            algo._mac_dict, ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
        )
        running_bit_ops = starting_bit_ops - algo._mac_dict['/conv1/Conv'] * 16 * 16 + \
                          algo._mac_dict['/conv1/Conv'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert relative_bit_ops == pareto_front_list[0][0]
        assert relative_bit_ops == pareto_front_list[1][0]

        running_bit_ops = running_bit_ops - algo._mac_dict['/fc/Gemm'] * 16 * 16 + algo._mac_dict['/fc/Gemm'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops
        assert relative_bit_ops == pareto_front_list[2][0]

    @pytest.mark.parametrize("allowed_accuracy_drop, len_of_pareto_list", [(0.13, 3), (0.1, 4)])
    def test_phase2_interpolation(self, sim, candidates, forward_pass_callback,
                                  eval_callback_phase1, eval_callback_phase2, results_dir,
                                  allowed_accuracy_drop, len_of_pareto_list):
        algo = GreedyMixedPrecisionAlgo(sim, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)

        pareto_front_list = self._run_phase2(algo, allowed_accuracy_drop, interpolation_search)

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == len_of_pareto_list

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, [phase2_eval_score_lookup_table, sim])
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

    @pytest.mark.parametrize("allowed_accuracy_drop, len_of_pareto_list", [(0.055, 2), (0.1, 4)])
    def test_phase2_binary(self, sim, candidates, forward_pass_callback,
                           eval_callback_phase1, eval_callback_phase2, results_dir, allowed_accuracy_drop,
                           len_of_pareto_list):
        algo = GreedyMixedPrecisionAlgo(sim, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)

        pareto_front_list = self._run_phase2(algo, allowed_accuracy_drop, binary_search)

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == len_of_pareto_list

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, [phase2_eval_score_lookup_table, sim])
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

    def _run_phase2(self, algo, allowed_accuracy_drop, search_algo):
        algo.baseline_candidate = W16A16
        algo.min_candidate = W16A8
        fp32_acc = 1.0

        accuracy_list = [
            (algo.quantizer_groups[0], W8A16, phase1_eval_score_lookup_table[(W8A16, "fp32")], 100),
            (algo.quantizer_groups[0], W16A8, phase1_eval_score_lookup_table[(W16A8, "fp32")], 90),
            (algo.quantizer_groups[8], W8A16, phase1_eval_score_lookup_table[("fp32", W8A16)], 80),
            (algo.quantizer_groups[8], W16A8, phase1_eval_score_lookup_table[("fp32", W16A8)], 70),
        ]

        return algo._create_pareto_front_list(allowed_accuracy_drop, accuracy_list, fp32_acc,
                                              algo.baseline_candidate, algo.min_candidate, search_algo, phase2_reverse = False)

    def test_supported_candidates_1(
            self, sim_supported_kernel, candidates, forward_pass_callback, eval_callback_phase1, eval_callback_phase2, results_dir
    ):
        """
        Pass in vanilla config file without any specialized supported_kernels and verify the generated candidates in
        quantizer_groups of GreedyMixedPrecisionAlgo object
        """
        # Create an accuracy list
        algo = GreedyMixedPrecisionAlgo(sim_supported_kernel, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback, use_all_amp_candidates=False)

        assert len(algo._supported_candidates_per_quantizer_group.keys()) == 13

        default_supported_kernels = [((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
                                     ((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
                                     ((8, QuantizationDataType.float), (16, QuantizationDataType.float))]

        for quantizer, quantizer_candidates in algo._supported_candidates_per_quantizer_group.items():
            # verify to make sure the candidates returned is always part of amp_candidates and they are part of
            # "Defaults"
            for c in quantizer_candidates:
                assert c in default_supported_kernels
                assert c in candidates

    def test_supported_candidates_2(
            self, candidates, forward_pass_callback, eval_callback_phase1, eval_callback_phase2, results_dir
    ):
        """
        Pass in vanilla config file without any specialized supported_kernels and verify the generated candidates in
        quantizer_groups of GreedyMixedPrecisionAlgo object
        """
        model = single_residual_model()
        quantsim_config = {
            "defaults": {
                "ops": {
                    "is_output_quantized": "True"
                },
                "params": {
                    "is_quantized": "True"
                },
                "supported_kernels": [
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 16,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 8,
                            "dtype": "int"
                        }
                    },
                    {
                        "activation": {
                            "bitwidth": 8,
                            "dtype": "int"
                        },
                        "param": {
                            "bitwidth": 16,
                            "dtype": "int"
                        }
                    }
                ]
            },
            "params": {
                "bias": {
                    "is_quantized": "False"
                }
            },
            "op_type": {
                "Conv": {
                    "supported_kernels":
                        [
                            {
                                "activation": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "float"
                                }
                            },
                            {
                                "activation": {
                                    "bitwidth": 8,
                                    "dtype": "int"
                                },
                                "param": {
                                    "bitwidth": 16,
                                    "dtype": "int"
                                }
                            },
                        ],
                    "is_input_quantized": "True",
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {
                            "is_quantized": "True"
                        },
                        "bias": {
                            "is_quantized": "False"
                        }
                    }
                }
            },
            "supergroups": [
            ],
            "model_input": {
                "is_input_quantized": "True"
            },
            "model_output": {}
        }


        with open(os.path.join(results_dir, 'quantsim_config.json'), 'w') as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(model, config_file=os.path.join(results_dir, 'quantsim_config.json'))

        # Create an accuracy list
        algo = GreedyMixedPrecisionAlgo(sim, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback, use_all_amp_candidates=False)

        assert len(algo._supported_candidates_per_quantizer_group.keys()) == 13

        # default_supported_kernels and conv_supported_kernels are the configurations added in the json file above.
        default_supported_kernels = [((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
                                     ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
                                     ((8, QuantizationDataType.int), (16, QuantizationDataType.int))]

        conv_supported_kernels = [((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
                                  ((8, QuantizationDataType.int), (16, QuantizationDataType.int))]
        quantizer_to_op_type = _get_quantizer_name_op_type(sim)
        quantizer_to_op_type['output'] = ['Gemm']

        for quantizer, quantizer_candidates in algo._supported_candidates_per_quantizer_group.items():
            quantizers = sorted(set(itertools.chain(quantizer.activation_quantizers,
                                                    quantizer.parameter_quantizers)))
            onnx_types = []

            for q in quantizers:
                onnx_types.append(quantizer_to_op_type[q])

            # verify to make sure the candidates returned is always part of amp_candidates and they are part of
            # "Defaults" or "Conv" appropriately
            for c in quantizer_candidates:
                if ['Conv'] in onnx_types:
                    assert c in conv_supported_kernels
                else:
                    assert c in default_supported_kernels


class TestAMPv2:
    def test_compute_sqnr(self):
        """ Verify _compute_sqnr() method """
        for noise in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            orig_tensor = np.random.randn(10, 10)
            noisy_tensor = orig_tensor + noise
            sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
            expected_sqnr = np.power(orig_tensor, 2).mean() / (noise ** 2 + 0.0001)
            assert np.isclose(sqnr, expected_sqnr)

        orig_tensor = np.ones((10, 10))
        noisy_tensor = np.zeros((10, 10))
        sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
        expected_sqnr = 1
        assert np.isclose(sqnr, expected_sqnr, rtol=1e-3)

        orig_tensor = np.ones((10, 10)) * 2
        noisy_tensor = np.zeros((10, 10))
        sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
        expected_sqnr = 1
        assert np.isclose(sqnr, expected_sqnr, rtol=1e-3)

        orig_tensor = np.ones((10, 10)) * 2
        noisy_tensor = np.ones((10, 10))
        sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
        expected_sqnr = 4
        assert np.isclose(sqnr, expected_sqnr, rtol=1e-3)

    def test_eval_callback_factory(self, sim):
        np.random.seed(0)
        dummy_input = np.random.rand(1, 3, 32, 32).astype(np.float32)

        class _Dataset(DataLoader):
            def __init__(self):
                super(_Dataset, self).__init__(dummy_input, 32, 1)
            def __iter__(self):
                yield dummy_input

        evaluator = EvalCallbackFactory(_Dataset()).sqnr(sim)
        sqnr = evaluator.func(sim.session, None)
        assert sqnr != 0.0
