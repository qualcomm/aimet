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

import copy
import itertools
import tempfile
import unittest
import unittest.mock
import os
import pickle
import shutil
import json
from aimet_torch.v1.qc_quantize_op import QcQuantizeWrapper
import numpy as np
import math
import functools

import torch
import torch.nn as nn

import pytest

from aimet_torch import onnx_utils
from aimet_torch.v1.quantsim import QuantizationSimModel

from aimet_common.defs import QuantizationDataType
from aimet_common.amp.utils import AMPSearchAlgo, calculate_starting_bit_ops, sort_accuracy_list
from aimet_common.amp.mixed_precision_algo import interpolation_search, brute_force_search, binary_search
from aimet_torch.amp.mixed_precision_algo import (
    GreedyMixedPrecisionAlgo,
    _compute_sqnr,
)
from aimet_torch.amp.quantizer_groups import QuantizerGroup
from aimet_common.defs import CallbackFunc


DEFAULT_BITWIDTH = 16
INPUT_SHAPE = (1, 1, 10, 10)


class SmallMnist(nn.Module):
    def __init__(self):
        super(SmallMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv2(x)
        return x


def forward_fn(model, _):
    torch.manual_seed(10)
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn(INPUT_SHAPE))


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
    (W16A8,  W8A8): 0.81,
    (W8A16,  W8A8): 0.82,
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

def eval_func(model, eval_score_lookup_table):
    # quantizer group 1
    input_quantizer = model.conv1.input_quantizers[0]
    conv1_param_quantizer = model.conv1.param_quantizers["weight"]
    if input_quantizer.enabled and conv1_param_quantizer.enabled:
        quantizer_1 = (
            (input_quantizer.bitwidth, input_quantizer.data_type),
            (conv1_param_quantizer.bitwidth, conv1_param_quantizer.data_type)
        )
    else:
        quantizer_1 = "fp32"

    # quantizer group 2
    conv2_param_quantizer = model.conv2.param_quantizers["weight"]
    relu1_output_quantizer = model.relu1.output_quantizers[0]
    if conv2_param_quantizer.enabled and relu1_output_quantizer.enabled:
        quantizer_2 = (
            (relu1_output_quantizer.bitwidth, relu1_output_quantizer.data_type),
            (conv2_param_quantizer.bitwidth, conv2_param_quantizer.data_type)
        )
    else:
        quantizer_2 = "fp32"

    key = (quantizer_1, quantizer_2)
    return eval_score_lookup_table[key]


@pytest.fixture
def eval_callback_phase1():
    return CallbackFunc(eval_func, phase1_eval_score_lookup_table)


@pytest.fixture
def eval_callback_phase2():
    return CallbackFunc(eval_func, phase2_eval_score_lookup_table)


@pytest.fixture(scope="session")
def model():
    return SmallMnist().to(device='cpu')


@pytest.fixture(scope="session")
def dummy_input():
    return torch.randn(INPUT_SHAPE)


@pytest.fixture
def candidates():
    # ((activation bitwidth, activation data type), (param bitwidth, param data type))
    return [W16A16, W8A16, W16A8]


@pytest.fixture
def sim(model, dummy_input):
    # Quantize the model to default bitwidth
    sim = QuantizationSimModel(model,
                               default_param_bw=DEFAULT_BITWIDTH,
                               default_output_bw=DEFAULT_BITWIDTH,
                               dummy_input=dummy_input)
    sim.compute_encodings(forward_fn, forward_pass_callback_args=None)
    return sim


@pytest.fixture
def results_dir():
    with tempfile.TemporaryDirectory() as path:
        os.makedirs(os.path.join(path, ".cache"))
        yield path
        shutil.rmtree(path)


@pytest.fixture(autouse=True)
def torch_manual_seed():
    torch.manual_seed(10)


class TestAutoMixedPrecision:
    def test_eval_func_edge_cases(
            self, sim, dummy_input, candidates, forward_pass_callback
    ):
        # Edge case: Non-float eval score
        eval_callback = CallbackFunc(lambda *_: (0.1, 0.2), None)
        with pytest.raises(RuntimeError):
            algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback, eval_callback,
                                            None, True, forward_pass_callback)
            algo.set_baseline()

    def test_early_exit_best_quantized_accuracy_inadequate(self, sim, dummy_input, candidates, eval_callback_phase1,
                                                           eval_callback_phase2, results_dir, forward_pass_callback):
        json_file_path = os.path.join(results_dir, 'pareto_list.json')
        pickle_file_path = os.path.join(results_dir, '.cache', 'pareto_list.pkl')

        # Remove the files before invoking AMP
        if (os.path.isfile(json_file_path)):
            try:
                os.remove(json_file_path)
            except:
                assert (False)

        if (os.path.isfile(pickle_file_path)):
            try:
                os.remove(pickle_file_path)
            except:
                assert (False)

        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)
        algo.run(allowed_accuracy_drop=0.0001)

        baseline_accuracy, _= algo._get_best_candidate()
        assert algo._final_eval_score == baseline_accuracy
        assert (not os.path.isfile(json_file_path))
        assert (not os.path.isfile(pickle_file_path))

    def test_phase1(self, sim, dummy_input, candidates, forward_pass_callback, eval_callback_phase1, results_dir):
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, unittest.mock.MagicMock(),
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

            all_active_quantizers = []
            for quantizer_group in algo.quantizer_groups:
                all_active_quantizers +=\
                        quantizer_group.get_active_quantizers(algo._module_name_dict)

            if call_count < len(algo.quantizer_groups) * (len(candidates) - 1):
                # During phase 1 loop, only one quantizer group can be activated at a time
                assert all_active_quantizers in active_quantizers.values()

            call_count += 1

        with unittest.mock.patch(
            'aimet_torch._base.amp.mixed_precision_algo.EvalCallbackFactory.sqnr',
            side_effect=assert_only_one_quantizer_group_enabled
        ):
            accuracy_list = algo._create_and_save_accuracy_list(algo.baseline_candidate)

        # All the active quantizers should be still active
        for quantizer_group in algo.quantizer_groups:
            active_quantizers[quantizer_group] ==\
                    quantizer_group.get_active_quantizers(algo._module_name_dict)


        assert len(accuracy_list) == 6
        # Check if accuracy list is in descending order
        assert accuracy_list[0][2] >= accuracy_list[1][2]
        assert accuracy_list[1][2] >= accuracy_list[2][2]
        assert accuracy_list[2][2] >= accuracy_list[3][2]
        assert accuracy_list[3][2] >= accuracy_list[4][2]
        assert accuracy_list[4][2] >= accuracy_list[5][2]

    def test_phase1_reverse(self, sim, dummy_input, candidates, forward_pass_callback, eval_callback_phase1, results_dir):
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, unittest.mock.MagicMock(),
                                        results_dir, True, forward_pass_callback, phase2_reverse = True)
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

            all_active_quantizers = []
            for quantizer_group in algo.quantizer_groups:
                all_active_quantizers +=\
                        quantizer_group.get_active_quantizers(algo._module_name_dict)

            if call_count < len(algo.quantizer_groups) * (len(candidates) - 1):
                # During phase 1 loop, only one quantizer group can be activated at a time
                assert all_active_quantizers in active_quantizers.values()

            call_count += 1

        with unittest.mock.patch(
            'aimet_torch._base.amp.mixed_precision_algo.EvalCallbackFactory.sqnr',
            side_effect=assert_only_one_quantizer_group_enabled
        ):
            accuracy_list = algo._create_and_save_accuracy_list(algo.baseline_candidate)

        # All the active quantizers should be still active
        for quantizer_group in algo.quantizer_groups:
            active_quantizers[quantizer_group] ==\
                    quantizer_group.get_active_quantizers(algo._module_name_dict)


        assert len(accuracy_list) == 6
        # Check if accuracy list is in descending order
        assert accuracy_list[0][2] >= accuracy_list[1][2]
        assert accuracy_list[1][2] >= accuracy_list[2][2]
        assert accuracy_list[2][2] >= accuracy_list[3][2]
        assert accuracy_list[3][2] >= accuracy_list[4][2]
        assert accuracy_list[4][2] >= accuracy_list[5][2]

    def test_save_load_accuracy_list(
            self, sim, dummy_input, candidates, forward_pass_callback, eval_callback_phase1, eval_callback_phase2, results_dir
    ):
        # Create an accuracy list
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)
        algo.set_baseline()
        accuracy_list = algo._create_and_save_accuracy_list(algo.baseline_candidate)

        # Load accuracy list
        # Note: When accuracy list is loaded new modules and quantizer groups get created so reference don't match with
        # quantizer groups in algo
        file_path = os.path.join(results_dir, '.cache', 'accuracy_list.pkl')
        with open(file_path, 'rb') as file:
            parsed_accuracy_list = pickle.load(file)

        # Check if parsed list is sorted
        assert parsed_accuracy_list[0][2] >= parsed_accuracy_list[1][2]
        assert parsed_accuracy_list[1][2] >= parsed_accuracy_list[2][2]
        assert parsed_accuracy_list[2][2] >= parsed_accuracy_list[3][2]

        # Replace parsed lists quantizer groups with the ones in algo
        # parsed_accuracy_list = aimet_common.amp.utils.map_quantizer_groups_for_acc_list(parsed_accuracy_list, algo.quantizer_groups)
        assert parsed_accuracy_list == accuracy_list

    def test_save_load_accuracy_list_reverse(
            self, sim, dummy_input, candidates, forward_pass_callback, eval_callback_phase1, eval_callback_phase2, results_dir):
        # Create an accuracy list
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback, phase2_reverse = True)
        algo.set_baseline()
        accuracy_list = algo._create_and_save_accuracy_list(algo.baseline_candidate)

        # Load accuracy list
        # Note: When accuracy list is loaded new modules and quantizer groups get created so reference don't match with
        # quantizer groups in algo
        file_path = os.path.join(results_dir, '.cache', 'accuracy_list.pkl')
        with open(file_path, 'rb') as file:
            parsed_accuracy_list = pickle.load(file)

        # Check if parsed list is sorted
        assert parsed_accuracy_list[0][2] >= parsed_accuracy_list[1][2]
        assert parsed_accuracy_list[1][2] >= parsed_accuracy_list[2][2]
        assert parsed_accuracy_list[2][2] >= parsed_accuracy_list[3][2]

        # Replace parsed lists quantizer groups with the ones in algo
        # parsed_accuracy_list = aimet_common.amp.utils.map_quantizer_groups_for_acc_list(parsed_accuracy_list, algo.quantizer_groups)
        assert parsed_accuracy_list == accuracy_list

    def test_phase2_brute_force(self, sim, dummy_input, candidates, forward_pass_callback,
                     eval_callback_phase1, eval_callback_phase2, results_dir):
        allowed_accuracy_drop = 0.12 # i.e. target accuracy = 0.88
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)

        pareto_front_list = self._run_phase2(algo, allowed_accuracy_drop, brute_force_search)

        # NOTE: Expected visit order
        #  candidate       | eval score | visit order   | corresponding entry in pareto curve
        # -----------------|------------|---------------|------------------------------------
        #  (W8A16, W16A16) | 0.94       | 0             | pareto_list[0]
        #  (W16A8, W16A16) | 0.93       | 1             | pareto_list[1]
        #  (W16A8, W8A16)  | 0.89       | 2             | pareto_list[2]
        #  (W16A8, W16A8)  | 0.85       | 3             | pareto_list[3]

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == 4

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, phase2_eval_score_lookup_table)
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

        # Test 3. Check bitops
        starting_bit_ops = calculate_starting_bit_ops(
            algo._mac_dict, ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
        )
        running_bit_ops = starting_bit_ops - algo._mac_dict['conv1'] * 16 * 16 + \
                          algo._mac_dict['conv1'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert relative_bit_ops == pareto_front_list[0][0]
        assert relative_bit_ops == pareto_front_list[1][0]

        running_bit_ops = running_bit_ops - algo._mac_dict['conv2'] * 16 * 16 + algo._mac_dict['conv2'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops
        assert relative_bit_ops == pareto_front_list[2][0]

    def test_phase2_brute_force_reverse(self, sim, dummy_input, candidates, forward_pass_callback,
                     eval_callback_phase1, eval_callback_phase2, results_dir):
        allowed_accuracy_drop = 0.12 # i.e. target accuracy = 0.88
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback, phase2_reverse = True)

        pareto_front_list = self._run_phase2_reverse(algo, allowed_accuracy_drop, brute_force_search)


        # NOTE: Expected visit order
        #  candidate       | eval score | visit order   | corresponding entry in pareto curve
        # -----------------|------------|---------------|------------------------------------
        #  (W16A8, W8A16)  | 0.89       | 0             | pareto_list[0]


        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == 1

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, phase2_eval_score_lookup_table)

        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

        # Test 3. Check bitops
        starting_bit_ops = calculate_starting_bit_ops(
            algo._mac_dict, ((8, QuantizationDataType.int), (16, QuantizationDataType.int))
        )

        running_bit_ops = starting_bit_ops - algo._mac_dict['conv2'] * 8 * 16 + \
                          algo._mac_dict['conv2'] * 16 * 8

        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert relative_bit_ops == pareto_front_list[0][0]

    def test_phase2_interpolation_0(self, sim, dummy_input, candidates, forward_pass_callback,
                                    eval_callback_phase1, eval_callback_phase2, results_dir):
        allowed_accuracy_drop = 0.13 # i.e. target accuracy = 0.87
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)

        pareto_front_list = self._run_phase2(algo, allowed_accuracy_drop, interpolation_search)

        # NOTE: Expected visit order
        #  candidate       | eval score | visit order   | corresponding entry in pareto curve
        # -----------------|------------|---------------|------------------------------------
        #  (W8A16, W16A16) | 0.94       | 1             | pareto_list[0]
        #  (W16A8, W16A16) | 0.93       | (not visited) | N/A
        #  (W16A8, W8A16)  | 0.89       | 2             | pareto_list[1]
        #  (W16A8, W16A8)  | 0.85       | 0             | pareto_list[2]

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == 3

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, phase2_eval_score_lookup_table)
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

        # Test 3. Check bitops
        starting_bit_ops = calculate_starting_bit_ops(
            algo._mac_dict, ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
        )
        running_bit_ops = starting_bit_ops - algo._mac_dict['conv1'] * 16 * 16 + \
                          algo._mac_dict['conv1'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert relative_bit_ops == pareto_front_list[0][0]

        running_bit_ops = running_bit_ops - algo._mac_dict['conv2'] * 16 * 16 + algo._mac_dict['conv2'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops
        assert relative_bit_ops == pareto_front_list[1][0]

    def test_phase2_interpolation_1(self, sim, dummy_input, candidates, forward_pass_callback,
                                    eval_callback_phase1, eval_callback_phase2, results_dir):
        allowed_accuracy_drop = 0.1 # i.e. target accuracy = 0.9
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)

        pareto_front_list = self._run_phase2(algo, allowed_accuracy_drop, interpolation_search)

        # NOTE: Expected visit order
        #  candidate       | eval score | visit order   | corresponding entry in pareto curve
        # -----------------|------------|---------------|------------------------------------
        #  (W8A16, W16A16) | 0.94       | 1             | pareto_list[0]
        #  (W16A8, W16A16) | 0.93       | 3             | pareto_list[1]
        #  (W16A8, W8A16)  | 0.89       | 2             | pareto_list[2]
        #  (W16A8, W16A8)  | 0.85       | 0             | pareto_list[3]

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == 4

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, phase2_eval_score_lookup_table)
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

        # Test 3. Check bitops
        starting_bit_ops = calculate_starting_bit_ops(
            algo._mac_dict, ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
        )
        running_bit_ops = starting_bit_ops - algo._mac_dict['conv1'] * 16 * 16 + \
                          algo._mac_dict['conv1'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert relative_bit_ops == pareto_front_list[0][0]
        assert relative_bit_ops == pareto_front_list[1][0]

        running_bit_ops = running_bit_ops - algo._mac_dict['conv2'] * 16 * 16 + algo._mac_dict['conv2'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops
        assert relative_bit_ops == pareto_front_list[2][0]
        assert relative_bit_ops == pareto_front_list[3][0]

    def test_phase2_interpolation_reverse(self, sim, dummy_input, candidates, forward_pass_callback,
                                    eval_callback_phase1, eval_callback_phase2, results_dir):
        allowed_accuracy_drop = 0.12 # i.e. target accuracy = 0.88
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback, phase2_reverse = True)

        pareto_front_list = self._run_phase2_reverse(algo, allowed_accuracy_drop, interpolation_search)

        # NOTE: Expected visit order
        #  candidate       | eval score | visit order   | corresponding entry in pareto curve
        # -----------------|------------|---------------|------------------------------------
        #  (W16A8, W8A16)  | 0.89       | 0             | pareto_list[0]

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == 1

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, phase2_eval_score_lookup_table)
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

        # Test 3. Check bitops
        starting_bit_ops = calculate_starting_bit_ops(
            algo._mac_dict, ((8, QuantizationDataType.int), (16, QuantizationDataType.int))
        )
        running_bit_ops = starting_bit_ops - algo._mac_dict['conv2'] * 8 * 16 + \
                          algo._mac_dict['conv2'] * 16 * 8
        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert relative_bit_ops == pareto_front_list[0][0]

    def test_phase2_binary_0(self, sim, dummy_input, candidates, forward_pass_callback,
                             eval_callback_phase1, eval_callback_phase2, results_dir):
        allowed_accuracy_drop = 0.12 # i.e. target accuracy = 0.88
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)

        pareto_front_list = self._run_phase2(algo, allowed_accuracy_drop, binary_search)

        # NOTE: Expected visit order
        #  candidate       | eval score | visit order   | corresponding entry in pareto curve
        # -----------------|------------|---------------|------------------------------------
        #  (W8A16, W16A16) | 0.94       | 1             | pareto_list[0]
        #  (W16A8, W16A16) | 0.93       | (not visited) | N/A
        #  (W16A8, W8A16)  | 0.89       | 2             | pareto_list[1]
        #  (W16A8, W16A8)  | 0.85       | 0             | pareto_list[2]

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == 3

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, phase2_eval_score_lookup_table)
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

        # Test 3. Check bitops
        starting_bit_ops = calculate_starting_bit_ops(
            algo._mac_dict, ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
        )
        running_bit_ops = starting_bit_ops - algo._mac_dict['conv1'] * 16 * 16 + \
                          algo._mac_dict['conv1'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert relative_bit_ops == pareto_front_list[0][0]

        running_bit_ops = running_bit_ops - algo._mac_dict['conv2'] * 16 * 16 + algo._mac_dict['conv2'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops
        assert relative_bit_ops == pareto_front_list[1][0]
        assert relative_bit_ops == pareto_front_list[2][0]

    def test_phase2_binary_1(self, sim, dummy_input, candidates, forward_pass_callback,
                             eval_callback_phase1, eval_callback_phase2, results_dir):
        allowed_accuracy_drop = 0.1 # i.e. target accuracy = 0.9
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)

        pareto_front_list = self._run_phase2(algo, allowed_accuracy_drop, binary_search)

        # NOTE: Expected visit order
        #  candidate       | eval score | visit order   | corresponding entry in pareto curve
        # -----------------|------------|---------------|------------------------------------
        #  (W8A16, W16A16) | 0.94       | 1             | pareto_list[0]
        #  (W16A8, W16A16) | 0.93       | 2             | pareto_list[1]
        #  (W16A8, W8A16)  | 0.89       | 3             | pareto_list[2]
        #  (W16A8, W16A8)  | 0.85       | 0             | pareto_list[3]

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == 4

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, phase2_eval_score_lookup_table)
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

        # Test 3. Check bitops
        starting_bit_ops = calculate_starting_bit_ops(
            algo._mac_dict, ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
        )
        running_bit_ops = starting_bit_ops - algo._mac_dict['conv1'] * 16 * 16 + \
                          algo._mac_dict['conv1'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert relative_bit_ops == pareto_front_list[0][0]
        assert relative_bit_ops == pareto_front_list[1][0]

        running_bit_ops = running_bit_ops - algo._mac_dict['conv2'] * 16 * 16 + algo._mac_dict['conv2'] * 8 * 16
        relative_bit_ops = running_bit_ops / starting_bit_ops
        assert relative_bit_ops == pareto_front_list[2][0]
        assert relative_bit_ops == pareto_front_list[3][0]

    def test_phase2_binary_reverse(self, sim, dummy_input, candidates, forward_pass_callback,
                             eval_callback_phase1, eval_callback_phase2, results_dir):
        allowed_accuracy_drop = 0.12 # i.e. target accuracy = 0.88
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback, phase2_reverse = True)

        pareto_front_list = self._run_phase2_reverse(algo, allowed_accuracy_drop, binary_search)

        # NOTE: Expected visit order
        #  candidate       | eval score | visit order   | corresponding entry in pareto curve
        # -----------------|------------|---------------|------------------------------------
        #  (W16A8, W8A16)  | 0.89       | 0             | pareto_list[0]

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == 1

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(sim.model, phase2_eval_score_lookup_table)
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

        # Test 3. Check bitops
        starting_bit_ops = calculate_starting_bit_ops(
            algo._mac_dict, ((8, QuantizationDataType.int), (16, QuantizationDataType.int))
        )
        running_bit_ops = starting_bit_ops - algo._mac_dict['conv2'] * 8 * 16 + \
                          algo._mac_dict['conv2'] * 16 * 8
        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert relative_bit_ops == pareto_front_list[0][0]

    @pytest.mark.parametrize("search_algo", [AMPSearchAlgo.BruteForce,
                                             AMPSearchAlgo.Interpolation,
                                             AMPSearchAlgo.Binary])
    def test_phase2_fallback_to_baseline(self, sim, dummy_input, candidates, forward_pass_callback,
                                         eval_callback_phase1, eval_callback_phase2, results_dir, search_algo):
        if search_algo == AMPSearchAlgo.BruteForce:
            search_algo = brute_force_search
        elif search_algo == AMPSearchAlgo.Interpolation:
            search_algo = interpolation_search
        elif search_algo == AMPSearchAlgo.Binary:
            search_algo = binary_search
        else:
            raise RuntimeError

        allowed_accuracy_drop = 0 # i.e. target accuracy = 1.0
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)

        _ = self._run_phase2(algo, allowed_accuracy_drop, search_algo)

        #  candidate       | eval score |
        # -----------------|------------|
        #  (W8A16, W16A16) | 0.94       |
        #  (W16A8, W16A16) | 0.93       |
        #  (W16A8, W8A16)  | 0.89       |
        #  (W16A8, W16A8)  | 0.85       |

        # Assert all layers are in W16A16 (baseline)
        for module in algo._sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                for quantizer in module.param_quantizers.values():
                    if quantizer.enabled:
                        assert quantizer.bitwidth == 16

                for quantizer in module.input_quantizers:
                    if quantizer.enabled:
                        assert quantizer.bitwidth == 16

                for quantizer in module.output_quantizers:
                    if quantizer.enabled:
                        assert quantizer.bitwidth == 16

    @pytest.mark.parametrize("search_algo", [AMPSearchAlgo.BruteForce,
                                             AMPSearchAlgo.Interpolation,
                                             AMPSearchAlgo.Binary])

    def test_phase2_fallback_to_baseline_reverse(self, sim, dummy_input, candidates, forward_pass_callback,
                                         eval_callback_phase1, eval_callback_phase2, results_dir, search_algo):
        if search_algo == AMPSearchAlgo.BruteForce:
            search_algo = brute_force_search
        elif search_algo == AMPSearchAlgo.Interpolation:
            search_algo = interpolation_search
        elif search_algo == AMPSearchAlgo.Binary:
            search_algo = binary_search
        else:
            raise RuntimeError

        allowed_accuracy_drop = 0 # i.e. target accuracy = 1.0
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback, phase2_reverse = True)

        _ = self._run_phase2_reverse(algo, allowed_accuracy_drop, search_algo)

        #  candidate       | eval score |
        # -----------------|------------|
        #  (W16A8,  W8A16) | 0.89       |
        #  (W16A8,  W16A16)| 0.93       |
        #  (W8A16,  W16A16)| 0.94       |
        #  (W16A16, W16A16)| 0.95       |

        # Assert all layers are in W16A16 (baseline)
        for module in algo._sim.model.modules():
            if isinstance(module, QcQuantizeWrapper):
                for quantizer in module.param_quantizers.values():
                    if quantizer.enabled:
                        assert quantizer.bitwidth == 16

                for quantizer in module.input_quantizers:
                    if quantizer.enabled:
                        assert quantizer.bitwidth == 16

                for quantizer in module.output_quantizers:
                    if quantizer.enabled:
                        assert quantizer.bitwidth == 16

    def _run_phase2(self, algo, allowed_accuracy_drop, search_algo):
        algo.baseline_candidate = W16A16
        algo.min_candidate = W16A8
        fp32_acc = 1.0

        accuracy_list = [
            (algo.quantizer_groups[0], W8A16, phase1_eval_score_lookup_table[(W8A16, "fp32")], 100),
            (algo.quantizer_groups[0], W16A8, phase1_eval_score_lookup_table[(W16A8, "fp32")], 90),
            (algo.quantizer_groups[1], W8A16, phase1_eval_score_lookup_table[("fp32", W8A16)], 80),
            (algo.quantizer_groups[1], W16A8, phase1_eval_score_lookup_table[("fp32", W16A8)], 70),
        ]

        return algo._create_pareto_front_list(allowed_accuracy_drop, accuracy_list, fp32_acc,
                                              algo.baseline_candidate, algo.min_candidate, search_algo, phase2_reverse = False)

    def _run_phase2_reverse(self, algo, allowed_accuracy_drop, search_algo):
        algo.baseline_candidate = W16A16
        algo.min_candidate = W16A8
        fp32_acc = 1.0

        accuracy_list = [
            (algo.quantizer_groups[0], W8A16, phase1_eval_score_lookup_table[(W8A16, "fp32")], 100),
            (algo.quantizer_groups[0], W16A8, phase1_eval_score_lookup_table[(W16A8, "fp32")], 90),
            (algo.quantizer_groups[1], W8A16, phase1_eval_score_lookup_table[("fp32", W8A16)], 80),
            (algo.quantizer_groups[1], W16A8, phase1_eval_score_lookup_table[("fp32", W16A8)], 70),
        ]

        return algo._create_pareto_front_list(allowed_accuracy_drop, accuracy_list, fp32_acc,
                                              algo.baseline_candidate, algo.min_candidate, search_algo, phase2_reverse = True)

    def test_save_and_load_pareto_list_drop_greater_than_starting(
            self, sim, dummy_input, candidates, forward_pass_callback, eval_callback_phase1, eval_callback_phase2, results_dir
    ):
        json_file_path = os.path.join(results_dir, 'pareto_list.json')
        pickle_file_path = os.path.join(results_dir, '.cache' , 'pareto_list.pkl')

        # Save pareto list for some values of quantizer groups
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)

        algo.run(0.065)

        assert os.path.isfile(json_file_path)
        assert os.path.isfile(pickle_file_path)
        with open(pickle_file_path, "rb") as f:
            assert pickle.load(f) == algo.pareto_list
        assert len(algo.pareto_list) == 4

        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, False, forward_pass_callback)
        algo.run(None, AMPSearchAlgo.BruteForce)

        assert os.path.isfile(json_file_path)
        assert os.path.isfile(pickle_file_path)
        with open(pickle_file_path, "rb") as f:
            assert pickle.load(f) == algo.pareto_list
        assert len(algo.pareto_list) == 6

    def test_save_and_load_pareto_list_drop_lesser_than_starting(
            self, sim, dummy_input, candidates, forward_pass_callback, eval_callback_phase1, eval_callback_phase2, results_dir
    ):
        json_file_path = os.path.join(results_dir, 'pareto_list.json')
        pickle_file_path = os.path.join(results_dir, '.cache' , 'pareto_list.pkl')

        # Save pareto list for some values of quantizer groups
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)
        algo.run(None, AMPSearchAlgo.BruteForce)

        assert os.path.isfile(json_file_path)
        assert os.path.isfile(pickle_file_path)
        with open(pickle_file_path, "rb") as f:
            assert pickle.load(f) == algo.pareto_list
        assert len(algo.pareto_list) == 6

        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, False, forward_pass_callback)
        algo.run(0.06)

        assert os.path.isfile(json_file_path)
        assert os.path.isfile(pickle_file_path)
        with open(pickle_file_path, "rb") as f:
            assert pickle.load(f) == algo.pareto_list
        # The length of pareto list should not change
        assert len(algo.pareto_list) == 6

    def test_sorting_logic_for_accuracy_list_with_same_acc_values(self):
        quantizer_groups = [QuantizerGroup(), QuantizerGroup(), QuantizerGroup()]

        index_of_quantizer_group = {quantizer_groups[0]: 0, quantizer_groups[1]: 1,
                                    quantizer_groups[2]: 2}
        accuracy_list = [(quantizer_groups[0], ((16, QuantizationDataType.float), (8, QuantizationDataType.float)), 0.9, 110),
                         (quantizer_groups[0], ((8, QuantizationDataType.int), (16, QuantizationDataType.int)), 0.89, 90),
                         (quantizer_groups[1], ((16, QuantizationDataType.int), (8, QuantizationDataType.int)), 0.9, 100),
                         (quantizer_groups[1], ((8, QuantizationDataType.int), (16, QuantizationDataType.int)), 0.89, 100),
                         (quantizer_groups[2], ((16, QuantizationDataType.int), (8, QuantizationDataType.int)), 0.9, 100),
                         (quantizer_groups[2], ((8, QuantizationDataType.int), (16, QuantizationDataType.int)), 0.88, 50),
                         (quantizer_groups[0], ((4, QuantizationDataType.int), (8, QuantizationDataType.int)), 0.9, 100)]

        sorted_accuracy_list = sort_accuracy_list(accuracy_list, index_of_quantizer_group)

        assert sorted_accuracy_list[0] == accuracy_list[0]
        assert sorted_accuracy_list[1] == accuracy_list[2]
        assert sorted_accuracy_list[2] == accuracy_list[4]
        assert sorted_accuracy_list[3] == accuracy_list[6]
        assert sorted_accuracy_list[4] == accuracy_list[3]
        assert sorted_accuracy_list[5] == accuracy_list[1]
        assert sorted_accuracy_list[6] == accuracy_list[5]

    def test_quantizer_group_hashing(self):
        quantizer_group = QuantizerGroup(("module_a",), ("module_b",), ("module_c",))
        quantizer_group_copy = copy.deepcopy(quantizer_group)
        assert hash(quantizer_group) == hash(quantizer_group_copy)

        dict_ = {}
        val = object()
        dict_[quantizer_group] = val
        assert quantizer_group_copy in dict_
        assert dict_[quantizer_group_copy] is val

    def test_supported_candidates_1(
            self, model, dummy_input, candidates, forward_pass_callback, eval_callback_phase1, eval_callback_phase2, results_dir
    ):
        """
        Pass in vanilla config file without any specialized supported_kernels and verify the generated candidates in
        quantizer_groups of GreedyMixedPrecisionAlgo object
        """

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

            sim = QuantizationSimModel(model,
                                       default_param_bw=DEFAULT_BITWIDTH,
                                       default_output_bw=DEFAULT_BITWIDTH,
                                       dummy_input=dummy_input,
                                       config_file=os.path.join(tempdir, 'quantsim_config.json'))

            sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

            # Create an accuracy list
            algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                            results_dir, True, forward_pass_callback, use_all_amp_candidates=False)
            algo.run(0.9)

            assert len(algo._supported_candidates_per_quantizer_group.keys()) == 4

            default_supported_kernels = [((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
                                          ((16, QuantizationDataType.float), (16, QuantizationDataType.float))]

            for quantizer, quantizer_candidates in algo._supported_candidates_per_quantizer_group.items():
                # verify to make sure the candidates returned is always part of amp_candidates and they are part of
                # "Defaults"
                for c in quantizer_candidates:
                    assert c in default_supported_kernels
                    assert c in candidates

    def test_supported_candidates_2(
            self, model, dummy_input, candidates, forward_pass_callback, eval_callback_phase1, eval_callback_phase2, results_dir
    ):
        """
        Pass in vanilla config file without any specialized supported_kernels and verify the generated candidates in
        quantizer_groups of GreedyMixedPrecisionAlgo object
        """

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
                            "is_quantized": "False"
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

        with tempfile.TemporaryDirectory() as tempdir:
            with open(os.path.join(tempdir, 'quantsim_config.json'), 'w') as f:
                json.dump(quantsim_config, f)

            sim = QuantizationSimModel(model,
                                       default_param_bw=DEFAULT_BITWIDTH,
                                       default_output_bw=DEFAULT_BITWIDTH,
                                       dummy_input=dummy_input,
                                       config_file=os.path.join(tempdir, 'quantsim_config.json'))

            sim.compute_encodings(forward_fn, forward_pass_callback_args=None)

            # Create an accuracy list
            algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                            results_dir, True, forward_pass_callback, use_all_amp_candidates=False)
            algo.run(0.9)

            assert len(algo._supported_candidates_per_quantizer_group.keys()) == 4

            # default_supported_kernels and conv_supported_kernels are the configurations added in the json file above.
            default_supported_kernels = [((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
                                         ((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
                                         ((8, QuantizationDataType.int), (16, QuantizationDataType.int))]

            conv_supported_kernels = [((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
                                      ((8, QuantizationDataType.int), (16, QuantizationDataType.int))]

            for quantizer_group, quantizer_candidates in algo._supported_candidates_per_quantizer_group.items():
                supported_kernel_ops = quantizer_group.supported_kernel_ops
                onnx_types = []
                for op in supported_kernel_ops:
                    onnx_types.append(
                        onnx_utils.map_torch_types_to_onnx.get(type(algo._module_name_dict[op]._module_to_wrap)))

                # verify to make sure the candidates returned is always part of amp_candidates and they are part of
                # "Defaults" or "Conv" appropriately
                for c in quantizer_candidates:
                    assert c in candidates
                    if ['Conv'] in onnx_types:
                        assert c in conv_supported_kernels
                    else:
                        assert c in default_supported_kernels

    def test_set_quantizer_groups_candidates_1(self, sim, dummy_input, candidates, forward_pass_callback,
                                             eval_callback_phase1, eval_callback_phase2, results_dir):
        # validate good case
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)
        qg = algo.quantizer_groups
        algo.set_quantizer_groups_candidates([(qg[0],
                                               ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
                                               )])
        algo.set_quantizer_groups_candidates([(qg[2], ((16, QuantizationDataType.int), (None, None))
                                               )])
        algo.set_quantizer_groups_candidates([(qg[0], ((16, QuantizationDataType.int),  (16, QuantizationDataType.int))),
                                              (qg[2], ((16, QuantizationDataType.int), (None, None)))])

    def test_set_quantizer_groups_candidates_2(self, sim, dummy_input, candidates, forward_pass_callback,
                                             eval_callback_phase1, eval_callback_phase2, results_dir):
        # validate failing case of activations and params
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)
        qg = algo.quantizer_groups
        with pytest.raises(AssertionError):
            algo.set_quantizer_groups_candidates([(qg[0],
                                                   ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
                                                   )])

    def test_set_quantizer_groups_candidates_3(self, sim, dummy_input, candidates, forward_pass_callback,
                                             eval_callback_phase1, eval_callback_phase2, results_dir):
        # validate failing case of activations only
        algo = GreedyMixedPrecisionAlgo(sim, dummy_input, candidates, eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)
        qg = algo.quantizer_groups
        with pytest.raises(AssertionError):
            algo.set_quantizer_groups_candidates([(qg[2],
                                                   ((16, QuantizationDataType.float), (None, None))
                                                   )])


def test_compute_sqnr():
    for noise in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        orig_tensor = torch.randn((10, 10))
        noisy_tensor = orig_tensor + noise
        sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
        expected_sqnr = orig_tensor.square().mean() / (noise ** 2 + 0.0001)
        assert np.isclose(sqnr, expected_sqnr)

    orig_tensor = torch.ones((10, 10))
    noisy_tensor = torch.zeros((10, 10))
    sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
    expected_sqnr = 1
    assert np.isclose(sqnr, expected_sqnr, rtol=1e-3)

    orig_tensor = torch.ones((10, 10)) * 2
    noisy_tensor = torch.zeros((10, 10))
    sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
    expected_sqnr = 1
    assert np.isclose(sqnr, expected_sqnr, rtol=1e-3)

    orig_tensor = torch.ones((10, 10)) * 2
    noisy_tensor = torch.ones((10, 10))
    sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
    expected_sqnr = 4
    assert np.isclose(sqnr, expected_sqnr, rtol=1e-3)


FUNCTIONS = {
    "identity": float,
    "log2": lambda x: math.log2(x+1),
    "tanh": lambda x: math.tanh(x/1000),
    "sigmoid": (lambda x: 1 / (1 + math.exp(-x/1000)))
}

@pytest.mark.parametrize("func_name", FUNCTIONS.keys())
@pytest.mark.parametrize("search_algo", [brute_force_search, binary_search, interpolation_search])
def test_search_algo(search_algo, func_name):
    ARRAY_LENGTH = 1000
    func = FUNCTIONS[func_name]

    # Create an array of descending floats
    values = [
        functools.partial(func, i) for i in reversed(range(ARRAY_LENGTH))
    ]

    for index in range(ARRAY_LENGTH):
        ith_value = values[index]()

        # Case 1. Target value exists in the array
        found = search_algo(values, target=ith_value)
        assert found == index

        # Case 2. Target value is slightly smaller than one of the element in the array
        found = search_algo(values, target=ith_value - 1e-5)
        assert found == index

        # Case 3. Target value is slightly bigger than one of the element in the array
        found = search_algo(values, target=ith_value + 1e-5)
        assert found == max(index-1, 0)

@pytest.mark.parametrize("func_name", FUNCTIONS.keys())
@pytest.mark.parametrize("search_algo", [brute_force_search, binary_search, interpolation_search])
def test_search_algo_reverse(search_algo, func_name):
    ARRAY_LENGTH = 10
    func = FUNCTIONS[func_name]

    # Create an array of increasing floats
    values = [
        functools.partial(func, i) for i in range(ARRAY_LENGTH)
    ]

    for index in range(ARRAY_LENGTH):
        ith_value = values[index]()

        # Case 1. Target value exists in the array
        found = search_algo(values, target=ith_value, phase2_reverse=True)
        assert found == index

        # Case 2. Target value is slightly smaller than one of the element in the array
        found = search_algo(values, target=ith_value - 1e-5, phase2_reverse=True)
        assert found == index

        # Case 3. Target value is slightly bigger than one of the element in the array
        found = search_algo(values, target=ith_value + 1e-5, phase2_reverse=True)
        assert found == min(index+1, len(values)-1)
