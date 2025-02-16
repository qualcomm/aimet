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

import os
import pickle
import shutil
import tempfile
import unittest
import unittest.mock
import numpy as np
import tensorflow as tf
import pytest

from aimet_tensorflow.keras.quantsim import QuantizationSimModel
from aimet_tensorflow.keras.amp.mixed_precision_algo import GreedyMixedPrecisionAlgo
from aimet_tensorflow.keras.amp.quantizer_groups import QuantizerGroup
from aimet_tensorflow.keras.amp.mixed_precision_algo import _compute_sqnr, EvalCallbackFactory, disable_all_quantizers
from aimet_tensorflow.keras.mixed_precision import choose_mixed_precision, choose_fast_mixed_precision
from aimet_tensorflow.keras.quant_sim.qc_quantize_wrapper import QcQuantizeWrapper

from aimet_common.defs import CallbackFunc
from aimet_common.amp.utils import calculate_starting_bit_ops, QuantizationDataType, AMPSearchAlgo
import aimet_common.amp.mixed_precision_algo
from aimet_common.amp.mixed_precision_algo import brute_force_search, interpolation_search


W8A8 = ((8, QuantizationDataType.int), (8, QuantizationDataType.int))
W8A16 = ((16, QuantizationDataType.int), (8, QuantizationDataType.int))
W16A8 = ((8, QuantizationDataType.int), (16, QuantizationDataType.int))
W16A16 = ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
FP16 = ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
FP32 = "FP32"


# Lookup table that maps (quantizer_group_1, quantizer_group_2) -> eval_score
phase1_eval_score_lookup_table = {
    (FP32, W8A8): 0.85,
    (W8A8, FP32): 0.9,
    (W16A8, FP32): 0.91,
    (W8A16, FP32): 0.92,
    (FP32, W16A8): 0.93,
    (FP32, W8A16): 0.94,
    (FP32, FP32): 1.0,
}

# Lookup table that maps (quantizer_group_1, quantizer_group_2) -> eval_score
phase2_eval_score_lookup_table = {
    (W8A8, W8A8): 0.8,
    (W8A16, W8A8): 0.81,
    (W16A8, W8A8): 0.82,
    (W8A8, W8A16): 0.83,
    (W8A8, W16A8): 0.84,
    (W16A16, W8A8): 0.85,
    (W8A16, W8A16): 0.86,
    (W8A16, W16A8): 0.87,
    (W16A8, W8A16): 0.88,
    (W16A8, W16A8): 0.89,
    (W8A8, W16A16): 0.9,
    (W16A8, W16A16): 0.91,
    (W8A16, W16A16): 0.92,
    (W16A16, W16A8): 0.93,
    (W16A16, W8A16): 0.94,
    (W16A16, W16A16): 0.95,
    (FP32, FP32): 1.0,
}


def eval_func(model: tf.keras.Model, eval_score_lookup_table):
    # p_0_quantizer = sim.quantizer_config('conv2d/Conv2D/ReadVariableOp_quantized')
    # o_0_quantizer = sim.quantizer_config('conv2d_input_quantized')

    # p_quantizer = sim.quantizer_config('conv2d_1/Conv2D/ReadVariableOp_quantized')
    # o_quantizer = sim.quantizer_config('conv2d/Relu_quantized')

    # quantizer group 1
    import aimet_common.libpymo as libpymo

    def find_quantizer(model, quantizer_name):
        layer = model.get_layer("qc_quantize_wrapper")
        layer = model.get_layer("qc_quantize_wrapper_1")
        for layer in model.layers:
            for quantizer in layer.input_quantizers + layer.output_quantizers + layer.param_quantizers:
                if quantizer.name == quantizer_name:
                    return quantizer
        raise RuntimeError("Not found")

    conv1_input_quantizer = find_quantizer(model, "conv2d_input_quantizer_0")
    conv1_param_quantizer = find_quantizer(model, "conv2d/kernel")

    if conv1_input_quantizer.is_enabled() and conv1_param_quantizer.is_enabled():
        quantizer_1 = (
            (conv1_input_quantizer.bitwidth, conv1_input_quantizer.data_type),
            (conv1_param_quantizer.bitwidth, conv1_param_quantizer.data_type),
        )
    else:
        quantizer_1 = FP32

    conv2_input_quantizer = find_quantizer(model, "conv2d_input_quantizer_0")
    conv2_param_quantizer = find_quantizer(model, "conv2d_1/kernel")

    if conv2_input_quantizer.is_enabled() and conv2_param_quantizer.is_enabled():
        quantizer_2 = (
            (conv2_input_quantizer.bitwidth, conv2_input_quantizer.data_type),
            (conv2_param_quantizer.bitwidth, conv2_param_quantizer.data_type),
        )
    else:
        quantizer_2 = FP32

    key = (quantizer_1, quantizer_2)
    return eval_score_lookup_table[key]


@pytest.fixture
def eval_callback_phase1():
    return CallbackFunc(eval_func, phase1_eval_score_lookup_table)


@pytest.fixture
def eval_callback_phase2():
    return CallbackFunc(eval_func, phase2_eval_score_lookup_table)

@pytest.fixture
def data_loader_wrapper():
    data_loader = [
        tf.random.uniform((128, 28, 28, 3)) for _ in range(1)
    ]
    data_loader_wrapper=lambda: data_loader
    return data_loader_wrapper


@pytest.fixture
def forward_pass_callback():
    def dummy_forward_pass(model, _):
        dummy_input = np.random.randn(20, 28, 28, 3)
        model.predict(dummy_input)

    return CallbackFunc(dummy_forward_pass, None)


@pytest.fixture
def sim():
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential((
        tf.keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 3), activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=3),
        tf.keras.layers.DepthwiseConv2D(kernel_size=3)
    ))
    sim = QuantizationSimModel(model, default_output_bw=4, default_param_bw=6)
    yield sim


@pytest.fixture
def quantizer_config_dict(sim):
    quantizer = sim.get_quant_wrapper_for_layer_name('conv2d')
    quantizer0 = sim.get_quant_wrapper_for_layer_name('conv2d_1')

    return {
        'conv2d/kernel': quantizer,
        'conv2d_input_quantizer_0': quantizer,
        'conv2d_1/kernel': quantizer0,
        'conv2d_1_input_quantizer_0': quantizer0,
    }

@pytest.fixture
def quantizer_groups():
    return [
        QuantizerGroup(
            parameter_quantizers=('conv2d/kernel',),
            input_quantizers=('conv2d_input_quantizer_0',),
        ),
        QuantizerGroup(
            parameter_quantizers=('conv2d_1/kernel',),
            input_quantizers=('conv2d_1_input_quantizer_0',),
        ),
    ]


@pytest.fixture
def results_dir():
    with tempfile.TemporaryDirectory() as tempdir:
        os.makedirs(os.path.join(tempdir, ".cache"), exist_ok=True)
        yield tempdir
        shutil.rmtree(tempdir)


@pytest.fixture
def int_candidates():
    return [W16A16, W8A16]


class TestAutoMixedPrecision:
    def test_fp32_accuracy_and_algo_params(
            self, sim, eval_callback_phase1, eval_callback_phase2,
            forward_pass_callback, results_dir, int_candidates
    ):
        algo = GreedyMixedPrecisionAlgo(sim, int_candidates,
                                        eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)
        algo.set_baseline()

        assert algo.fp32_accuracy == phase2_eval_score_lookup_table[(FP32, FP32)]
        assert algo.baseline_candidate == W16A16

    def test_eval_func_edge_cases(
            self, sim, forward_pass_callback, results_dir, int_candidates
    ):
        # Edge case: None-float eval score
        eval_callback = CallbackFunc(lambda *_: (0.1, 0.2), None)
        with pytest.raises(RuntimeError):
            algo = GreedyMixedPrecisionAlgo(sim, int_candidates,
                                            eval_callback, eval_callback, results_dir, True, forward_pass_callback)
            algo.set_baseline()

    def test_early_exit_best_quantized_accuracy_inadequate(self, sim, eval_callback_phase1, eval_callback_phase2,
                                                           forward_pass_callback, results_dir, int_candidates):
        json_file_path = os.path.join(results_dir, 'pareto_list.json')
        pickle_file_path = os.path.join(results_dir, '.cache', 'pareto_list.pkl')
        #Remove the files before invoking AMP
        if(os.path.isfile(json_file_path)):
            try:
                os.remove(json_file_path)
            except:
                assert(False)

        if(os.path.isfile(pickle_file_path)):
            try:
                os.remove(pickle_file_path)
            except:
                assert(False)

        algo = GreedyMixedPrecisionAlgo(sim, int_candidates,
                                        eval_callback_phase1, eval_callback_phase2, results_dir, True,
                                        forward_pass_callback)
        algo.run(allowed_accuracy_drop=0.0001, search_algo=AMPSearchAlgo.BruteForce)

        baseline_accuracy, _= algo._get_best_candidate()
        assert algo._final_eval_score == baseline_accuracy
        assert (not os.path.isfile(json_file_path))
        assert (not os.path.isfile(pickle_file_path))

    def test_create_accuracy_list(
            self, sim, quantizer_groups, quantizer_config_dict, eval_callback_phase1,
            eval_callback_phase2, forward_pass_callback, results_dir, int_candidates
    ):
        fp32_accuracy = phase2_eval_score_lookup_table[(W16A16, W16A16)]
        baseline_candidate = W16A16

        algo = GreedyMixedPrecisionAlgo(sim, int_candidates,
                                        eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)
        algo.set_baseline(fp32_accuracy, baseline_candidate)

        # quantizer_groups = algo.quantizer_groups
        algo.quantizer_groups = quantizer_groups
        algo._module_name_dict = quantizer_config_dict
        algo._supported_candidates_per_quantizer_group = {}

        for quantizer_group in quantizer_groups:
            algo._supported_candidates_per_quantizer_group[quantizer_group] = int_candidates

        active_quantizers = {
            quantizer_group: quantizer_group.get_active_quantizers(algo._module_name_dict)
            for quantizer_group in quantizer_groups
        }

        call_count = 0
        def assert_only_one_quantizer_group_enabled(*args, **kwargs):
            nonlocal call_count

            all_active_quantizers = []
            for quantizer_group in quantizer_groups:
                all_active_quantizers +=\
                        quantizer_group.get_active_quantizers(algo._module_name_dict)

            if call_count < len(quantizer_groups) * (len(int_candidates) - 1):
                # During phase 1 loop, only one quantizer group can be activated at a time
                assert all_active_quantizers in active_quantizers.values()

            call_count += 1

        with unittest.mock.patch(
            'aimet_tensorflow.keras.amp.mixed_precision_algo.EvalCallbackFactory.sqnr',
            side_effect=assert_only_one_quantizer_group_enabled
        ):
            accuracy_list = algo._create_and_save_accuracy_list(algo.baseline_candidate)

        # All the active quantizers should be still active
        for quantizer_group in quantizer_groups:
            active_quantizers[quantizer_group] ==\
                    quantizer_group.get_active_quantizers(algo._module_name_dict)

        assert len(accuracy_list) == 2
        assert accuracy_list[0][2] >= accuracy_list[1][2]

        # Check save and load accuracy list
        accuracy_list_path = os.path.join(results_dir, '.cache', 'accuracy_list.pkl')
        with open(accuracy_list_path, 'rb') as f:
            loaded_accuracy_list = pickle.load(f)
        assert loaded_accuracy_list == accuracy_list

    @pytest.mark.parametrize("fp32_accuracy", [0.9, 0.001, -0.1]) # To test accuracy list of positive, negative, and mixture of both values.
    def test_pareto_list_base(
            self, sim, quantizer_groups, quantizer_config_dict, eval_callback_phase1, eval_callback_phase2,
            forward_pass_callback, results_dir, int_candidates, fp32_accuracy,
    ):
        baseline_candidate = W16A16

        algo = GreedyMixedPrecisionAlgo(sim, int_candidates,
                                        eval_callback_phase1, eval_callback_phase2,
                                        results_dir, True, forward_pass_callback)
        algo.set_baseline(fp32_accuracy, baseline_candidate)
        algo.min_candidate = W8A16
        algo.quantizer_groups = quantizer_groups
        algo._module_name_dict = quantizer_config_dict
        algo._supported_candidates_per_quantizer_group = {}

        for quantizer_group in quantizer_groups:
            algo._supported_candidates_per_quantizer_group[quantizer_group] = int_candidates

        accuracy_list = [
            (quantizer_groups[0], W8A16, fp32_accuracy, 584064 * (16 * 8 - 8 * 8)),
            (quantizer_groups[1], W8A16, fp32_accuracy - 0.005, 2230272 * (16 * 8 - 8 * 8)),
        ]
        with unittest.mock.patch('aimet_tensorflow.keras.quantsim.QuantizationSimModel.compute_encodings'):
            pareto_front_list = algo._create_pareto_front_list(0.2,
                                                               accuracy_list,
                                                               fp32_accuracy,
                                                               algo.baseline_candidate,
                                                               algo.min_candidate,
                                                               brute_force_search,
                                                               phase2_reverse = False)

        mac_dict = algo._mac_dict
        starting_bit_ops = calculate_starting_bit_ops(mac_dict, W16A16)
        running_bit_ops = starting_bit_ops - mac_dict['conv2d'] * 16 * 16 + \
                          mac_dict['conv2d'] * 16 * 8
        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert len(pareto_front_list) == 2
        assert relative_bit_ops == pareto_front_list[0][0]

        running_bit_ops = running_bit_ops - mac_dict['conv2d_1'] * 16 * 16 + \
                          mac_dict['conv2d_1'] * 16 * 8
        relative_bit_ops = running_bit_ops / starting_bit_ops
        assert relative_bit_ops == pareto_front_list[1][0]

    def test_pareto_list_drop_less_than_before(
            self, sim, quantizer_groups, quantizer_config_dict, eval_callback_phase1,
            eval_callback_phase2, forward_pass_callback, results_dir, int_candidates
    ):
        fp32_accuracy = phase2_eval_score_lookup_table[(W16A16, W16A16)]
        baseline_candidate = W16A16

        algo = GreedyMixedPrecisionAlgo(sim, int_candidates,
                                        eval_callback_phase1, eval_callback_phase2,
                                        results_dir, False, forward_pass_callback)
        algo.set_baseline(fp32_accuracy, baseline_candidate)
        algo.min_candidate = W8A16
        algo.quantizer_groups = quantizer_groups
        algo._module_name_dict = quantizer_config_dict

        accuracy_list = [
            (quantizer_groups[0], W8A16, 0.9, 584064 * (16 * 8 - 8 * 8)),
            (quantizer_groups[1], W8A16, 0.895, 2230272 * (16 * 8 - 8 * 8)),
        ]

        with unittest.mock.patch('aimet_tensorflow.keras.quantsim.QuantizationSimModel.compute_encodings'):
            pareto_front_list = algo._create_pareto_front_list(0.2,
                                                               accuracy_list,
                                                               fp32_accuracy,
                                                               algo.baseline_candidate,
                                                               algo.min_candidate,
                                                               brute_force_search,
                                                               phase2_reverse = False)
        assert len(pareto_front_list) == 2

        # change drop to lower value
        with unittest.mock.patch('aimet_tensorflow.keras.quantsim.QuantizationSimModel.compute_encodings'):
            pareto_front_list = algo._create_pareto_front_list(0.1,
                                                               accuracy_list,
                                                               fp32_accuracy,
                                                               algo.baseline_candidate,
                                                               algo.min_candidate,
                                                               brute_force_search,
                                                               phase2_reverse = False)
        assert len(pareto_front_list) == 2

    def test_pareto_list_drop_greater_than_before(
            self, sim, quantizer_groups, quantizer_config_dict, eval_callback_phase1,
            eval_callback_phase2, forward_pass_callback, results_dir, int_candidates
    ):
        fp32_accuracy = phase2_eval_score_lookup_table[(W16A16, W16A16)]
        baseline_candidate = W16A16

        algo = GreedyMixedPrecisionAlgo(sim, int_candidates,
                                        eval_callback_phase1, eval_callback_phase2,
                                        results_dir, False, forward_pass_callback)
        algo.set_baseline(fp32_accuracy, baseline_candidate)
        algo.min_candidate = W8A16
        algo.quantizer_groups = quantizer_groups
        algo._module_name_dict = quantizer_config_dict

        accuracy_list = [
            (quantizer_groups[0], W8A16, 0.9, 584064 * (16 * 8 - 8 * 8)),
            (quantizer_groups[1], W8A16, 0.895, 2230272 * (16 * 8 - 8 * 8)),
        ]

        with unittest.mock.patch('aimet_tensorflow.keras.quantsim.QuantizationSimModel.compute_encodings'):
            pareto_front_list = algo._create_pareto_front_list(0.002,
                                                               accuracy_list,
                                                               fp32_accuracy,
                                                               algo.baseline_candidate,
                                                               algo.min_candidate,
                                                               brute_force_search,
                                                               phase2_reverse = False)
        assert len(pareto_front_list) == 1

        # change drop to lower value
        with unittest.mock.patch('aimet_tensorflow.keras.quantsim.QuantizationSimModel.compute_encodings'):
            pareto_front_list = algo._create_pareto_front_list(0.1,
                                                               accuracy_list,
                                                               fp32_accuracy,
                                                               algo.baseline_candidate,
                                                               algo.min_candidate,
                                                               brute_force_search,
                                                               phase2_reverse = False)
        assert len(pareto_front_list) == 2

    def test_compare_sqnr_callback(self, sim, data_loader_wrapper):
        def forward_pass_callback(model, _):
            for x in data_loader_wrapper():
                model(x)
                return

        sim.compute_encodings(forward_pass_callback, None)

        # Get the callback function in which reference model is provided
        sqnr_eval_callback = EvalCallbackFactory(data_loader_wrapper).sqnr(sim._model_without_wrappers)
        sqnr1 = sqnr_eval_callback.func(sim.model, sqnr_eval_callback.args)

        # Get the callback function in which reference model is not provided
        sqnr_eval_callback = EvalCallbackFactory(data_loader_wrapper).sqnr()
        sqnr2 = sqnr_eval_callback.func(sim.model, sqnr_eval_callback.args)

        assert np.isclose(sqnr1, sqnr2)

    def test_compute_sqnr_validation(self):
        orig_numpy_array = np.random.random((32, 500))
        noisy_numpy_array = np.random.random((32, 500))
        noisy_tf_tensor = tf.random.uniform((32, 500))

        with pytest.raises(ValueError):
            _ = _compute_sqnr(orig_numpy_array, noisy_tf_tensor)

        with pytest.raises(ValueError):
            _ = _compute_sqnr(orig_numpy_array.tolist(), noisy_numpy_array.tolist())

        _ = _compute_sqnr(orig_numpy_array, noisy_numpy_array)


    def test_disable_quantizers(self, sim):
        with disable_all_quantizers(sim.model):

            # all the quantizers should be disabled
            for layer in sim.model.layers:
                if not isinstance(layer, QcQuantizeWrapper):
                    continue

                for quantizer in layer.param_quantizers + \
                                 layer.input_quantizers + \
                                 layer.output_quantizers:
                    assert not quantizer.is_enabled()


    def test_choose_mixed_precision(self, sim, eval_callback_phase1, eval_callback_phase2,
                                    forward_pass_callback, results_dir, int_candidates):
        choose_mixed_precision(sim, int_candidates, eval_callback_phase1, eval_callback_phase2, 0.009, results_dir, False, forward_pass_callback)

    def test_choose_fast_mixed_precision(self, sim, data_loader_wrapper, eval_callback_phase2,
                                         forward_pass_callback, results_dir, int_candidates):

        pareto_list = choose_fast_mixed_precision(sim, int_candidates, data_loader_wrapper, eval_callback_phase2, 0.06, results_dir, False, forward_pass_callback)

        # Length of the pareto list should be 2
        assert len(pareto_list) == 2


    def test_compute_sqnr(self):
        for noise in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            orig_tensor = tf.random.uniform((10, 10))
            noisy_tensor = orig_tensor + noise
            sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
            expected_sqnr = tf.reduce_mean(orig_tensor ** 2) / (noise ** 2 + 0.0001)
            assert np.isclose(sqnr, expected_sqnr)

        orig_tensor = tf.ones((10, 10))
        noisy_tensor = tf.zeros((10, 10))
        sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
        expected_sqnr = 1
        assert np.isclose(sqnr, expected_sqnr, rtol=1e-3)

        orig_tensor = tf.ones((10, 10)) * 2
        noisy_tensor = tf.zeros((10, 10))
        sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
        expected_sqnr = 1
        assert np.isclose(sqnr, expected_sqnr, rtol=1e-3)

        orig_tensor = tf.ones((10, 10)) * 2
        noisy_tensor = tf.ones((10, 10))
        sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
        expected_sqnr = 4
        assert np.isclose(sqnr, expected_sqnr, rtol=1e-3)
