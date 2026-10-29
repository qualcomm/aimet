# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import tempfile
import pytest
import itertools
import json
import unittest.mock
import ml_dtypes
import numpy as np
import onnxruntime as ort
import torch
import os
import shutil

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.amp.mixed_precision_algo import (
    GreedyMixedPrecisionAlgo,
    _compute_sqnr,
    EvalCallbackFactory,
)
from aimet_onnx.defs import DataLoader
from aimet_onnx.utils import OrtInferenceSession, make_dummy_input, make_psnr_eval_fn
from aimet_onnx.mixed_precision import (
    choose_mixed_precision,
    analyze_mixed_precision_sensitivity,
    apply_amp,
)

from aimet_onnx.common.defs import (
    QuantizationDataType,
    CallbackFunc,
    int8,
    int16,
    float16,
)
from aimet_onnx.common.amp.mixed_precision_algo import (
    interpolation_search,
    brute_force_search,
    binary_search,
)
from aimet_onnx.common.amp.utils import calculate_starting_bit_ops, AMPSearchAlgo

from .models.test_models import single_residual_model
from .models import models_for_tests

INPUT_SHAPE = (1, 3, 32, 32)


def forward_fn(session, _):
    np.random.seed(0)
    test_data = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    session.run(None, {"input": test_data})


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
    (W8A8, W8A8): 0.8,
    (W8A16, W8A8): 0.81,
    (W16A8, W8A8): 0.82,
    (W16A16, W8A8): 0.83,
    (W8A8, W16A8): 0.84,
    (W16A8, W16A8): 0.85,
    (W8A16, W16A8): 0.86,
    (W16A16, W16A8): 0.87,
    (W8A8, W8A16): 0.88,
    (W16A8, W8A16): 0.89,
    (W8A16, W8A16): 0.90,
    (W16A16, W8A16): 0.91,
    (W8A8, W16A16): 0.92,
    (W16A8, W16A16): 0.93,
    (W8A16, W16A16): 0.94,
    (W16A16, W16A16): 0.95,
    ("fp32", "fp32"): 1.0,
}


def eval_func(model, args):
    eval_score_lookup_table, sim = args
    # quantizer group 1
    input_quantizer = sim.qc_quantize_op_dict["input"]

    conv0_param_quantizer = sim.qc_quantize_op_dict[
        sim.connected_graph.ordered_ops[0].inputs[1].name
    ]
    if input_quantizer.enabled and conv0_param_quantizer.enabled:
        quantizer_1 = (
            (input_quantizer.bitwidth, QuantizationDataType.int),
            (conv0_param_quantizer.bitwidth, QuantizationDataType.int),
        )
    else:
        quantizer_1 = "fp32"

    # quantizer group 2
    fc_weight_quantizer = sim.qc_quantize_op_dict["fc.weight"]
    fc_output_quantizer = sim.qc_quantize_op_dict["/avgpool/AveragePool_output_0"]
    if fc_weight_quantizer.enabled and fc_output_quantizer.enabled:
        quantizer_2 = (
            (fc_output_quantizer.bitwidth, QuantizationDataType.int),
            (fc_weight_quantizer.bitwidth, QuantizationDataType.int),
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
            "ops": {"is_output_quantized": "True"},
            "params": {"is_quantized": "True"},
            "supported_kernels": [
                {
                    "activation": {"bitwidth": 16, "dtype": "int"},
                    "param": {"bitwidth": 16, "dtype": "int"},
                },
                {
                    "activation": {"bitwidth": 16, "dtype": "float"},
                    "param": {"bitwidth": 16, "dtype": "float"},
                },
            ],
        },
        "params": {"bias": {"is_quantized": "False"}},
        "op_type": {},
        "supergroups": [],
        "model_input": {"is_input_quantized": "True"},
        "model_output": {},
    }

    with tempfile.TemporaryDirectory() as tempdir:
        with open(os.path.join(tempdir, "quantsim_config.json"), "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(
            model, config_file=os.path.join(tempdir, "quantsim_config.json")
        )
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
        input_quantizers, output_quantizers, param_quantizers = sim.get_op_quantizers(
            cg_op
        )
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
    def test_phase1(
        self, sim, candidates, forward_pass_callback, eval_callback_phase1, results_dir
    ):
        algo = GreedyMixedPrecisionAlgo(
            sim,
            candidates,
            eval_callback_phase1,
            unittest.mock.MagicMock(),
            results_dir,
            True,
            forward_pass_callback,
        )
        algo.set_baseline()

        input_group = [
            qg for qg in algo.quantizer_groups if "input" in qg.activation_quantizers
        ][0]
        candidate = input_group.get_candidate(algo._module_name_dict)
        # Check if quantizer group is set to maximum bitwidth
        assert algo.baseline_candidate == candidate

        active_quantizers = {
            quantizer_group: quantizer_group.get_active_quantizers(
                algo._module_name_dict
            )
            for quantizer_group in algo.quantizer_groups
        }

        call_count = 0

        def assert_only_one_quantizer_group_enabled(*args, **kwargs):
            nonlocal call_count

            found_quantizer_groups = []
            for quantizer_group in algo.quantizer_groups:
                if quantizer_group.get_active_quantizers(algo._module_name_dict):
                    found_quantizer_groups.append(
                        quantizer_group.get_active_quantizers(algo._module_name_dict)
                    )

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
                    avgpool_output_name = (
                        algo._sim.connected_graph.get_all_ops()["AveragePool_12"]
                        .get_module()
                        .output[0]
                    )
                    avgpool_quantizer = algo._sim.qc_quantize_op_dict[
                        avgpool_output_name
                    ]
                    fc_weight_quantizer = algo._sim.qc_quantize_op_dict["fc.weight"]
                    assert [fc_weight_quantizer] in found_quantizer_groups
                    assert [
                        avgpool_quantizer,
                        fc_weight_quantizer,
                    ] in found_quantizer_groups

            call_count += 1

        with unittest.mock.patch(
            "aimet_onnx.amp.mixed_precision_algo.EvalCallbackFactory.sqnr",
            side_effect=assert_only_one_quantizer_group_enabled,
        ):
            accuracy_list = algo._create_and_save_accuracy_list(algo.baseline_candidate)

        # All the active quantizers should be still active
        for quantizer_group in algo.quantizer_groups:
            assert active_quantizers[
                quantizer_group
            ] == quantizer_group.get_active_quantizers(algo._module_name_dict)

        assert len(accuracy_list) == 22
        # Check if accuracy list is in descending order
        assert accuracy_list[0][2] >= accuracy_list[1][2]
        assert accuracy_list[1][2] >= accuracy_list[2][2]
        assert accuracy_list[2][2] >= accuracy_list[3][2]
        assert accuracy_list[3][2] >= accuracy_list[4][2]
        assert accuracy_list[4][2] >= accuracy_list[5][2]

    def test_phase2_brute_force(
        self,
        sim,
        candidates,
        forward_pass_callback,
        eval_callback_phase1,
        eval_callback_phase2,
        results_dir,
    ):
        allowed_accuracy_drop = 0.12
        algo = GreedyMixedPrecisionAlgo(
            sim,
            candidates,
            eval_callback_phase1,
            eval_callback_phase2,
            results_dir,
            True,
            forward_pass_callback,
        )
        algo.set_baseline()
        pareto_front_list = self._run_phase2(
            algo, allowed_accuracy_drop, brute_force_search
        )

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == 4

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(
            sim.model, [phase2_eval_score_lookup_table, sim]
        )
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

        # Test 3. Check bitops
        starting_bit_ops = calculate_starting_bit_ops(
            algo._mac_dict,
            ((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
        )
        running_bit_ops = (
            starting_bit_ops
            - algo._mac_dict["/conv1/Conv"] * 16 * 16
            + algo._mac_dict["/conv1/Conv"] * 8 * 16
        )
        relative_bit_ops = running_bit_ops / starting_bit_ops

        assert relative_bit_ops == pareto_front_list[0][0]
        assert relative_bit_ops == pareto_front_list[1][0]

        running_bit_ops = (
            running_bit_ops
            - algo._mac_dict["/fc/Gemm"] * 16 * 16
            + algo._mac_dict["/fc/Gemm"] * 8 * 16
        )
        relative_bit_ops = running_bit_ops / starting_bit_ops
        assert relative_bit_ops == pareto_front_list[2][0]

    @pytest.mark.parametrize(
        "allowed_accuracy_drop, len_of_pareto_list", [(0.13, 3), (0.1, 4)]
    )
    def test_phase2_interpolation(
        self,
        sim,
        candidates,
        forward_pass_callback,
        eval_callback_phase1,
        eval_callback_phase2,
        results_dir,
        allowed_accuracy_drop,
        len_of_pareto_list,
    ):
        algo = GreedyMixedPrecisionAlgo(
            sim,
            candidates,
            eval_callback_phase1,
            eval_callback_phase2,
            results_dir,
            True,
            forward_pass_callback,
        )

        pareto_front_list = self._run_phase2(
            algo, allowed_accuracy_drop, interpolation_search
        )

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == len_of_pareto_list

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(
            sim.model, [phase2_eval_score_lookup_table, sim]
        )
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

    @pytest.mark.parametrize(
        "allowed_accuracy_drop, len_of_pareto_list", [(0.055, 2), (0.1, 4)]
    )
    def test_phase2_binary(
        self,
        sim,
        candidates,
        forward_pass_callback,
        eval_callback_phase1,
        eval_callback_phase2,
        results_dir,
        allowed_accuracy_drop,
        len_of_pareto_list,
    ):
        algo = GreedyMixedPrecisionAlgo(
            sim,
            candidates,
            eval_callback_phase1,
            eval_callback_phase2,
            results_dir,
            True,
            forward_pass_callback,
        )

        pareto_front_list = self._run_phase2(algo, allowed_accuracy_drop, binary_search)

        # Test 1. Check number of data points visited
        assert len(pareto_front_list) == len_of_pareto_list

        # Test 2. Check final accuracy
        eval_score = eval_callback_phase2.func(
            sim.model, [phase2_eval_score_lookup_table, sim]
        )
        assert eval_score == algo._final_eval_score
        assert eval_score >= 1.0 - allowed_accuracy_drop

    def _run_phase2(self, algo, allowed_accuracy_drop, search_algo):
        algo.baseline_candidate = W16A16
        algo.min_candidate = W16A8
        fp32_acc = 1.0

        input_group = [
            qg for qg in algo.quantizer_groups if "input" in qg.activation_quantizers
        ][0]
        fc_group = [
            qg for qg in algo.quantizer_groups if "fc.weight" in qg.parameter_quantizers
        ][0]
        accuracy_list = [
            (input_group, W8A16, phase1_eval_score_lookup_table[(W8A16, "fp32")], 100),
            (input_group, W16A8, phase1_eval_score_lookup_table[(W16A8, "fp32")], 90),
            (fc_group, W8A16, phase1_eval_score_lookup_table[("fp32", W8A16)], 80),
            (fc_group, W16A8, phase1_eval_score_lookup_table[("fp32", W16A8)], 70),
        ]
        return algo._create_pareto_front_list(
            allowed_accuracy_drop,
            accuracy_list,
            fp32_acc,
            algo.baseline_candidate,
            algo.min_candidate,
            search_algo,
            phase2_reverse=False,
        )

    def test_supported_candidates_1(
        self,
        sim_supported_kernel,
        candidates,
        forward_pass_callback,
        eval_callback_phase1,
        eval_callback_phase2,
        results_dir,
    ):
        """
        Pass in vanilla config file without any specialized supported_kernels and verify the generated candidates in
        quantizer_groups of GreedyMixedPrecisionAlgo object
        """
        # Create an accuracy list
        algo = GreedyMixedPrecisionAlgo(
            sim_supported_kernel,
            candidates,
            eval_callback_phase1,
            eval_callback_phase2,
            results_dir,
            True,
            forward_pass_callback,
            use_all_amp_candidates=False,
        )

        assert len(algo._supported_candidates_per_quantizer_group.keys()) == 14

        default_supported_kernels = [
            ((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
            ((8, QuantizationDataType.float), (16, QuantizationDataType.float)),
        ]

        for (
            quantizer,
            quantizer_candidates,
        ) in algo._supported_candidates_per_quantizer_group.items():
            # verify to make sure the candidates returned is always part of amp_candidates and they are part of
            # "Defaults"
            for c in quantizer_candidates:
                assert c in default_supported_kernels
                assert c in candidates

    def test_supported_candidates_2(
        self,
        candidates,
        forward_pass_callback,
        eval_callback_phase1,
        eval_callback_phase2,
        results_dir,
    ):
        """
        Pass in vanilla config file without any specialized supported_kernels and verify the generated candidates in
        quantizer_groups of GreedyMixedPrecisionAlgo object
        """
        model = single_residual_model()
        quantsim_config = {
            "defaults": {
                "ops": {"is_output_quantized": "True"},
                "params": {"is_quantized": "True"},
                "supported_kernels": [
                    {
                        "activation": {"bitwidth": 16, "dtype": "int"},
                        "param": {"bitwidth": 16, "dtype": "int"},
                    },
                    {
                        "activation": {"bitwidth": 16, "dtype": "int"},
                        "param": {"bitwidth": 8, "dtype": "int"},
                    },
                    {
                        "activation": {"bitwidth": 8, "dtype": "int"},
                        "param": {"bitwidth": 16, "dtype": "int"},
                    },
                ],
            },
            "params": {"bias": {"is_quantized": "False"}},
            "op_type": {
                "Conv": {
                    "supported_kernels": [
                        {
                            "activation": {"bitwidth": 16, "dtype": "float"},
                            "param": {"bitwidth": 16, "dtype": "float"},
                        },
                        {
                            "activation": {"bitwidth": 8, "dtype": "int"},
                            "param": {"bitwidth": 16, "dtype": "int"},
                        },
                    ],
                    "is_output_quantized": "True",
                    "params": {
                        "weight": {"is_quantized": "True"},
                        "bias": {"is_quantized": "False"},
                    },
                }
            },
            "supergroups": [],
            "model_input": {"is_input_quantized": "True"},
            "model_output": {},
        }

        with open(os.path.join(results_dir, "quantsim_config.json"), "w") as f:
            json.dump(quantsim_config, f)

        sim = QuantizationSimModel(
            model, config_file=os.path.join(results_dir, "quantsim_config.json")
        )

        # Create an accuracy list
        algo = GreedyMixedPrecisionAlgo(
            sim,
            candidates,
            eval_callback_phase1,
            eval_callback_phase2,
            results_dir,
            True,
            forward_pass_callback,
            use_all_amp_candidates=False,
        )

        assert len(algo._supported_candidates_per_quantizer_group.keys()) == 14

        # default_supported_kernels and conv_supported_kernels are the configurations added in the json file above.
        default_supported_kernels = [
            ((16, QuantizationDataType.int), (16, QuantizationDataType.int)),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((8, QuantizationDataType.int), (16, QuantizationDataType.int)),
        ]

        conv_supported_kernels = [
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
            ((8, QuantizationDataType.int), (16, QuantizationDataType.int)),
        ]
        quantizer_to_op_type = _get_quantizer_name_op_type(sim)
        quantizer_to_op_type["output"] = ["Gemm"]

        for (
            quantizer,
            quantizer_candidates,
        ) in algo._supported_candidates_per_quantizer_group.items():
            quantizers = sorted(
                set(
                    itertools.chain(
                        quantizer.activation_quantizers, quantizer.parameter_quantizers
                    )
                )
            )
            onnx_types = []

            for q in quantizers:
                onnx_types.append(quantizer_to_op_type[q])

            # verify to make sure the candidates returned is always part of amp_candidates and they are part of
            # "Defaults" or "Conv" appropriately
            for c in quantizer_candidates:
                if ["Conv"] in onnx_types:
                    assert c in conv_supported_kernels
                else:
                    assert c in default_supported_kernels

    def test_respect_frozen_encodings(
        self, sim, forward_pass_callback, eval_callback_phase1, results_dir
    ):
        quantizer = [
            q
            for name, q in sim.qc_quantize_op_dict.items()
            if name in sim.param_names and q.enabled
        ][0]
        quantizer.set_bitwidth(4)
        sim.compute_encodings(forward_pass_callback.func, forward_pass_callback.args)
        quantizer.freeze_encodings()
        candidates = [W8A8, W8A16]
        algo = GreedyMixedPrecisionAlgo(
            sim,
            candidates,
            eval_callback_phase1,
            unittest.mock.MagicMock(),
            results_dir,
            True,
            forward_pass_callback,
        )
        algo.run(0.01, AMPSearchAlgo.Binary)
        assert quantizer.bitwidth == 4

    @pytest.mark.parametrize(
        "model",
        (
            single_residual_model().model,
            models_for_tests.dynamic_matmul_model(10),
            models_for_tests.matmul_with_constant_first_input(),
            models_for_tests.weight_matmul_model(),
            models_for_tests.dynamic_conv_model(),
            models_for_tests.mobilenetv2().model,
            models_for_tests.depthwise_transposed_conv_model().model,
            models_for_tests.model_with_split_matmul(),
            models_for_tests.hierarchical_model().model,
        ),
    )
    def test_choose_mixed_precision(self, model, tmpdir):
        np.random.seed(0)

        sim = QuantizationSimModel(model, config_file="htp_v73")
        enabled_quantizers = {q for q in sim.qc_quantize_op_dict.values() if q.enabled}
        total_bits = 16 * len(enabled_quantizers)

        forward_callback = CallbackFunc(
            lambda sess, _: sess.run(None, make_dummy_input(model)), None
        )

        def phase_2_callback(sess, _):
            bits = sum(q.bitwidth if q.enabled else 16 for q in enabled_quantizers)
            return bits / total_bits

        # Define dummy eval callbacks
        eval_callback_phase1 = CallbackFunc(lambda sess, _: np.random.rand())
        eval_callback_phase2 = CallbackFunc(phase_2_callback, None)

        candidates = [
            ((16, QuantizationDataType.float), (16, QuantizationDataType.float)),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
        ]

        # Apply mixed precision
        choose_mixed_precision(
            sim,
            candidates,
            eval_callback_phase1,
            eval_callback_phase2,
            0.4,
            tmpdir,
            True,
            forward_callback,
        )

        # Assert that no param quantizers are in int16 (not a valid candidate)
        for name in sim.param_names:
            quantizer = sim.qc_quantize_op_dict[name]
            assert not (
                quantizer.bitwidth == 16
                and quantizer.data_type == QuantizationDataType.int
            )

        # Assert that the final result meets the accuracy metric
        assert sum(q.bitwidth for q in enabled_quantizers) <= total_bits
        assert sum(q.bitwidth for q in enabled_quantizers) >= total_bits * 0.6

        # Assert that the final mixed-precision profile obeys config file's exception rules
        for op in sim.connected_graph.ordered_ops:
            if not op.type in ("MatMul", "Gemm"):
                continue

            q1, q2 = (
                sim._get_closest_enabled_quantizer(op.inputs[0]),
                sim._get_closest_enabled_quantizer(op.inputs[1]),
            )
            if not q1 or not q2:
                continue

            # Config requires symmetric second input for 16-bit matmul
            if q2.bitwidth == 16 and not q2.data_type == QuantizationDataType.float:
                assert q2.use_symmetric_encodings

            # 8 x 16 MatMul is not a valid combination
            if q1.bitwidth == 8:
                assert q2.bitwidth == 8

    def test_mixed_precision_with_tied_quantizers(self, tmpdir):
        model = single_residual_model().model
        sim = QuantizationSimModel(model, param_type="int8", activation_type="int16")
        maxpool_op = next(
            iter(op for op in sim.connected_graph.ordered_ops if op.type == "MaxPool")
        )
        maxpool_output_quantizer = sim.qc_quantize_op_dict[maxpool_op.outputs[0].name]
        tied_quantizers = {}
        # Tie all output quantizers until first maxpool
        for op in sim.connected_graph.ordered_ops:
            if op == maxpool_op:
                break
            if op.outputs[0].name not in sim.qc_quantize_op_dict:
                continue
            tied_quantizers[op.outputs[0].name] = maxpool_output_quantizer

        maxpool_output_quantizer.enabled = True
        sim.set_quantizers(tied_quantizers)

        def eval_callback(*args):
            return (
                0.0
                if maxpool_output_quantizer.enabled
                and maxpool_output_quantizer.bitwidth == 8
                else 1.0
            )

        def calib_callback(session, _):
            session.run(None, make_dummy_input(model))

        """
        Given: Model with a sensitive shared quantizer
        When: Run AMP
        Then: All QuantizerGroups containing that quantizer should be flagged as sensitive
        """

        candidates = [
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
        ]

        choose_mixed_precision(
            sim,
            candidates,
            CallbackFunc(eval_callback, None),
            CallbackFunc(eval_callback, None),
            0.1,
            tmpdir,
            False,
            CallbackFunc(calib_callback, None),
        )

        other_quantizers = [
            q
            for q in sim.qc_quantize_op_dict.values()
            if q.enabled and q is not maxpool_output_quantizer
        ]

        # Shared quantizer must be set to int16 based on phase 2 callback
        assert maxpool_output_quantizer.enabled
        assert maxpool_output_quantizer.bitwidth == 16
        # Maxpool quantizer should be most sensitive based on phase 1 callback
        # All other quantizer should be set to int8
        for quantizer in other_quantizers:
            assert quantizer.enabled
            assert quantizer.bitwidth == 8

    @pytest.mark.parametrize(
        "model",
        (
            single_residual_model().model,
            models_for_tests.dynamic_matmul_model(10),
            # models_for_tests.matmul_with_constant_first_input(),
            # models_for_tests.weight_matmul_model(),
            models_for_tests.dynamic_conv_model(),
            models_for_tests.mobilenetv2().model,
            # models_for_tests.depthwise_transposed_conv_model().model,
            models_for_tests.model_with_split_matmul(),
            models_for_tests.hierarchical_model().model,
        ),
    )
    def test_two_phase_mixed_precision(self, model):
        np.random.seed(0)

        sim = QuantizationSimModel(model, config_file="htp_v73")
        enabled_quantizers = {
            name: q for name, q in sim.qc_quantize_op_dict.items() if q.enabled
        }
        total_bits = 16 * len(enabled_quantizers)
        # Define dummy eval callbacks
        eval_fn = lambda sess: np.random.rand()

        candidates = [
            (float16, float16),
            (int8, int16),
            (int8, int8),
        ]

        calibration_input = [make_dummy_input(model)]

        mp_results = analyze_mixed_precision_sensitivity(
            sim, candidates, eval_fn, calibration_input
        )

        mp_results["int8,int8"][0].keys() == enabled_quantizers.keys()
        mp_results["int8,int8"][1].keys() == enabled_quantizers.keys()

        def phase_2_callback(sess):
            bits = sum(
                q.bitwidth if q.enabled else 16 for q in enabled_quantizers.values()
            )
            return bits / total_bits >= 0.6

        apply_amp(sim, mp_results, phase_2_callback)

        assert sum(q.bitwidth for q in enabled_quantizers.values()) <= total_bits
        assert sum(q.bitwidth for q in enabled_quantizers.values()) >= total_bits * 0.6
        assert phase_2_callback(sim.session)

        # Assert that the final mixed-precision profile obeys config file's exception rules
        for op in sim.connected_graph.ordered_ops:
            if not op.type in ("MatMul", "Gemm"):
                continue

            q1, q2 = (
                sim._get_closest_enabled_quantizer(op.inputs[0]),
                sim._get_closest_enabled_quantizer(op.inputs[1]),
            )
            if not q1 or not q2:
                continue

            # Config requires symmetric second input for 16-bit matmul
            if q2.bitwidth == 16 and not q2.data_type == QuantizationDataType.float:
                assert q2.use_symmetric_encodings

            # 8 x 16 MatMul is not a valid combination
            if q1.bitwidth == 8:
                assert q2.bitwidth == 8


class TestAMPv2:
    def test_compute_sqnr(self):
        """Verify _compute_sqnr() method"""
        for noise in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            orig_tensor = np.random.randn(10, 10)
            noisy_tensor = orig_tensor + noise
            sqnr = _compute_sqnr(orig_tensor, noisy_tensor)
            expected_sqnr = np.power(orig_tensor, 2).mean() / (noise**2 + 0.0001)
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

    def test_compute_sqnr_bfloat16(self):
        """
        Given: bfloat16 tensors
        When: Compute their SQNR
        Then: It matches the SQNR of the same values in float32. Reducing many squares in
              bfloat16 would lose most of the sum
        """
        rng = np.random.default_rng(0)
        orig = rng.standard_normal(100_000).astype(ml_dtypes.bfloat16)
        noisy = (orig.astype(np.float32) + 0.01).astype(ml_dtypes.bfloat16)

        sqnr = _compute_sqnr(orig, noisy)

        expected = _compute_sqnr(orig.astype(np.float32), noisy.astype(np.float32))
        assert np.isclose(sqnr, expected, rtol=1e-4)

    @pytest.mark.cuda
    @pytest.mark.parametrize("io_dtype", [torch.float32, torch.bfloat16])
    def test_amp_bfloat16_model(self, io_dtype):
        """
        Given: Model computing in bfloat16, with float32 or bfloat16 graph inputs/outputs
        When: Analyze its mixed precision sensitivity and apply AMP with a PSNR target
              halfway between those of int8 and int16 activations
        Then: AMP meets the target with a mix of int8 and int16 activations, and the
              built-in SQNR eval callback runs on the bfloat16 graph
        """
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CUDA EP not available")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        model = models_for_tests.simple_mlp_model(torch.bfloat16, io_dtype=io_dtype)
        rng = np.random.default_rng(0)
        np_dtype = np.float32 if io_dtype == torch.float32 else ml_dtypes.bfloat16
        inputs = [
            {"input": rng.standard_normal((8, 16)).astype(np_dtype)} for _ in range(2)
        ]
        psnr_eval_fn = make_psnr_eval_fn(OrtInferenceSession(model, providers), inputs)

        def sim_psnr(activation_type):
            sim = QuantizationSimModel(
                model,
                param_type=int8,
                activation_type=activation_type,
                providers=providers,
            )
            sim.compute_encodings(inputs)
            return sim, psnr_eval_fn(sim.session)

        _, int16_psnr = sim_psnr(int16)
        sim, int8_psnr = sim_psnr(int8)
        assert int16_psnr > int8_psnr + 2
        target = (int8_psnr + int16_psnr) / 2

        mp_results = analyze_mixed_precision_sensitivity(
            sim, [(int8, int8), (int8, int16)], psnr_eval_fn, inputs
        )
        apply_amp(sim, mp_results, lambda session: psnr_eval_fn(session) >= target)

        assert psnr_eval_fn(sim.session) >= target
        activation_bitwidths = {
            sim.qc_quantize_op_dict[name].bitwidth
            for name in sim.activation_names
            if sim.qc_quantize_op_dict[name].enabled
        }
        assert activation_bitwidths == {8, 16}

        class _Dataset(DataLoader):
            def __init__(self):
                super().__init__(inputs[0]["input"], 8, 1)

            def __iter__(self):
                yield inputs[0]["input"]

        sqnr = EvalCallbackFactory(_Dataset()).sqnr(sim).func(sim.session, None)
        assert np.isfinite(sqnr) and sqnr > 0
