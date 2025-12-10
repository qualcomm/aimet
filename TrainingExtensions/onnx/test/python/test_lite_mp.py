# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import tempfile
import onnxruntime

from aimet_onnx.common.defs import QuantizationDataType
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx import analyze_per_layer_sensitivity
from aimet_onnx import int8, int16, float16
from aimet_onnx.utils import make_dummy_input, make_psnr_eval_fn
from aimet_onnx.lite_mp import flip_layers_to_higher_precision
from .models import models_for_tests


class TestLiteMp:
    @pytest.mark.parametrize("percent_flip", [30, 50.2, 80])
    def test_flip_to_float(self, percent_flip):
        model = models_for_tests.single_residual_model().model
        fp_session = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CUDAExecutionProvider"]
        )

        sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
        q_ops = [op for op in sim.qc_quantize_op_dict.values() if op.enabled]
        int8_count = sum(1 for op in q_ops if op.bitwidth == 8)
        fp_count = sum(1 for op in q_ops if op.data_type == QuantizationDataType.float)
        assert fp_count == 0

        inputs = [make_dummy_input(model)]
        psnr_eval_fn = make_psnr_eval_fn(fp_session, inputs)
        layer_sensitivity_dict = analyze_per_layer_sensitivity(
            sim, eval_fn=psnr_eval_fn
        )
        flip_layers_to_higher_precision(
            sim, layer_sensitivity_dict, percent_flip, override_precision=float16
        )

        new_int8_count = sum(1 for op in q_ops if op.bitwidth == 8 and op.enabled)
        assert new_int8_count < int8_count
        assert new_int8_count <= (100 - percent_flip) / 100 * int8_count

        sim.compute_encodings(inputs=[make_dummy_input(model)])

    @pytest.mark.parametrize("percent_flip", [30.5, 50, 80])
    def test_flip_to_int16(self, percent_flip):
        model = models_for_tests.single_residual_model().model
        fp_session = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CUDAExecutionProvider"]
        )

        sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
        q_ops = [op for op in sim.qc_quantize_op_dict.values() if op.enabled]
        int8_count = sum(1 for op in q_ops if op.bitwidth == 8)
        fp_count = sum(1 for op in q_ops if op.data_type == QuantizationDataType.float)
        assert fp_count == 0

        inputs = [make_dummy_input(model)]
        psnr_eval_fn = make_psnr_eval_fn(fp_session, inputs)
        layer_sensitivity_dict = analyze_per_layer_sensitivity(
            sim, eval_fn=psnr_eval_fn
        )
        flip_layers_to_higher_precision(
            sim, layer_sensitivity_dict, percent_flip, override_precision=int16
        )

        int16_count = sum(1 for op in q_ops if op.bitwidth == 16 and op.enabled)
        assert int16_count >= percent_flip / 100 * int8_count

        sim.compute_encodings(inputs=[make_dummy_input(model)])


def test_multi_output_psnr_eval_fn():
    model = models_for_tests.multi_output_model().model
    fp_session = onnxruntime.InferenceSession(model.SerializeToString())
    inputs = [make_dummy_input(model)]

    psnr_eval_fn_0 = make_psnr_eval_fn(fp_session, inputs, output_indices=0)
    psnr_eval_fn_1 = make_psnr_eval_fn(fp_session, inputs, output_indices=1)
    psnr_eval_fn_all = make_psnr_eval_fn(fp_session, inputs, output_indices=None)

    sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
    sim.compute_encodings(inputs)

    assert psnr_eval_fn_0(sim.session) != psnr_eval_fn_1(sim.session)
    assert psnr_eval_fn_all(sim.session) == min(
        psnr_eval_fn_0(sim.session), psnr_eval_fn_1(sim.session)
    )


def test_multi_output_psnr_ignore_integer_output():
    model = models_for_tests.integer_output_model()
    fp_session = onnxruntime.InferenceSession(model.SerializeToString())
    inputs = [make_dummy_input(model)]
    # Default PSNR eval FN should ignore integer output
    default_psnr_eval_fn = make_psnr_eval_fn(fp_session, inputs, output_indices=None)
    single_output_evals = [
        make_psnr_eval_fn(fp_session, inputs, output_indices=i)
        for i in range(len(model.graph.output))
    ]
    # Explicit PSNR eval FN should not ignore integer output
    all_output_psnr_eval_fn = make_psnr_eval_fn(
        fp_session, inputs, output_indices=[i for i in range(len(model.graph.output))]
    )

    sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
    sim.compute_encodings(inputs)

    assert default_psnr_eval_fn(sim.session) == min(
        fn(sim.session) for fn in single_output_evals[:-1]
    )
    assert all_output_psnr_eval_fn(sim.session) == min(
        fn(sim.session) for fn in single_output_evals
    )
