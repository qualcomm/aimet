# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import tempfile
import warnings
import ml_dtypes
import numpy as np
import onnxruntime
import torch

from aimet_onnx.common.defs import QuantizationDataType
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx import analyze_per_layer_sensitivity
from aimet_onnx import int8, int16, float16, int4, qtype
from aimet_onnx.utils import (
    OrtInferenceSession,
    make_dummy_input,
    make_psnr_eval_fn,
)
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

    def test_flip_to_int16_applies_layernorm_exception_rule(self):
        model = models_for_tests.standalone_layernorm(input_shape=(1, 8, 32))
        sim = QuantizationSimModel(
            model, param_type=int8, activation_type=int8, config_file="htp_v81"
        )
        assert not sim.qc_quantize_op_dict["scale"].use_symmetric_encodings

        layer_sensitivity_dict = {
            layer_name: 0.0 for layer_name in sim.connected_graph.get_all_ops()
        }
        flip_layers_to_higher_precision(
            sim, layer_sensitivity_dict, percent_to_flip=100, override_precision=int16
        )

        # After flipping, int16 exception rule should force scale to symmetric
        assert sim.qc_quantize_op_dict["scale"].use_symmetric_encodings

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

    @pytest.mark.parametrize(
        "param_type, act_type", [(int8, int16), ("int8", "int16"), (float16, float16)]
    )
    def test_flip_to_precision_tuple(self, param_type, act_type):
        model = models_for_tests.single_residual_model().model

        sim = QuantizationSimModel(model, param_type=int4, activation_type=int8)
        enabled_quantizers = {
            name for name, q in sim.qc_quantize_op_dict.items() if q.enabled
        }

        layer_sensitivity_dict = {
            op.name: 1.0 for op in sim.connected_graph.ordered_ops
        }

        # Flip 100% of layers to override precision
        flip_layers_to_higher_precision(
            sim, layer_sensitivity_dict, 100, override_precision=(param_type, act_type)
        )

        for param_name in sim.param_names:
            quantizer = sim.qc_quantize_op_dict[param_name]
            if quantizer.enabled:
                assert quantizer.precision() == qtype.as_qtype(param_type)

        for activation_name in sim.activation_names:
            quantizer = sim.qc_quantize_op_dict[activation_name]
            if quantizer.enabled:
                assert quantizer.precision() == qtype.as_qtype(act_type)

        # Lite MP does not disable quantizers
        assert enabled_quantizers == {
            name for name, q in sim.qc_quantize_op_dict.items() if q.enabled
        }

    @pytest.mark.parametrize(
        "precision", [(int8, int8, int16), (int8, int4), ("unsupported", int16)]
    )
    def test_flip_layers_rejects_invalid_precisions(self, precision):
        model = models_for_tests.single_residual_model().model

        sim = QuantizationSimModel(model, param_type=int4, activation_type=int8)

        layer_sensitivity_dict = {
            op.name: 1.0 for op in sim.connected_graph.ordered_ops
        }

        with pytest.raises(ValueError):
            flip_layers_to_higher_precision(
                sim, layer_sensitivity_dict, override_precision=precision
            )

    @pytest.mark.cuda
    @pytest.mark.parametrize("override_precision", [int16, float16])
    @pytest.mark.parametrize("io_dtype", [torch.float32, torch.bfloat16])
    def test_flip_bfloat16_model(self, io_dtype, override_precision):
        """
        Given: An int8 sim of a model computing in bfloat16, with float32 or bfloat16 graph
               inputs/outputs
        When: Rank its layers with the default PSNR callback and flip half of them to
              override_precision
        Then: Every layer with quantizers is ranked, the most sensitive layers' quantizers
              get the override precision, and the sim's PSNR improves after recalibration
        """
        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            pytest.skip("CUDA EP not available")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        model = models_for_tests.simple_mlp_model(torch.bfloat16, io_dtype=io_dtype)
        rng = np.random.default_rng(0)
        np_dtype = np.float32 if io_dtype == torch.float32 else ml_dtypes.bfloat16
        inputs = [
            {"input": rng.standard_normal((8, 16)).astype(np_dtype)} for _ in range(2)
        ]

        psnr_eval_fn = make_psnr_eval_fn(OrtInferenceSession(model, providers), inputs)
        sim = QuantizationSimModel(
            model, param_type=int8, activation_type=int8, providers=providers
        )
        sim.compute_encodings(inputs)
        psnr_before = psnr_eval_fn(sim.session)

        layer_sensitivity_dict = analyze_per_layer_sensitivity(
            sim, eval_fn=psnr_eval_fn
        )
        cg_ops = sim.connected_graph.get_all_ops()

        def op_quantizers(layer_name):
            inputs_q, outputs_q, params_q = sim.get_op_quantizers(cg_ops[layer_name])
            return [q for q in [*inputs_q, *outputs_q, *params_q.values()] if q.enabled]

        # Only layers with quantizers are ranked, e.g. not the trailing Cast
        assert set(layer_sensitivity_dict) == {
            name for name in cg_ops if op_quantizers(name)
        }
        flip_layers_to_higher_precision(
            sim,
            layer_sensitivity_dict,
            percent_to_flip=50,
            override_precision=override_precision,
        )

        flipped = sorted(layer_sensitivity_dict, key=layer_sensitivity_dict.get)[:2]
        expected = override_precision.to_legacy_repr()
        for layer_name in flipped:
            for quantizer in op_quantizers(layer_name):
                assert (quantizer.data_type, quantizer.bitwidth) == expected, layer_name

        sim.compute_encodings(inputs)
        assert psnr_eval_fn(sim.session) > psnr_before + 3


def test_multi_output_psnr_eval_fn():
    model = models_for_tests.multi_output_model().model
    fp_session = onnxruntime.InferenceSession(model.SerializeToString())
    inputs = [make_dummy_input(model)]

    psnr_eval_fn_0 = make_psnr_eval_fn(fp_session, inputs, output_indices=0)
    psnr_eval_fn_1 = make_psnr_eval_fn(fp_session, inputs, output_indices=1)
    with pytest.warns():
        default_psnr_eval_fn = make_psnr_eval_fn(fp_session, inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        explicit_all_psnr_eval_fn = make_psnr_eval_fn(fp_session, inputs, None)

    sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
    sim.compute_encodings(inputs)

    assert psnr_eval_fn_0(sim.session) != psnr_eval_fn_1(sim.session)
    assert default_psnr_eval_fn(sim.session) == min(
        psnr_eval_fn_0(sim.session), psnr_eval_fn_1(sim.session)
    )
    assert default_psnr_eval_fn(sim.session) == explicit_all_psnr_eval_fn(sim.session)


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
