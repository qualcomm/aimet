# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for aimet_onnx.analysis.sensitivity (GenAILab-free, tiny ONNX model)."""

import ml_dtypes
import numpy as np
import onnxruntime
import pytest
import torch

from aimet_onnx.quantsim import QuantizationSimModel, compute_encodings
from aimet_onnx import int8, int16
from aimet_onnx.lite_mp import flip_layers_to_higher_precision
from aimet_onnx.utils import OrtInferenceSession, make_dummy_input, make_psnr_eval_fn
from aimet_onnx.analysis import (
    SensitivityMetric,
    make_topk_logit_psnr_metric,
    analyze_per_quantizer_sensitivity,
    get_quantizer_op_names,
    group_by_op_name,
)

from ..models import models_for_tests


def _calibrated_sim():
    """Build and calibrate a small W8A8 sim plus its FP session and inputs."""
    model = models_for_tests.single_residual_model().model
    fp_session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sim = QuantizationSimModel(model, param_type=int8, activation_type=int8)
    inputs = [make_dummy_input(model)]
    with compute_encodings(sim):
        sim.session.run(None, inputs[0])
    return sim, fp_session, inputs


class TestSensitivityMetric:
    def test_ranking_direction(self):
        higher = SensitivityMetric("h", lambda s: 0.0, higher_is_worse=True)
        lower = SensitivityMetric("l", lambda s: 0.0, higher_is_worse=False)
        assert higher.sensitivity_score(5.0) == 5.0
        assert lower.sensitivity_score(5.0) == -5.0

    def test_non_callable_eval_fn_raises(self):
        with pytest.raises(ValueError):
            SensitivityMetric("bad", eval_fn=object())


class TestPerQuantizerSensitivity:
    def test_returns_score_per_quantizer(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        scores = analyze_per_quantizer_sensitivity(sim, metric)

        enabled = [n for n, q in sim.qc_quantize_op_dict.items() if q.enabled]
        assert set(scores) == set(enabled)
        assert all(isinstance(v, float) for v in scores.values())

    def test_ordered_most_sensitive_first(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        scores = analyze_per_quantizer_sensitivity(sim, metric)
        ranked = [metric.sensitivity_score(v) for v in scores.values()]
        assert ranked == sorted(ranked, reverse=True)

    def test_group_fn_collapses_units(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        # Collapse all quantizers into one group -> a single score entry.
        scores = analyze_per_quantizer_sensitivity(
            sim, metric, group_fn=lambda name: "all"
        )
        assert list(scores) == ["all"]

    def test_group_fn_none_skips_quantizer(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        # Drop everything -> no groups -> error.
        with pytest.raises(RuntimeError):
            analyze_per_quantizer_sensitivity(sim, metric, group_fn=lambda name: None)

    def test_restores_enabled_state(self):
        sim, fp_session, inputs = _calibrated_sim()
        before = {n: q.enabled for n, q in sim.qc_quantize_op_dict.items()}
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        analyze_per_quantizer_sensitivity(sim, metric)
        after = {n: q.enabled for n, q in sim.qc_quantize_op_dict.items()}
        assert before == after

    def test_group_fn_subset_restricts_sweep(self):
        # A group_fn that returns None for all but a chosen subset (the pattern
        # used for KV-cache-only sweeps) restricts the analysis to that subset.
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        selected = [n for n, q in sim.qc_quantize_op_dict.items() if q.enabled][:2]
        scores = analyze_per_quantizer_sensitivity(
            sim, metric, group_fn=lambda name: name if name in selected else None
        )
        assert set(scores) == set(selected)


class TestQuantizerOpNames:
    def test_every_quantizer_maps_to_an_op(self):
        sim, _, _ = _calibrated_sim()
        op_names = get_quantizer_op_names(sim)
        assert set(op_names) == set(sim.qc_quantize_op_dict)
        assert set(op_names.values()) <= set(sim.connected_graph.get_all_ops())

    def test_param_maps_to_owning_op(self):
        # A weight belongs to the op it parameterizes, not to a consumer of that
        # op's output.
        sim, _, _ = _calibrated_sim()
        op_names = get_quantizer_op_names(sim)
        assert op_names["conv3.weight"] == "/conv3/Conv"
        assert op_names["fc.weight"] == "/fc/Gemm"

    def test_activation_maps_to_producer_op(self):
        sim, _, _ = _calibrated_sim()
        op_names = get_quantizer_op_names(sim)
        assert op_names["/relu1/Relu_output_0"] == "/relu1/Relu"

    def test_graph_input_maps_to_consumer_op(self):
        # 'input' has no producer, so it is attributed to the op consuming it.
        sim, _, _ = _calibrated_sim()
        op_names = get_quantizer_op_names(sim)
        assert op_names["input"] == "/conv1/Conv"

    def test_agrees_with_onnx_graph(self):
        # Check the mapping against the ONNX graph itself rather than the
        # connected graph it is derived from: an initializer resolves to the node
        # consuming it, a node output to the node producing it.
        sim, _, _ = _calibrated_sim()
        op_names = get_quantizer_op_names(sim)
        # The sim's own graph has QcQuantizeOp nodes spliced in, so compare
        # against a fresh copy of the float graph the sim was built from.
        graph = models_for_tests.single_residual_model().model.graph

        initializers = {init.name for init in graph.initializer}
        producer = {out: node.name for node in graph.node for out in node.output}
        consumers = {}
        for node in graph.node:
            for tensor in node.input:
                consumers.setdefault(tensor, []).append(node.name)

        for name, op_name in op_names.items():
            if name in initializers:
                assert consumers[name] == [op_name]
            elif name in producer:
                assert producer[name] == op_name


class TestGroupByOpName:
    def test_scores_are_keyed_by_op_name(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        scores = analyze_per_quantizer_sensitivity(
            sim, metric, group_fn=group_by_op_name(sim)
        )
        cg_ops = sim.connected_graph.get_all_ops()
        assert scores
        assert set(scores) <= set(cg_ops)

        # Every enabled quantizer's op is represented, and nothing else is.
        expected = {
            get_quantizer_op_names(sim)[name]
            for name, q in sim.qc_quantize_op_dict.items()
            if q.enabled
        }
        assert set(scores) == expected

    def test_keys_are_accepted_by_lite_mp(self):
        # The point of op-name keying: results feed lite_mp with no remapping.
        sim, fp_session, inputs = _calibrated_sim()
        metric = SensitivityMetric(
            "psnr", make_psnr_eval_fn(fp_session, inputs), higher_is_worse=False
        )
        scores = analyze_per_quantizer_sensitivity(
            sim, metric, group_fn=group_by_op_name(sim)
        )
        flip_layers_to_higher_precision(
            sim, scores, percent_to_flip=100, override_precision=int16
        )


class TestTopkLogitPsnrMetric:
    def test_returns_float(self):
        sim, fp_session, inputs = _calibrated_sim()
        metric = make_topk_logit_psnr_metric(fp_session, inputs, k=5)
        assert metric.higher_is_worse is False
        score = metric(sim.session)
        assert isinstance(score, float)

    @pytest.mark.cuda
    def test_bfloat16_logits(self):
        """
        Given: Model computing in bfloat16, with bfloat16 graph inputs/outputs
        When: Evaluate the top-k logit PSNR metric on its sim
        Then: It returns a PSNR, close to that of the equivalent float32 model
        """
        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            pytest.skip("CUDA EP not available")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        rng = np.random.default_rng(0)
        fp32_inputs = [{"input": rng.standard_normal((4, 16)).astype(np.float32)}]

        def score(model, inputs):
            fp_session = OrtInferenceSession(model, providers)
            metric = make_topk_logit_psnr_metric(fp_session, inputs, k=4)
            sim = QuantizationSimModel(
                model, param_type="int8", activation_type="int16", providers=providers
            )
            sim.compute_encodings(inputs)
            return metric(sim.session)

        fp32_score = score(
            models_for_tests.simple_mlp_model(torch.float32), fp32_inputs
        )
        bf16_score = score(
            models_for_tests.simple_mlp_model(torch.bfloat16, io_dtype=torch.bfloat16),
            [{"input": fp32_inputs[0]["input"].astype(ml_dtypes.bfloat16)}],
        )
        assert isinstance(bf16_score, float)
        assert abs(bf16_score - fp32_score) < 3.0
