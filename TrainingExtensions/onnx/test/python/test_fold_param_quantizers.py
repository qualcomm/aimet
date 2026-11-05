# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for QuantizationSimModel.fold_param_quantizers (aimet-onnx)"""

import json
import os
import tempfile

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import helper, numpy_helper

import aimet_onnx
from aimet_onnx.common.defs import QuantizationDataType, QuantScheme
from aimet_onnx.qc_quantize_op import QcQuantizeOp
from aimet_onnx.quantsim import (
    QuantizationSimModel,
    set_grouped_blockwise_quantization_for_weights,
)
from aimet_onnx.utils import (
    OrtInferenceSession,
    _run_session_to_numpy,
    make_dummy_input,
    to_array,
)

from .models.models_for_tests import build_dummy_model, simple_mlp_model

CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_calibrated_sim(**kwargs):
    model = build_dummy_model()
    dummy_input = make_dummy_input(model)
    sim = QuantizationSimModel(model, **kwargs)
    sim.compute_encodings([dummy_input])
    return sim, model, dummy_input


def _constant_weight_matmul_model(in_features=8, out_features=6):
    """MatMul whose weight is produced by a ``Constant`` node (not an initializer)."""
    weight = np.random.randn(in_features, out_features).astype(np.float32)
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["weight"],
        name="weight_const",
        value=numpy_helper.from_array(weight, name="weight_value"),
    )
    matmul = helper.make_node(
        "MatMul", inputs=["input", "weight"], outputs=["output"], name="matmul"
    )
    inp = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [1, in_features]
    )
    out = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, [1, out_features]
    )
    graph = helper.make_graph([const_node, matmul], "const_matmul", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


class TestFoldParamQuantizers:
    def test_output_is_numerically_equivalent(self):
        """Folding should not change the quantized inference output."""
        sim, _, dummy_input = _make_calibrated_sim()

        out_before = sim.session.run(None, dummy_input)[0]

        sim.fold_param_quantizers()

        out_after = sim.session.run(None, dummy_input)[0]
        assert np.allclose(out_before, out_after, atol=1e-6)

    def test_param_quantize_nodes_removed(self):
        """All folded param QcQuantizeOp nodes should be removed from the graph."""
        sim, _, _ = _make_calibrated_sim()

        sim.fold_param_quantizers()

        folded_names = set(sim._folded_param_quantizers)
        assert folded_names, "expected some param quantizers to fold"

        # No QcQuantizeOp node should remain for the folded params
        remaining = [
            node
            for node in sim.model.model.graph.node
            if node.op_type == "QcQuantizeOp" and node.input[0] in folded_names
        ]
        assert not remaining

        # The quantizer objects are retained (still enabled, in the dict) so encodings
        # can still be exported; they simply no longer have a node in the graph.
        for name in folded_names:
            assert name in sim.qc_quantize_op_dict
            assert sim.qc_quantize_op_dict[name].enabled

    def test_initializers_overwritten_with_qdq_values(self):
        """The model initializers should hold quantize-dequantize'd weights after fold."""
        sim, _, _ = _make_calibrated_sim()

        weight_name = "fc_w"
        init_before = {
            init.name: np.frombuffer(init.raw_data, dtype=np.float32).copy()
            for init in sim.model.model.graph.initializer
            if init.name == weight_name
        }[weight_name]

        # The expected QDQ value is what the partial-model QDQ computation produces.
        expected = sim._get_qdq_parameters()[weight_name].flatten()

        sim.fold_param_quantizers()

        init_after = {
            init.name: np.frombuffer(init.raw_data, dtype=np.float32).copy()
            for init in sim.model.model.graph.initializer
            if init.name == weight_name
        }[weight_name]

        # The folded weights should differ from the original float weights...
        assert not np.allclose(init_before, init_after)
        # ...and should match the simulated quantize-dequantize values.
        assert np.allclose(init_after, expected, atol=1e-6)

    def test_activation_quantizers_untouched(self):
        """Folding params should not remove activation quantizers."""
        sim, _, _ = _make_calibrated_sim()

        activation_nodes_before = {
            node.name
            for node in sim.model.model.graph.node
            if node.op_type == "QcQuantizeOp" and node.input[0] in sim.activation_names
        }

        sim.fold_param_quantizers()

        activation_nodes_after = {
            node.name
            for node in sim.model.model.graph.node
            if node.op_type == "QcQuantizeOp" and node.input[0] in sim.activation_names
        }
        assert activation_nodes_before == activation_nodes_after
        for name in sim.activation_names:
            assert name in sim.qc_quantize_op_dict

    def test_int32_params_are_skipped(self):
        """int32 (>= 32 bit) param quantizers must not be folded."""
        sim, _, _ = _make_calibrated_sim()

        # Force one param quantizer to 32 bits and recompute its encoding.
        weight_name = "fc_w"
        sim.qc_quantize_op_dict[weight_name].set_bitwidth(32)
        sim.qc_quantize_op_dict[weight_name].data_type = QuantizationDataType.int
        sim._rebuild_session()

        sim.fold_param_quantizers()

        # The int32 param should remain an active, in-graph quantizer.
        assert weight_name in sim.qc_quantize_op_dict
        assert weight_name not in sim._folded_param_quantizers
        assert any(
            node.op_type == "QcQuantizeOp" and node.input[0] == weight_name
            for node in sim.model.model.graph.node
        )

    def test_fold_is_idempotent(self):
        """A second fold call should be a safe no-op."""
        sim, _, dummy_input = _make_calibrated_sim()

        sim.fold_param_quantizers()
        folded = dict(sim._folded_param_quantizers)
        out_after_first = sim.session.run(None, dummy_input)[0]

        sim.fold_param_quantizers()
        assert dict(sim._folded_param_quantizers) == folded
        out_after_second = sim.session.run(None, dummy_input)[0]
        assert np.allclose(out_after_first, out_after_second)

    @pytest.mark.parametrize("encoding_version", ["1.0.0", "2.0.0"])
    def test_export_emits_folded_param_encodings(self, encoding_version, tmp_dir):
        """Export after folding should still emit encodings for the folded params."""
        sim, _, _ = _make_calibrated_sim()

        sim.fold_param_quantizers()
        folded_names = set(sim._folded_param_quantizers)

        sim.export(tmp_dir, "folded_sim", encoding_version=encoding_version)

        with open(os.path.join(tmp_dir, "folded_sim.encodings")) as f:
            encodings = json.load(f)

        if encoding_version == "1.0.0":
            exported_param_names = {enc["name"] for enc in encodings["param_encodings"]}
        else:
            exported_param_names = {enc["name"] for enc in encodings["encodings"]}

        assert folded_names <= exported_param_names

        # The exported model should not contain QcQuantizeOps for the folded params.
        exported = onnx.load(os.path.join(tmp_dir, "folded_sim.onnx"))
        assert not any(
            node.op_type == "QcQuantizeOp" and node.input[0] in folded_names
            for node in exported.graph.node
        )

    def test_fold_constant_node_param(self):
        """Params produced by a Constant node (not an initializer) should fold correctly."""
        model = _constant_weight_matmul_model()
        dummy_input = make_dummy_input(model)
        sim = QuantizationSimModel(
            model, param_type=aimet_onnx.int8, activation_type=aimet_onnx.int16
        )
        sim.compute_encodings([dummy_input])

        out_before = sim.session.run(None, dummy_input)[0]

        sim.fold_param_quantizers()

        # The Constant-node weight should have been folded...
        assert "weight" in sim._folded_param_quantizers
        # ...its QcQuantizeOp removed...
        assert not any(
            node.op_type == "QcQuantizeOp" and node.input[0] == "weight"
            for node in sim.model.model.graph.node
        )
        # ...and inference output should be unchanged.
        out_after = sim.session.run(None, dummy_input)[0]
        assert np.allclose(out_before, out_after, atol=1e-6)

    def test_fold_float16_params(self):
        """Float (fp16) param quantizers fold via an fp16 round-trip, not int QDQ."""
        sim, _, dummy_input = _make_calibrated_sim(
            param_type=aimet_onnx.float16, activation_type=aimet_onnx.float16
        )

        out_before = sim.session.run(None, dummy_input)[0]

        # The folded value should be the float16 round-trip of the raw weights.
        weight_name = "fc_w"
        raw = {
            init.name: numpy_helper.to_array(init)
            for init in sim.model.model.graph.initializer
        }[weight_name]
        expected = raw.astype(np.float16).astype(raw.dtype)

        sim.fold_param_quantizers()

        assert weight_name in sim._folded_param_quantizers
        init_after = {
            init.name: np.frombuffer(init.raw_data, dtype=np.float32).copy()
            for init in sim.model.model.graph.initializer
            if init.name == weight_name
        }[weight_name]
        assert np.allclose(init_after, expected.flatten(), atol=0)

        out_after = sim.session.run(None, dummy_input)[0]
        assert np.allclose(out_before, out_after, atol=1e-6)

    def test_fold_does_not_build_partial_onnx_graph(self, monkeypatch):
        """
        Regression guard for the protobuf 2GB limit.

        ``_get_qdq_parameters`` must compute QDQ params directly from each quantizer,
        without materializing every parameter into a throwaway ONNX graph (which would
        overflow protobuf's 2GB message cap for large models). This asserts the method
        never calls ``onnx.helper.make_graph``.
        """
        sim, _, _ = _make_calibrated_sim()

        def _fail(*args, **kwargs):
            raise AssertionError(
                "_get_qdq_parameters must not build an ONNX graph (protobuf 2GB limit)"
            )

        monkeypatch.setattr(onnx.helper, "make_graph", _fail)

        sim.fold_param_quantizers()
        assert sim._folded_param_quantizers

    def test_fold_holds_one_qdq_param_at_a_time(self):
        """
        Folding streams through the params: at most one quantize-dequantized param is
        alive at a time, so peak memory grows by the largest param, not by a full copy
        of the model's weights.
        """
        sim, _, _ = _make_calibrated_sim()

        alive = set()
        max_alive = 0
        original = QcQuantizeOp.quantize_dequantize

        class _Tracked(np.ndarray):
            def __del__(self):
                alive.discard(id(self))

        def tracked(self, input_tensor):
            nonlocal max_alive
            out = original(self, input_tensor).view(_Tracked)
            alive.add(id(out))
            max_alive = max(max_alive, len(alive))
            return out

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(QcQuantizeOp, "quantize_dequantize", tracked)
            sim.fold_param_quantizers()

        # Guard the premise: several params must fold for "one at a time" to mean anything
        assert len(sim._folded_param_quantizers) > 1
        assert max_alive == 1

    def test_fold_clears_typed_data_fields(self):
        """
        Given: A weight stored in a typed field (float_data) rather than raw_data
        When: Param quantizers are folded
        Then: The weight holds only raw_data, so the model passes onnx.checker, and the
              output is unchanged
        """
        model = build_dummy_model()
        weight = next(i for i in model.graph.initializer if i.name == "fc_w")
        values = numpy_helper.to_array(weight).copy()
        weight.ClearField("raw_data")
        weight.float_data.extend(values.flatten().tolist())
        dummy_input = make_dummy_input(model)
        sim = QuantizationSimModel(model)
        sim.compute_encodings([dummy_input])
        out_before = sim.session.run(None, dummy_input)[0]

        sim.fold_param_quantizers()

        folded = next(i for i in sim.model.model.graph.initializer if i.name == "fc_w")
        assert not folded.float_data and folded.HasField("raw_data")
        onnx.checker.check_tensor(folded)
        out_after = sim.session.run(None, dummy_input)[0]
        assert np.allclose(out_before, out_after, atol=1e-6)

    def test_fold_rejects_scalar_constant_param(self, monkeypatch):
        """
        Given: A param produced by a Constant that stores it in a list attribute
               (value_floats) instead of a 'value' tensor
        When: Param quantizers are folded
        Then: Folding raises a clear error instead of corrupting the node
        """
        sim, _, _ = _make_calibrated_sim()
        weight_name = "fc_w"
        init = next(
            i for i in sim.model.model.graph.initializer if i.name == weight_name
        )
        values = numpy_helper.to_array(init).flatten().tolist()
        sim.model.model.graph.initializer.remove(init)
        sim.model.model.graph.node.insert(
            0,
            helper.make_node(
                "Constant", [], [weight_name], name="w_const", value_floats=values
            ),
        )

        with pytest.raises(RuntimeError, match="only the 'value' \\(tensor\\) form"):
            sim.fold_param_quantizers()

    @pytest.mark.parametrize(
        "attr, values, expected_field",
        [
            ("value_float", 0.5, "f"),
            ("value_floats", [0.5, 0.25], "floats"),
            ("value_int", 3, "i"),
            ("value_ints", [3, 4], "ints"),
        ],
    )
    def test_overwrite_scalar_and_list_constants(self, attr, values, expected_field):
        """
        Given: A Constant holding its value in a scalar or list attribute
        When: Its value is overwritten
        Then: The value lands in that attribute's own field
        """
        node = helper.make_node("Constant", [], ["c"], name="c", **{attr: values})
        graph = helper.make_graph([node], "g", [], [])
        model = helper.make_model(graph)
        new_value = np.asarray(values) * 2
        QuantizationSimModel._overwrite_parameters(model, {"c": new_value})

        written = model.graph.node[0].attribute[0]
        assert written.name == attr
        assert np.allclose(getattr(written, expected_field), new_value)


def _bf16_sim(param_type, *, lpbq=False):
    model = simple_mlp_model(torch.bfloat16)
    sim = QuantizationSimModel(
        model,
        providers=CUDA_PROVIDERS,
        param_type=param_type,
        activation_type=aimet_onnx.int16,
        quant_scheme=QuantScheme.min_max,
    )
    if lpbq:
        set_grouped_blockwise_quantization_for_weights(
            sim, ("Gemm", "MatMul"), 4, 8, block_size=4, strict=False
        )
    return sim


def _in_graph_qdq_weights(sim, inputs, names):
    """The ``<param>_qdq`` tensors the in-graph QcQuantizeOps feed their consumers."""
    probe = onnx.ModelProto()
    probe.CopyFrom(sim.model.model)
    for name in names:
        probe.graph.output.append(
            helper.make_tensor_value_info(
                name + "_qdq", onnx.TensorProto.BFLOAT16, None
            )
        )
    session = OrtInferenceSession(
        probe, sim.providers, session_options=sim._ort_session_options
    )
    outputs = [name + "_qdq" for name in names]
    return dict(zip(names, _run_session_to_numpy(session, inputs, outputs)))


def _initializer(sim, name):
    return next(
        to_array(init)
        for init in sim.model.model.graph.initializer
        if init.name == name
    )


@pytest.mark.cuda
class TestFoldBfloat16Params:
    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        # ORT's CPU EP has no bfloat16 MatMul/Gemm kernels
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CUDA EP not available")

    @pytest.mark.parametrize(
        "param_type, lpbq",
        [(aimet_onnx.int8, False), (aimet_onnx.int4, False), (aimet_onnx.int4, True)],
        ids=["int8", "int4", "int4_lpbq"],
    )
    def test_folded_weights_match_in_graph_qdq(self, param_type, lpbq):
        """
        Given: A bfloat16 sim on CUDA, calibrated
        When: Param quantizers are folded (on the host CPU)
        Then: Each folded bfloat16 weight equals what the in-graph CUDA QcQuantizeOp
              produces on every forward, and outputs are unchanged

        Bit-exact here, but not guaranteed at scale: the CUDA kernel divides with
        --use_fast_math, so a weight within ~1e-3 of a rounding tie (a few per million)
        can land one quantization step away from the host-side fold.
        """
        np.random.seed(0)
        sim = _bf16_sim(param_type, lpbq=lpbq)
        inputs = {"input": np.random.randn(8, 16).astype(np.float32)}
        sim.compute_encodings([inputs])
        names = [
            name for name in sim.param_names if sim.qc_quantize_op_dict[name].enabled
        ]
        assert names

        in_graph = _in_graph_qdq_weights(sim, inputs, names)
        out_before = _run_session_to_numpy(sim.session, inputs)[0]

        sim.fold_param_quantizers()

        assert set(sim._folded_param_quantizers) == set(names)
        for name in names:
            folded = _initializer(sim, name)
            assert folded.dtype == ml_dtypes.bfloat16
            np.testing.assert_array_equal(
                folded.view(np.uint16), in_graph[name].view(np.uint16), err_msg=name
            )
        out_after = _run_session_to_numpy(sim.session, inputs)[0]
        np.testing.assert_array_equal(out_after, out_before)

    def test_float16_quantizer_leaves_bfloat16_weights_unchanged(self):
        """
        Given: A bfloat16 sim whose weights have enabled float16 quantizers, and a weight
               value outside float16's range
        When: Param quantizers are folded
        Then: The weights are left as they are, as the in-graph QcQuantizeOp passes 16-bit
              floats through unchanged, rather than round-tripped through float16
        """
        model = simple_mlp_model(torch.bfloat16)
        weight = next(
            init
            for init in model.graph.initializer
            if init.data_type == onnx.TensorProto.BFLOAT16 and len(init.dims) == 2
        )
        value = to_array(weight).copy()
        value.flat[0] = 1e5  # float16 max is 65504
        weight.raw_data = value.tobytes()

        sim = QuantizationSimModel(
            model,
            providers=CUDA_PROVIDERS,
            param_type=aimet_onnx.float16,
            activation_type=aimet_onnx.float16,
        )
        inputs = {"input": np.random.randn(4, 16).astype(np.float32)}
        names = [
            name for name in sim.param_names if sim.qc_quantize_op_dict[name].enabled
        ]
        assert weight.name in names
        in_graph = _in_graph_qdq_weights(sim, inputs, names)
        raw = {name: _initializer(sim, name).copy() for name in names}

        sim.fold_param_quantizers()

        for name in names:
            folded = _initializer(sim, name)
            np.testing.assert_array_equal(
                folded.view(np.uint16), raw[name].view(np.uint16)
            )
            np.testing.assert_array_equal(
                folded.view(np.uint16), in_graph[name].view(np.uint16)
            )
        assert np.isfinite(_initializer(sim, weight.name).astype(np.float32)).all()
