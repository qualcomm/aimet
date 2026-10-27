# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for aimet_onnx.export_passes."""

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

import aimet_onnx
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.export_passes import convert_bfloat16_to_float32_pass
from aimet_onnx.quantsim import QuantizationSimModel

from .models import models_for_tests

BF16, FP32 = TensorProto.BFLOAT16, TensorProto.FLOAT
CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _bf16_tensor(name, values):
    values = np.asarray(values, dtype=np.float32)
    return helper.make_tensor(
        name, BF16, values.shape, values.astype(ml_dtypes.bfloat16).tobytes(), raw=True
    )


def _make_model(nodes, inputs, outputs, initializers=(), functions=(), opset=21):
    graph = helper.make_graph(nodes, "g", inputs, outputs, list(initializers))
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", opset),
            *(helper.make_opsetid(f.domain, 1) for f in functions),
        ],
        functions=list(functions),
    )
    model.ir_version = 10
    return model


def _bf16_sites(model: onnx.ModelProto) -> list:
    """Every place in ``model`` (including subgraphs and functions) still holding bf16."""
    sites = []

    def visit_nodes(nodes, where):
        for node in nodes:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT and attr.i == BF16:
                    sites.append(f"{where}:{node.op_type}.{attr.name}")
                elif (
                    attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == BF16
                ):
                    sites.append(f"{where}:{node.op_type}.{attr.name}")
                elif attr.type == onnx.AttributeProto.GRAPH:
                    visit_graph(attr.g, f"{where}/{node.op_type}.{attr.name}")

    def visit_graph(graph, where):
        for value in (*graph.input, *graph.output, *graph.value_info):
            if value.type.tensor_type.elem_type == BF16:
                sites.append(f"{where}:{value.name}")
        sites.extend(
            f"{where}:{init.name}"
            for init in graph.initializer
            if init.data_type == BF16
        )
        visit_nodes(graph.node, where)

    visit_graph(model.graph, "main")
    for function in model.functions:
        visit_nodes(function.node, f"function {function.name}")
    return sites


class TestConvertBFloat16ToFloat32:
    def test_graph_io_initializers_and_casts(self):
        """
        Given: A bf16 graph with bf16 I/O, a bf16 initializer and Casts to bf16
        When: Convert
        Then: Every bf16 site becomes fp32 and initializer values are preserved
        """
        weight = np.array([0.5, -1.25, 3.0, 0.0078125], dtype=np.float32)
        model = _make_model(
            [
                helper.make_node("Cast", ["x"], ["x_bf16"], to=BF16),
                helper.make_node("Mul", ["x_bf16", "w"], ["y_bf16"]),
                helper.make_node("Cast", ["y_bf16"], ["y_fp32"], to=FP32),
                helper.make_node("Cast", ["y_fp32"], ["y"], to=BF16),
            ],
            [helper.make_tensor_value_info("x", BF16, [4])],
            [helper.make_tensor_value_info("y", BF16, [4])],
            [_bf16_tensor("w", weight)],
        )
        assert _bf16_sites(model)

        converted = convert_bfloat16_to_float32_pass(model)

        assert _bf16_sites(converted) == []
        onnx.checker.check_model(converted, full_check=True)
        (w,) = converted.graph.initializer
        assert w.data_type == FP32
        np.testing.assert_array_equal(numpy_helper.to_array(w), weight)
        # Casts to other types are untouched.
        casts = [n for n in converted.graph.node if n.op_type == "Cast"]
        assert [a.i for n in casts for a in n.attribute if a.name == "to"] == [
            FP32,
            FP32,
            FP32,
        ]

    def test_input_model_is_not_modified(self):
        model = _make_model(
            [helper.make_node("Identity", ["x"], ["y"])],
            [helper.make_tensor_value_info("x", BF16, [2])],
            [helper.make_tensor_value_info("y", BF16, [2])],
        )
        before = model.SerializeToString()

        convert_bfloat16_to_float32_pass(model)

        assert model.SerializeToString() == before

    def test_constant_and_constant_of_shape_values(self):
        """
        Given: Constant and ConstantOfShape nodes with bf16 tensor values
        When: Convert
        Then: The values become fp32 with the same numbers
        """
        model = _make_model(
            [
                helper.make_node(
                    "Constant", [], ["c"], value=_bf16_tensor("c", [1.5, -2.0])
                ),
                helper.make_node(
                    "ConstantOfShape",
                    ["shape"],
                    ["filled"],
                    value=_bf16_tensor("v", [0.25]),
                ),
                helper.make_node("Add", ["c", "filled"], ["y"]),
            ],
            [helper.make_tensor_value_info("shape", TensorProto.INT64, [1])],
            [helper.make_tensor_value_info("y", BF16, [2])],
        )

        converted = convert_bfloat16_to_float32_pass(model)

        assert _bf16_sites(converted) == []
        onnx.checker.check_model(converted, full_check=True)
        const, const_of_shape = converted.graph.node[:2]
        np.testing.assert_array_equal(
            numpy_helper.to_array(const.attribute[0].t), [1.5, -2.0]
        )
        np.testing.assert_array_equal(
            numpy_helper.to_array(const_of_shape.attribute[0].t), [0.25]
        )

    def test_qdq_scales(self):
        """
        Given: A Q/DQ pair on a bf16 tensor with a bf16 scale
        When: Convert
        Then: The scale becomes fp32 and the graph runs in ORT with the same result
        """
        model = _make_model(
            [
                helper.make_node("QuantizeLinear", ["x", "s", "zp"], ["q"]),
                helper.make_node("DequantizeLinear", ["q", "s", "zp"], ["y"]),
            ],
            [helper.make_tensor_value_info("x", BF16, [4])],
            [helper.make_tensor_value_info("y", BF16, [4])],
            [
                _bf16_tensor("s", 0.125),
                numpy_helper.from_array(np.array(0, dtype=np.int8), "zp"),
            ],
        )

        converted = convert_bfloat16_to_float32_pass(model)

        assert _bf16_sites(converted) == []
        onnx.checker.check_model(converted, full_check=True)
        x = np.array([0.1, -0.3, 1.0, 20.0], dtype=np.float32)
        (y,) = ort.InferenceSession(
            converted.SerializeToString(), providers=["CPUExecutionProvider"]
        ).run(None, {"x": x})
        np.testing.assert_array_equal(
            y, np.clip(np.round(x / 0.125), -128, 127) * 0.125
        )

    def test_qdq_dtype_attributes(self):
        """
        Given: Opset-23 Q/DQ whose output_dtype / precision attributes name bf16
        When: Convert
        Then: Those attributes name fp32
        """
        model = _make_model(
            [
                helper.make_node(
                    "QuantizeLinear", ["x", "s32", "zp"], ["q"], precision=BF16
                ),
                helper.make_node(
                    "DequantizeLinear", ["q", "s32", "zp"], ["y"], output_dtype=BF16
                ),
            ],
            [helper.make_tensor_value_info("x", FP32, [4])],
            [helper.make_tensor_value_info("y", BF16, [4])],
            [
                numpy_helper.from_array(np.array(0.125, dtype=np.float32), "s32"),
                numpy_helper.from_array(np.array(0, dtype=np.int8), "zp"),
            ],
            opset=23,
        )
        assert sorted(_bf16_sites(model)) == [
            "main:DequantizeLinear.output_dtype",
            "main:QuantizeLinear.precision",
            "main:y",
        ]

        converted = convert_bfloat16_to_float32_pass(model)

        # ORT 1.23 has no opset-23 Q/DQ kernels, so this is checked structurally only.
        assert _bf16_sites(converted) == []
        onnx.checker.check_model(converted, full_check=True)

    def test_if_subgraphs(self):
        """
        Given: An If node whose branches hold bf16 initializers, values and Casts
        When: Convert
        Then: The branches are converted too
        """

        def branch(name, value):
            return helper.make_graph(
                [helper.make_node("Cast", ["k"], ["out"], to=BF16)],
                name,
                [],
                [helper.make_tensor_value_info("out", BF16, [1])],
                [_bf16_tensor("k", [value])],
            )

        model = _make_model(
            [
                helper.make_node(
                    "If",
                    ["cond"],
                    ["y"],
                    then_branch=branch("then", 1.0),
                    else_branch=branch("else", -1.0),
                )
            ],
            [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
            [helper.make_tensor_value_info("y", BF16, [1])],
        )
        assert any("If.then_branch" in site for site in _bf16_sites(model))

        converted = convert_bfloat16_to_float32_pass(model)

        assert _bf16_sites(converted) == []
        onnx.checker.check_model(converted, full_check=True)
        session = ort.InferenceSession(
            converted.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        assert session.run(None, {"cond": np.array(True)})[0].tolist() == [1.0]
        assert session.run(None, {"cond": np.array(False)})[0].tolist() == [-1.0]

    def test_model_local_functions(self):
        """
        Given: A model-local function containing a Cast to bf16
        When: Convert
        Then: The function body is converted too
        """
        function = helper.make_function(
            "custom",
            "ToBf16",
            ["a"],
            ["b"],
            [helper.make_node("Cast", ["a"], ["b"], to=BF16)],
            [helper.make_opsetid("", 21)],
        )
        model = _make_model(
            [helper.make_node("ToBf16", ["x"], ["y"], domain="custom")],
            [helper.make_tensor_value_info("x", FP32, [2])],
            [helper.make_tensor_value_info("y", BF16, [2])],
            functions=[function],
        )
        assert any(site.startswith("function") for site in _bf16_sites(model))

        converted = convert_bfloat16_to_float32_pass(model)

        assert _bf16_sites(converted) == []
        onnx.checker.check_model(converted, full_check=True)

    def test_model_without_bfloat16_is_unchanged(self):
        model = _make_model(
            [helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT16)],
            [helper.make_tensor_value_info("x", FP32, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2])],
        )

        converted = convert_bfloat16_to_float32_pass(model)

        assert onnx.helper.printable_graph(
            converted.graph
        ) == onnx.helper.printable_graph(model.graph)

    @pytest.mark.cuda
    def test_bfloat16_qdq_model_from_quantsim(self):
        """
        Given: A bf16 QDQ model from to_onnx_qdq() of a calibrated bf16-I/O sim
        When: Convert
        Then: The result is a valid fp32 model with the same Q/DQ structure that ORT
              can run, giving the same result on CPU and CUDA
        """
        if not "CUDAExecutionProvider" in ort.get_available_providers():
            pytest.skip("CUDA EP not available")

        rng = np.random.default_rng(0)
        inputs = [
            rng.standard_normal((4, 16)).astype(ml_dtypes.bfloat16) for _ in range(4)
        ]
        sim = QuantizationSimModel(
            models_for_tests.simple_mlp_model(torch.bfloat16, io_dtype=torch.bfloat16),
            providers=CUDA_PROVIDERS,
            param_type=aimet_onnx.int8,
            activation_type=aimet_onnx.int8,
            quant_scheme=QuantScheme.min_max,
        )
        sim.compute_encodings([{"input": x} for x in inputs])
        bf16_qdq = sim.to_onnx_qdq(export_int32_bias=True)
        assert _bf16_sites(bf16_qdq)

        converted = convert_bfloat16_to_float32_pass(bf16_qdq)

        assert _bf16_sites(converted) == []
        onnx.checker.check_model(converted, full_check=True)
        assert sum(n.op_type == "QuantizeLinear" for n in converted.graph.node) == sum(
            n.op_type == "QuantizeLinear" for n in bf16_qdq.graph.node
        )
        x = inputs[0].astype(np.float32)
        outputs = [
            ort.InferenceSession(
                converted.SerializeToString(), providers=providers
            ).run(None, {"input": x})[0]
            for providers in (CUDA_PROVIDERS, ["CPUExecutionProvider"])
        ]
        assert outputs[0].dtype == np.float32
        # CPU and CUDA kernels round differently, which can move a value sitting on a
        # rounding boundary by one quantization step of the output.
        output_scale = (
            sim.qc_quantize_op_dict["/fc2/Gemm_output_0"].get_encodings()[0].delta
        )
        np.testing.assert_allclose(outputs[0], outputs[1], atol=output_scale * 1.01)
