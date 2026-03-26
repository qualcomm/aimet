# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ONNX graph fusions."""

import os
import pytest
from packaging import version
import numpy as np
import torch
import onnx
import onnx_ir
import onnxruntime
import onnxscript
from aimet_onnx.utils import make_dummy_input, get_node_attribute
from aimet_onnx import QuantizationSimModel

from aimet_onnx.graph_passes.fusions import fuse_supergroups, inline_all_supergroups
from aimet_onnx.graph_passes.fusions.fusion_registry import AIMET_SUPERGROUP_DOMAIN
from ..models import models_for_tests
from ..models.test_models import rmsnorm_model


def create_layernorm_model(
    tmpdir, elementwise_affine=True, bias=True, epsilon=1e-5, opset=16
):
    # Input shape: [batch_size, seq_len, hidden_size]
    input_shape = [1, 32, 64]
    hidden_size = input_shape[-1]

    class LayerNormModel(torch.nn.Module):
        def __init__(self):
            super(LayerNormModel, self).__init__()
            self.linear = torch.nn.Linear(64, 64)
            self.layernorm = torch.nn.LayerNorm(
                hidden_size,
                eps=epsilon,
                bias=bias,
                elementwise_affine=elementwise_affine,
            )
            self.linear2 = torch.nn.Linear(64, 64)

        def forward(self, x):
            x = self.linear(x)
            x = self.layernorm(x)
            return self.linear2(x)

    model = LayerNormModel()
    dummy_input = torch.randn(*input_shape)
    model_path = os.path.join(tmpdir, "layernorm.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    model = onnx.load(model_path)

    return model


def layernorm_with_pow_2_as_multiply(tmpdir):
    model = create_layernorm_model(tmpdir)
    ir_model = onnx_ir.from_proto(model)
    for node in ir_model.graph.all_nodes():
        if node.op_type == "Pow":
            node.op_type = "Mul"
            node.replace_input_with(1, node.inputs[0])
    onnx_ir.passes.common.RemoveUnusedNodesPass().call(ir_model)
    return onnx_ir.to_proto(ir_model)


def layernorm_with_pow_3(tmpdir):
    model = create_layernorm_model(tmpdir)
    ir_model = onnx_ir.from_proto(model)
    new_const = onnx_ir.val(
        "new_pow_const", const_value=onnx_ir.tensor(np.array(3.0, dtype=np.float32))
    )
    ir_model.graph.register_initializer(new_const)
    for node in ir_model.graph.all_nodes():
        if node.op_type == "Pow":
            node.replace_input_with(1, new_const)
    onnx_ir.passes.common.RemoveUnusedNodesPass().call(ir_model)
    return onnx_ir.to_proto(ir_model)


def layernorm_with_negative_epsilon(tempdir):
    model = create_layernorm_model(tempdir)
    ir_model = onnx_ir.from_proto(model)
    new_const = onnx_ir.val(
        "new_epsilon", const_value=onnx_ir.tensor(np.array(-1e-5, dtype=np.float32))
    )
    ir_model.graph.register_initializer(new_const)
    ir_model.graph.node("/layernorm/Add").replace_input_with(1, new_const)
    return onnx_ir.to_proto(ir_model)


def layernorm_with_no_reducemean_axis(tmpdir):
    model = create_layernorm_model(tmpdir)
    ir_model = onnx_ir.from_proto(model)
    reduce_means = [
        node for node in ir_model.graph.all_nodes() if node.op_type == "ReduceMean"
    ]
    for rm in reduce_means:
        rm.attributes.clear()
    return onnx_ir.to_proto(ir_model)


class TestLayerNormFusion:
    """Tests for LayerNormalization pattern fusion."""

    # TODO: Match layernorm without bias/affine transform
    @pytest.mark.parametrize("bias", [True])
    @pytest.mark.parametrize("affine", [True])
    @pytest.mark.parametrize("opset_version", range(13, 17))
    @pytest.mark.parametrize("epsilon", [1e-1, 1e-3, 1e-5])
    @pytest.mark.parametrize("include_rmsnorm_pattern", [True, False])
    def test_fuses_single_layernorm(
        self, tmp_path, opset_version, bias, affine, epsilon, include_rmsnorm_pattern
    ):
        layernorm_model = create_layernorm_model(
            tmp_path,
            elementwise_affine=affine,
            bias=bias,
            opset=opset_version,
            epsilon=epsilon,
        )
        model = onnx_ir.from_proto(layernorm_model)
        inputs = make_dummy_input(layernorm_model)

        session = onnxruntime.InferenceSession(layernorm_model.SerializeToString())
        output_pre_fusion = session.run(None, inputs)

        patterns = (
            ["LayerNormalization", "RMSNormalization"]
            if include_rmsnorm_pattern
            else ["LayerNormalization"]
        )
        fused_model = fuse_supergroups(model, patterns=patterns, verbose=True)

        model_proto = onnx_ir.to_proto(fused_model)
        layernorms = [
            node
            for node in model_proto.graph.node
            if node.op_type == "LayerNormalization"
        ]
        assert len(layernorms) == 1

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_post_fusion = session.run(None, inputs)

        assert np.allclose(output_pre_fusion[0], output_post_fusion[0], atol=1e-5)

    def test_fuses_layernorm_with_pow_as_multiply(self, tmp_path):
        layernorm_model = layernorm_with_pow_2_as_multiply(tmp_path)
        model = onnx_ir.from_proto(layernorm_model)
        inputs = make_dummy_input(layernorm_model)

        session = onnxruntime.InferenceSession(layernorm_model.SerializeToString())
        output_pre_fusion = session.run(None, inputs)

        fused_model = fuse_supergroups(model, patterns=["LayerNormalization"])

        model_proto = onnx_ir.to_proto(fused_model)
        layernorms = [
            node
            for node in model_proto.graph.node
            if node.op_type == "LayerNormalization"
        ]
        assert len(layernorms) == 1

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_post_fusion = session.run(None, inputs)

        assert np.allclose(output_pre_fusion[0], output_post_fusion[0], atol=1e-5)

    @pytest.mark.parametrize(
        "invalid_graph_factory",
        [
            layernorm_with_pow_3,
            layernorm_with_negative_epsilon,
            layernorm_with_no_reducemean_axis,
            lambda tmpdir: create_layernorm_model(tmpdir, elementwise_affine=False),
            lambda tmpdir: create_layernorm_model(tmpdir, bias=False),
            lambda tmpdir: create_layernorm_model(
                tmpdir, bias=False, elementwise_affine=False
            ),
        ],
    )
    def test_rejects_invalid_layernorm_patterns(self, tmp_path, invalid_graph_factory):
        """Test that invalid LayerNorm patterns are rejected during fusion."""
        model_proto = invalid_graph_factory(tmp_path)
        num_nodes_before = len(model_proto.graph.node)
        inputs = make_dummy_input(model_proto)
        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_pre_fusion = session.run(None, inputs)
        model = onnx_ir.from_proto(model_proto)

        # Attempt fusion
        fused_model = fuse_supergroups(model, patterns=["LayerNormalization"])
        model_proto = onnx_ir.to_proto(fused_model)
        assert len(model_proto.graph.node) == num_nodes_before

        # Ensure no LayerNormalization nodes were created
        layernorm_nodes = [
            node
            for node in model_proto.graph.node
            if node.op_type == "LayerNormalization"
        ]
        assert len(layernorm_nodes) == 0

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_post_fusion = session.run(None, inputs)
        assert np.allclose(output_pre_fusion[0], output_post_fusion[0], atol=1e-5)

    def test_multiple_layernorm_instances(self, tmp_path):
        """Test fusion with multiple LayerNorm instances in the model."""

        class DoubleLayerNormModel(torch.nn.Module):
            def __init__(self):
                super(DoubleLayerNormModel, self).__init__()
                self.ln1 = torch.nn.LayerNorm(128)
                self.ln2 = torch.nn.LayerNorm(128)

            def forward(self, x):
                x = self.ln1(x)
                x = self.ln2(x)
                return x

        model = DoubleLayerNormModel()
        dummy_input = torch.randn(1, 32, 128)
        model_path = os.path.join(tmp_path, "double_layernorm.onnx")
        torch.onnx.export(
            model, dummy_input, model_path, opset_version=13, dynamo=False
        )

        model_proto = onnx.load(model_path)
        inputs = make_dummy_input(model_proto)

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output = session.run(None, inputs)

        layernorm_nodes = [
            node
            for node in model_proto.graph.node
            if node.op_type == "LayerNormalization"
        ]
        assert len(layernorm_nodes) == 0

        model = onnx_ir.from_proto(model_proto)

        # Apply fusion
        fused_model = fuse_supergroups(model, patterns=["LayerNormalization"])
        model_proto = onnx_ir.to_proto(fused_model)

        # Both instances should potentially be fused
        layernorm_nodes = [
            node
            for node in model_proto.graph.node
            if node.op_type == "LayerNormalization"
        ]
        assert len(layernorm_nodes) == 2
        assert len(model_proto.functions) == 2
        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_post_fusion = session.run(None, inputs)
        assert np.allclose(output[0], output_post_fusion[0], atol=1e-5)

    def test_quantsim_with_fused_layernorm(self, tmp_path):
        """Test that QuantSim works with fused LayerNorm nodes."""
        # Get decomposed LayerNorm model
        layernorm_model = create_layernorm_model(tmp_path, opset=13)
        model = onnx_ir.from_proto(layernorm_model)

        # Apply fusion
        fused_model = fuse_supergroups(model, patterns=["LayerNormalization"])
        model_proto = onnx_ir.to_proto(fused_model)

        """
        When: Creating a QuantizationSimModel with the fused LayerNorm model
        Then: 1) LayerNorm weight is detected as a parameter
              2) LayerNorm weight is quantized with parameter type
              3) sim.compute_encodings runs without error
        """
        sim = QuantizationSimModel(
            model_proto,
            param_type="int8",
            activation_type="int16",
            config_file="htp_v73",
        )

        assert "layernorm.weight" in sim.param_names
        assert sim.qc_quantize_op_dict["layernorm.weight"].bitwidth == 8
        sim.compute_encodings([make_dummy_input(model_proto)])


def create_simple_linear_model(tmpdir):
    """Create a simple model with a single Linear layer (Matmul + Add)."""
    input_shape = [1, 32, 64]

    class LinearModel(torch.nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.linear = torch.nn.Linear(64, 128)

        def forward(self, x):
            return self.linear(x)

    model = LinearModel()
    dummy_input = torch.randn(*input_shape)
    model_path = os.path.join(tmpdir, "linear.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    return onnx.load(model_path)


def matmuladd_with_3d_weight(tmpdir):
    """Create a model with 3D weight (should not match)."""
    model_proto = create_simple_linear_model(tmpdir)

    # Find the weight tensor and modify it to be 3D
    for init in model_proto.graph.initializer:
        if "MatMul" in init.name:
            # Get the current weight data
            weight_array = np.frombuffer(init.raw_data, dtype=np.float32).reshape(
                init.dims
            )
            # Change shape from [in, out] to [1, in, out]
            new_array = np.expand_dims(weight_array, axis=0)
            # Update the initializer
            init.dims[:] = new_array.shape
            init.raw_data = new_array.tobytes()
            break

    return model_proto


def matmuladd_with_2d_bias(tmpdir):
    """Create a model with 2D bias (should not match)."""
    model_proto = create_simple_linear_model(tmpdir)

    # Find the bias tensor and modify it to be 2D
    for init in model_proto.graph.initializer:
        if "bias" in init.name:
            bias_array = np.frombuffer(init.raw_data, dtype=np.float32).reshape(
                init.dims
            )
            new_array = np.expand_dims(bias_array, axis=0)
            init.dims[:] = new_array.shape
            init.raw_data = new_array.tobytes()
            break

    return model_proto


class TestMatmulAddFusion:
    @pytest.mark.parametrize(
        "model_factory, expected_matches",
        [
            (lambda path: models_for_tests.matmul_bias_add_model(bias_first=True), 1),
            (lambda path: models_for_tests.matmul_bias_add_model(bias_first=False), 1),
            (lambda path: models_for_tests.matmul_add_model(), 0),  # Not a bias add
            (lambda path: models_for_tests.unfusable_matmul_add().model, 0),
            (
                lambda _: models_for_tests.matmul_add_with_transpose(
                    True, dynamic_weight=False
                ),
                1,
            ),
            (
                lambda _: models_for_tests.matmul_add_with_transpose(
                    False, dynamic_weight=False
                ),
                1,
            ),
            # Dynamic weight: should not be matched into Gemm
            (
                lambda _: models_for_tests.matmul_add_with_transpose(
                    True, dynamic_weight=True
                ),
                0,
            ),
            (
                lambda _: models_for_tests.matmul_add_with_transpose(
                    False, dynamic_weight=True
                ),
                0,
            ),
            (create_simple_linear_model, 1),
            (create_layernorm_model, 2),
            # Invalid patterns:
            (matmuladd_with_3d_weight, 0),
            (matmuladd_with_2d_bias, 0),
        ],
    )
    def test_fuses_matmul_add(self, tmp_path, model_factory, expected_matches):
        onnx_model = model_factory(tmp_path)

        model = onnx_ir.from_proto(onnx_model)
        inputs = make_dummy_input(onnx_model)

        session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        output_pre_fusion = session.run(None, inputs)

        fused_model = fuse_supergroups(model, patterns=["MatmulAdd"])

        model_proto = onnx_ir.to_proto(fused_model)
        supergroups = [
            node
            for node in model_proto.graph.node
            if node.op_type == "Gemm" and node.domain == "aimet.supergroup"
        ]
        assert len(supergroups) == expected_matches

        functions = {(f.domain, f.name, f.overload): f for f in model_proto.functions}
        # Verify transB attribute is properly set
        for supergroup in supergroups:
            func_key = supergroup.domain, supergroup.op_type, supergroup.overload
            assert func_key in functions
            function = functions[func_key]
            transposed = any(node.op_type == "Transpose" for node in function.node)
            assert transposed == bool(get_node_attribute(supergroup, "transB"))

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        output_post_fusion = session.run(None, inputs)

        assert np.allclose(output_pre_fusion[0], output_post_fusion[0], atol=1e-5)

    def test_transb_attribute_with_transpose(self):
        """Test that transB=1 when pattern has Transpose node."""
        onnx_model = models_for_tests.matmul_add_with_transpose()
        model = onnx_ir.from_proto(onnx_model)

        fused_model = fuse_supergroups(model, patterns=["MatmulAdd"])
        model_proto = onnx_ir.to_proto(fused_model)

        gemm_nodes = [
            node
            for node in model_proto.graph.node
            if node.op_type == "Gemm" and node.domain == "aimet.supergroup"
        ]
        assert len(gemm_nodes) == 1

        # Check transB attribute
        gemm_node = gemm_nodes[0]
        trans_b_attr = next(
            (attr for attr in gemm_node.attribute if attr.name == "transB"), None
        )
        assert trans_b_attr is not None, "transB attribute not found"
        assert trans_b_attr.i == 1, f"Expected transB=1, got transB={trans_b_attr.i}"

    def test_transb_attribute_without_transpose(self):
        """Test that transB=0 when pattern has no Transpose node."""
        onnx_model = models_for_tests.matmul_bias_add_model()
        model = onnx_ir.from_proto(onnx_model)

        fused_model = fuse_supergroups(model, patterns=["MatmulAdd"])
        model_proto = onnx_ir.to_proto(fused_model)

        gemm_nodes = [
            node
            for node in model_proto.graph.node
            if node.op_type == "Gemm" and node.domain == "aimet.supergroup"
        ]
        assert len(gemm_nodes) == 1

        # Check transB attribute
        gemm_node = gemm_nodes[0]
        trans_b_attr = next(
            (attr for attr in gemm_node.attribute if attr.name == "transB"), None
        )
        assert trans_b_attr is None or trans_b_attr.i == 0

    @pytest.mark.parametrize("trans_b", [True, False])
    def test_matmul_add_pattern_with_quantsim(self, trans_b):
        """Test that QuantSim works with fused MatmulAdd nodes."""
        if trans_b:
            onnx_model = models_for_tests.matmul_add_with_transpose()
        else:
            onnx_model = models_for_tests.matmul_bias_add_model()

        model = onnx_ir.from_proto(onnx_model)

        fused_model = fuse_supergroups(model, patterns=["MatmulAdd"])
        model_proto = onnx_ir.to_proto(fused_model)

        """
        When: Creating a QuantizationSimModel with the fused MatmulAdd model
        Then: 1) Weight and bias parameters are correctly identified
              2) Per-channel quantization is enabled for the layer
              3) Quantization axis is correctly set based on transB attribute
              4) sim.compute_encodings runs without error
        """
        sim = QuantizationSimModel(
            model_proto,
            param_type="int8",
            activation_type="int16",
            config_file="htp_v81",
        )
        assert len(sim.connected_graph.ordered_ops) == 1
        (matmul,) = sim.connected_graph.ordered_ops
        assert len(matmul.parameters) == 2

        # Check that weight and bias are correctly identified
        weight = next(
            (
                param
                for param, param_type in matmul.parameters.values()
                if param_type == "weight"
            ),
            None,
        )
        bias = next(
            (
                param
                for param, param_type in matmul.parameters.values()
                if param_type == "bias"
            ),
            None,
        )
        assert weight and bias

        # Check that weight quantizer is correctly configured based on transB attribute
        weight_quantizer = sim.qc_quantize_op_dict[weight.name]
        assert weight_quantizer.quant_info.usePerChannelMode
        assert weight_quantizer.quant_info.channelAxis == 0 if trans_b else 1

        sim.compute_encodings([make_dummy_input(model_proto)])

        # Check that bias is properly concretized
        with sim._concretize_int32_bias_quantizers():
            assert bias.name in sim.qc_quantize_op_dict
            assert sim.qc_quantize_op_dict[bias.name].enabled
            assert sim.qc_quantize_op_dict[bias.name].bitwidth == 32


class TestRMSNormFusion:
    @pytest.mark.parametrize("opset", [13, 17, 22])
    @pytest.mark.parametrize("elementwise_affine", [True, False])
    @pytest.mark.parametrize("mul_for_pow", [True, False])
    @pytest.mark.parametrize(
        "mul_rsqrt_pattern", ["mul_rsqrt", "div_sqrt", "mul_reciprocal_sqrt"]
    )
    def test_rmsnorm_fusion_variants(
        self, elementwise_affine, mul_for_pow, mul_rsqrt_pattern, opset
    ):
        dim = 32
        model = rmsnorm_model(
            dim=dim,
            elementwise_affine=elementwise_affine,
            mul_for_pow=mul_for_pow,
            mul_rsqrt_pattern=mul_rsqrt_pattern,
            opset=opset,
        )
        dummy_input = make_dummy_input(model)
        session = onnxruntime.InferenceSession(model.SerializeToString())
        original_output = session.run(None, dummy_input)[0]

        ir_model = onnx_ir.from_proto(model)
        fused_model = fuse_supergroups(
            ir_model, patterns=["RMSNormalization"], verbose=True
        )

        model_proto = onnx_ir.to_proto(fused_model)

        op_name = "RMSNormalization"
        rmsnorm_nodes = [
            node for node in model_proto.graph.node if node.op_type == op_name
        ]
        num_inputs = 2 if elementwise_affine else 1
        assert len(rmsnorm_nodes) == 1
        assert all(len(node.input) == num_inputs for node in rmsnorm_nodes)

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        fused_output = session.run(None, dummy_input)[0]
        assert np.allclose(original_output, fused_output)

    @pytest.mark.skip_on_windows_amd64(
        "torch.onnx.export fails for rmsnorm_model on Windows AMD64"
    )
    @pytest.mark.skipif(
        version.parse(onnxscript.__version__) < version.parse("0.4.0"),
        reason="Requires onnxscript >= 0.4.0",
    )
    @pytest.mark.parametrize("elementwise_affine", [True, False])
    @pytest.mark.parametrize("eps", [None, 1e-3])
    @pytest.mark.parametrize("opset", [13, 17, 22])
    def test_torch_rmsnorm(self, tmp_path, elementwise_affine, eps, opset):
        model = models_for_tests.rmsnorm_model(
            tmp_path, eps=eps, elementwise_affine=elementwise_affine, opset=opset
        )
        dummy_input = make_dummy_input(model)
        session = onnxruntime.InferenceSession(model.SerializeToString())
        original_output = session.run(None, dummy_input)[0]

        ir_model = onnx_ir.from_proto(model)
        fused_model = fuse_supergroups(
            ir_model, patterns=["RMSNormalization"], verbose=True
        )

        model_proto = onnx_ir.to_proto(fused_model)

        op_name = "RMSNormalization"
        rmsnorm_nodes = [
            node for node in model_proto.graph.node if node.op_type == op_name
        ]
        assert len(rmsnorm_nodes) == 1
        num_inputs = 2 if elementwise_affine else 1
        assert all(len(node.input) == num_inputs for node in rmsnorm_nodes)

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        fused_output = session.run(None, dummy_input)[0]
        assert np.allclose(original_output, fused_output)

    @pytest.mark.skip_on_windows_amd64(
        "torch.onnx.export fails for llama_rmsnorm_model on Windows AMD64"
    )
    @pytest.mark.skipif(
        version.parse(onnxscript.__version__) < version.parse("0.4.0"),
        reason="Requires onnxscript >= 0.4.0",
    )
    @pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
    @pytest.mark.parametrize(
        "model_factory, expected_matches",
        [
            (lambda path: models_for_tests.llama_rmsnorm_model(path, opset=16), 1),
            (lambda path: models_for_tests.llama_rmsnorm_model(path, opset=22), 1),
            (create_layernorm_model, 1),  # Layernorm contains RMSNorm internally
            # Invalid patterns - should not match
            (models_for_tests.rmsnorm_invalid_multiple_axes, 0),
            (models_for_tests.rmsnorm_invalid_negative_epsilon, 0),
            (models_for_tests.rmsnorm_invalid_wrong_power, 0),
            (models_for_tests.rmsnorm_invalid_intermediate_output, 0),
        ],
    )
    def test_rmsnorm_fusion(self, tmp_path, model_factory, expected_matches):
        model = model_factory(tmp_path)
        dummy_input = make_dummy_input(model)
        session = onnxruntime.InferenceSession(model.SerializeToString())
        original_output = session.run(None, dummy_input)[0]

        ir_model = onnx_ir.from_proto(model)
        fused_model = fuse_supergroups(
            ir_model, patterns=["RMSNormalization"], verbose=True
        )

        model_proto = onnx_ir.to_proto(fused_model)
        rmsnorm_nodes = [
            node
            for node in model_proto.graph.node
            if node.op_type == "RMSNormalization"
        ]
        assert len(rmsnorm_nodes) == expected_matches

        session = onnxruntime.InferenceSession(model_proto.SerializeToString())
        fused_output = session.run(None, dummy_input)[0]
        assert np.allclose(original_output, fused_output)


class TestFusion:
    def test_unknown_pattern_raises_error(self, tmp_path):
        """Test that unknown pattern names raise ValueError."""
        model_proto = create_layernorm_model(tmp_path)
        model = onnx_ir.from_proto(model_proto)

        with pytest.raises(ValueError, match="Unknown pattern names"):
            fuse_supergroups(model, patterns=["UnknownPattern"])

    @pytest.mark.parametrize(
        "providers",
        [["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]],
    )
    @pytest.mark.parametrize(
        "model_factory",
        [
            lambda path: create_layernorm_model(path),
            lambda path: models_for_tests.single_residual_model().model,
            lambda path: models_for_tests.squeezenet1_0(path).model,
            lambda path: models_for_tests.simple_relu_model().model,
            lambda path: models_for_tests.standalone_layernorm([1, 32, 64]),
        ],
    )
    def test_fusion_does_not_impact_accuracy(self, tmp_path, model_factory, providers):
        """Test that fusion runs without errors on a variety of models."""
        model_proto = model_factory(tmp_path)

        dummy_input = make_dummy_input(model_proto)
        session = onnxruntime.InferenceSession(
            model_proto.SerializeToString(), providers=providers
        )
        output_pre_fusion = session.run(None, dummy_input)

        # Apply fusion
        model = onnx_ir.from_proto(model_proto)
        fused_model = fuse_supergroups(
            model,
            patterns=["LayerNormalization", "MatmulAdd", "RMSNormalization"],
            verbose=True,
        )

        # Ensure the fused model can be converted back to proto
        model_proto = onnx_ir.to_proto(fused_model)
        session = onnxruntime.InferenceSession(
            model_proto.SerializeToString(), providers=providers
        )
        output_post_fusion = session.run(None, dummy_input)

        assert np.allclose(output_pre_fusion[0], output_post_fusion[0], atol=1e-5)


class TestInlineAllSupergroups:
    """Tests for inline_all_supergroups: unfusing supergroup functions back to primitives."""

    @pytest.mark.skip_on_windows_amd64(
        "torch.onnx.export fails for llama_rmsnorm_model on Windows AMD64"
    )
    @pytest.mark.skipif(
        version.parse(onnxscript.__version__) < version.parse("0.4.0"),
        reason="Requires onnxscript >= 0.4.0",
    )
    @pytest.mark.skip_on_windows_arm64("transformers is not available on Windows ARM64")
    @pytest.mark.parametrize(
        "model_factory",
        [
            create_layernorm_model,
            create_simple_linear_model,
            lambda p: rmsnorm_model(
                dim=32,
                elementwise_affine=True,
                mul_for_pow=False,
                mul_rsqrt_pattern="div_sqrt",
                opset=17,
            ),
            lambda p: rmsnorm_model(
                dim=32,
                elementwise_affine=False,
                mul_for_pow=False,
                mul_rsqrt_pattern="div_sqrt",
                opset=17,
            ),
            lambda path: models_for_tests.single_residual_model().model,
            lambda path: models_for_tests.squeezenet1_0(path).model,
            lambda path: models_for_tests.simple_relu_model().model,
            lambda path: models_for_tests.standalone_layernorm([1, 32, 64]),
            models_for_tests.llama_rmsnorm_model,
        ],
    )
    def test_fuse_then_inline_restores_original_names(self, tmp_path, model_factory):
        """Node/value names before fusion match names after fuse -> inline round-trip.

        Constant nodes may be renamed by the onnxscript rewriter during fusion,
        so those are excluded from the comparison.
        """
        fusion_patterns = ["LayerNormalization", "RMSNormalization", "MatmulAdd"]
        model_proto = model_factory(tmp_path)
        ir_model = onnx_ir.from_proto(model_proto)

        # Snapshot original names (ignoring constants which may be renamed by the rewriter)
        orig_node_names = set(
            node.name
            for node in ir_model.graph.all_nodes()
            if not node.op_type == "Constant"
        )
        orig_value_names = set(
            v.name
            for node in ir_model.graph.all_nodes()
            if node.op_type != "Constant"
            for v in node.outputs
        )

        fuse_supergroups(ir_model, patterns=fusion_patterns)
        inline_all_supergroups(ir_model)

        # No supergroup functions remain
        assert not any(
            func.domain == AIMET_SUPERGROUP_DOMAIN
            for func in ir_model.functions.values()
        )

        assert not any(
            node.domain == AIMET_SUPERGROUP_DOMAIN
            for node in ir_model.graph.all_nodes()
        )

        # Original non-Constant names are fully restored
        final_node_names = [
            n.name for n in ir_model.graph.all_nodes() if not n.op_type == "Constant"
        ]
        final_value_names = [
            v.name
            for n in ir_model.graph.all_nodes()
            if n.op_type != "Constant"
            for v in n.outputs
        ]

        assert set(final_node_names) == orig_node_names
        assert set(final_value_names) == orig_value_names
        assert len(set(final_node_names)) == len(
            final_node_names
        )  # No duplicate node names
        assert len(set(final_value_names)) == len(
            final_value_names
        )  # No duplicate value names

        # Ensure the inlined model produces the same outputs as the original
        inputs = make_dummy_input(model_proto)
        expected = onnxruntime.InferenceSession(model_proto.SerializeToString()).run(
            None, inputs
        )
        actual = onnxruntime.InferenceSession(
            onnx_ir.to_proto(ir_model).SerializeToString()
        ).run(None, inputs)
        assert np.allclose(expected[0], actual[0], atol=1e-5)
