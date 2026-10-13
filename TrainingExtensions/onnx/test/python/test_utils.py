# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import gc
import os
import tempfile

import onnx
import onnxruntime
import torch
from packaging import version
import pytest

import copy

import numpy as np
from onnx import numpy_helper, helper, TensorProto
from onnxruntime.quantization.onnx_quantizer import ONNXModel
from aimet_onnx.meta.connectedgraph import ConnectedGraph
from aimet_onnx.utils import (
    find_shared_param_names,
    duplicate_shared_initializers,
    OrtInferenceSession,
)
from aimet_onnx import qtype
import aimet_onnx.utils as utils
from aimet_onnx.utils import ParamUtils, disable_quantizers
from aimet_onnx.adaround.utils import ModelData
from aimet_onnx.quantsim import QuantizationSimModel

from .models import models_for_tests


class TestUtils:
    """
    Test functions in utils
    """

    def test_remove_nodes(self):
        """
        Test remove nodes by given type
        """
        model = models_for_tests.build_dummy_model()
        node_ls = [node.op_type for node in model.graph.node]
        assert node_ls == ["Conv", "Relu", "MaxPool", "Flatten", "Gemm"]
        # Remove first layer of dummy model
        utils.remove_nodes_with_type("Conv", model.graph)
        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ["Relu", "MaxPool", "Flatten", "Gemm"]
        # Remove last layer of dummy model
        utils.remove_nodes_with_type("Gemm", model.graph)
        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ["Relu", "MaxPool", "Flatten"]
        # Check connection of each layer
        onnx.checker.check_model(model)

    def test_replace_nodes(self):
        """
        Test replace op type of nodes with given op type
        """
        model = models_for_tests.build_dummy_model()
        node_ls = [node.op_type for node in model.graph.node]
        assert node_ls == ["Conv", "Relu", "MaxPool", "Flatten", "Gemm"]

        utils.replace_node_with_op("Conv", "CustomOp", model.graph)
        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ["CustomOp", "Relu", "MaxPool", "Flatten", "Gemm"]

    def test_get_weights(self):
        """
        Test get weights
        """
        model = models_for_tests.build_dummy_model()
        for node in model.graph.initializer:
            assert node.raw_data == utils.get_weights(node.name, model.graph)

    def test_list_nodes(self):
        """
        Test get nodes with ordered
        """
        model = models_for_tests.build_dummy_model()
        node_dict = utils.get_ordered_dict_of_nodes(model.graph)
        node_keys = list(node_dict.keys())

        for i, node in enumerate(model.graph.node):
            assert node_keys[i] == node.name
            assert node_dict[node.name] == node

    def test_weight_utils(self):
        model = models_for_tests.build_dummy_model()
        for node in model.graph.node:
            if node.op_type == "Conv":
                weights = ParamUtils.get_param(model, node, 1)
                weights_shape = ParamUtils.get_shape(model, node, 1)
                bias = ParamUtils.get_param(model, node, 2)
                bias_shape = ParamUtils.get_shape(model, node, 2)
                assert bias_shape == [1]
                assert weights_shape == [1, 3, 3, 3]
                assert weights.name == "conv_w"
                assert bias.name == "conv_b"

            if node.op_type == "Gemm":
                weights = ParamUtils.get_param(model, node, 1)
                weights_shape = ParamUtils.get_shape(model, node, 1)
                bias = ParamUtils.get_param(model, node, 2)
                bias_shape = ParamUtils.get_shape(model, node, 2)
                assert bias_shape == [10]
                assert weights_shape == [256, 10]
                assert weights.name == "fc_w"
                assert bias.name == "fc_b"

    def test_utils_transposed_conv_model(self):
        model = models_for_tests.transposed_conv_model()
        model = model.model
        for node in model.graph.node:
            if node.op_type == "ConvTranspose":
                weights = ParamUtils.get_param(model, node, 1)
                weights_shape = ParamUtils.get_shape(model, node, 1)
                bias = ParamUtils.get_param(model, node, 2)
                bias_shape = ParamUtils.get_shape(model, node, 2)
                assert bias_shape == [10]
                assert weights_shape == [10, 10, 3, 3]
                assert weights.name == "conv1.weight"
                assert bias.name == "conv1.bias"
                break

    def test_utils_const_param_model(self):
        model = models_for_tests.const_param_model()
        for node in model.graph.node:
            if node.op_type == "InstanceNormalization":
                weights = ParamUtils.get_param(model, node, 1)
                weights_shape = ParamUtils.get_shape(model, node, 1)
                bias = ParamUtils.get_param(model, node, 2)
                bias_shape = ParamUtils.get_shape(model, node, 2)
                assert bias_shape == [32]
                assert weights_shape == [32]
                assert (
                    weights.name == "/down_blocks.0/resnets.0/norm1/Constant_1_output_0"
                )
                assert bias.name == "/down_blocks.0/resnets.0/norm1/Constant_2_output_0"
                break

    def test_remove_node(self):
        """
        Test remove node from model
        """
        model = models_for_tests.build_dummy_model()
        node_ls = [node.op_type for node in model.graph.node]
        assert node_ls == ["Conv", "Relu", "MaxPool", "Flatten", "Gemm"]
        gemm_node = model.graph.node[-1]
        utils.remove_node(gemm_node, model.graph)

        new_node_ls = [node.op_type for node in model.graph.node]
        assert new_node_ls == ["Conv", "Relu", "MaxPool", "Flatten"]
        assert model.graph.output[0].name in model.graph.node[-1].output

    def test_remove_node_for_initializer_pruning(self):
        """
        Verify initializers are completely deleted from the model if they are no longer used
        """
        model = models_for_tests.model_with_initializers_in_graph_input()
        bn_node = model.graph.node[1]
        utils.remove_node(bn_node, model.graph)

        model_input = [inp.name for inp in model.graph.input]
        model_init = [init.name for init in model.graph.initializer]

        # Initializers should be removed from both the lists- inputs and initializers.
        # Missing this will lead to failure in ORT Inference Session creation.
        assert all([not inp.startswith("bn_") for inp in model_input])
        assert all([not init.startswith("bn_") for init in model_init])

    def test_get_attribute(self):
        """
        Test get attribute value from node
        """
        model = models_for_tests.build_dummy_model()
        conv_layer = model.graph.node[0]
        assert utils.get_node_attribute(conv_layer, "pads") == [1, 1, 1, 1]
        assert utils.get_node_attribute(conv_layer, "kernel_shape") == [3, 3]

    def test_replace_relu6_with_relu(self):
        if version.parse(torch.__version__) >= version.parse("1.13"):
            model = models_for_tests.depthwise_conv_model_with_relu6()
            relu6_count = 0
            original_relu_count = 0
            for node in model.model.graph.node:
                if node.op_type == "Clip":
                    relu6_count += 1
                if node.op_type == "Relu":
                    original_relu_count += 1

            utils.replace_relu6_with_relu(model)

            relu_count = 0
            for node in model.model.graph.node:
                if node.op_type == "Relu":
                    relu_count += 1

            assert relu_count - original_relu_count == relu6_count

    def test_create_model_data_single_residual_model(self):
        model = models_for_tests.transposed_conv_model_without_bn()
        sim = QuantizationSimModel(model)
        model_data = ModelData(sim)
        assert len(model_data.module_to_info) == 3

    def test_disable_quantizers(self):
        model = models_for_tests.single_residual_model().model
        sim = QuantizationSimModel(model)
        enabled_quantizers = set(
            name
            for name, quantizer in sim.qc_quantize_op_dict.items()
            if quantizer.enabled
        )

        with disable_quantizers(sim, set(sim.param_names)):
            for name in sim.param_names:
                assert not sim.qc_quantize_op_dict[name].enabled

            for name in enabled_quantizers - set(sim.param_names):
                assert sim.qc_quantize_op_dict[name].enabled

        for name in enabled_quantizers:
            assert sim.qc_quantize_op_dict[name].enabled

        with disable_quantizers(sim, set(sim.activation_names)):
            for name in sim.activation_names:
                assert not sim.qc_quantize_op_dict[name].enabled

            for name in enabled_quantizers - set(sim.activation_names):
                assert sim.qc_quantize_op_dict[name].enabled

        for name in enabled_quantizers:
            assert sim.qc_quantize_op_dict[name].enabled

        with pytest.raises(RuntimeError):
            with disable_quantizers(sim, {"nonexistant_quantizer"}):
                pass

    @pytest.mark.skip("Upgrade to ONNX==1.19.0 required to enable this test")
    def test_custom_opset_version_upgrade(self):
        model = models_for_tests.build_dummy_model()

        from aimet_onnx.common.onnx._utils import _convert_version_with_external_weights

        upgraded_model = _convert_version_with_external_weights(model, 21)

        assert model.opset_import[0].version == 13
        assert upgraded_model.opset_import[0].version == 21
        onnx.checker.check_model(upgraded_model)

    def test_make_psnr_eval_fn_float16(self):
        """
        output_indices=None should pick up float16 outputs
        and compute_psnr should accept fp16 arrays without raising.
        """
        model = models_for_tests.single_residual_model(dtype=torch.float16).model
        fp_session = onnxruntime.InferenceSession(model.SerializeToString())
        inputs = [utils.make_dummy_input(model)]
        psnr_eval_fn = utils.make_psnr_eval_fn(fp_session, inputs, output_indices=None)
        assert psnr_eval_fn(fp_session) > 0

    def test_make_psnr_eval_fn_no_float_outputs(self):
        model = models_for_tests.gather_op_with_int_data_model()
        fp_session = onnxruntime.InferenceSession(model.SerializeToString())
        inputs = [{"model_input": np.asarray([[0, 1, 2, 3]], dtype=np.int64)}]

        with (
            pytest.warns(),
            pytest.raises(RuntimeError, match="Provide output_indices"),
        ):
            utils.make_psnr_eval_fn(fp_session, inputs)

        psnr_eval_fn = utils.make_psnr_eval_fn(fp_session, inputs, output_indices=0)
        assert psnr_eval_fn(fp_session) > 0

    def test_contains_tensor_type(self):
        model = models_for_tests.diverse_ops()
        assert not utils.contains_tensor_type(model, onnx.TensorProto.BFLOAT16)
        assert not utils.contains_tensor_type(model, onnx.TensorProto.FLOAT16)
        assert utils.contains_tensor_type(model, onnx.TensorProto.FLOAT)

        model = models_for_tests.diverse_ops(onnx.TensorProto.FLOAT16)
        assert not utils.contains_tensor_type(model, onnx.TensorProto.FLOAT)
        assert utils.contains_tensor_type(model, onnx.TensorProto.FLOAT16)

        model = models_for_tests.single_residual_model(dtype=torch.float32).model
        assert not utils.contains_tensor_type(model, onnx.TensorProto.BFLOAT16)
        assert not utils.contains_tensor_type(model, onnx.TensorProto.FLOAT16)
        assert utils.contains_tensor_type(model, onnx.TensorProto.FLOAT)

        model = models_for_tests.single_residual_model(dtype=torch.float16).model
        assert not utils.contains_tensor_type(model, onnx.TensorProto.BFLOAT16)
        assert utils.contains_tensor_type(model, onnx.TensorProto.FLOAT16)
        assert not utils.contains_tensor_type(model, onnx.TensorProto.FLOAT)

        model = models_for_tests.model_with_cast(onnx.TensorProto.BFLOAT16)
        assert utils.contains_tensor_type(model, onnx.TensorProto.BFLOAT16)
        assert utils.contains_tensor_type(model, onnx.TensorProto.FLOAT)
        assert not utils.contains_tensor_type(model, onnx.TensorProto.FLOAT16)

    @pytest.mark.parametrize("symmetric", [True, False])
    @pytest.mark.parametrize("bitwidth", [4, 8, 16])
    def test_tf_encoding_conversion(self, symmetric, bitwidth):
        scales = np.array([0.1, 0.5, 10.5, 20.5], np.float32).reshape(2, 2)
        if symmetric:
            offsets = np.full_like(scales, -(2 ** (bitwidth - 1)))
        else:
            offsets = np.random.randint(
                -(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1) - 1, scales.shape
            ).astype(np.float32)

        tf_encodings = utils.numpy_to_TfEncoding(scales, offsets, qtype.int(bitwidth))

        num_steps = 2**bitwidth - 1
        assert len(tf_encodings) == scales.size
        for enc, scale, offset in zip(
            tf_encodings, scales.flatten(), offsets.flatten()
        ):
            assert enc.bw == bitwidth
            assert enc.delta == scale
            assert enc.offset == offset
            assert np.allclose(enc.min, scale * offset)
            assert np.allclose(enc.max, (num_steps + offset) * scale)

        # Round-trip back to scale/offset arrays preserves the original values
        recovered_scales, recovered_offsets = utils.numpy_from_TfEncoding(
            tf_encodings, scales.shape
        )
        assert np.array_equal(recovered_scales, scales)
        assert np.array_equal(recovered_offsets, offsets)

    def test_to_tf_encoding_rejects_non_int_dtype(self):
        scales = np.array([0.1, 0.5], np.float32)
        offsets = np.zeros_like(scales)
        with pytest.raises(NotImplementedError):
            utils.numpy_to_TfEncoding(
                scales, offsets, qtype.float(exponent_bits=5, mantissa_bits=10)
            )


class TestDuplicateSharedInitializers:
    """
    Tests for utils.duplicate_shared_initializers and its interaction with find_shared_param_names.
    """

    @staticmethod
    def _make_shared_weight_model() -> onnx.ModelProto:
        """
        Build a minimal two-Conv model whose weight initializer is shared:

            input --> Conv(weight) --> Conv(weight) --> output

        Both Conv nodes reference the same initializer name, so the graph
        contains one shared tensor before duplication.
        """
        CHANNELS = 4
        weight = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="shared.weight",
        )
        nodes = [
            helper.make_node(
                "Conv",
                inputs=["model_input", weight.name],
                outputs=["conv_1.output"],
                name="conv_1",
            ),
            helper.make_node(
                "Conv",
                inputs=["conv_1.output", weight.name],
                outputs=["model_output"],
                name="conv_2",
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "SharedWeightModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            initializer=[weight],
        )
        model = models_for_tests.make_model(graph)
        onnx.checker.check_model(model)
        return model

    def test_duplicate_shared_initializers_output_equivalence(self):
        """
        After calling duplicate_shared_initializers the model must produce identical
        outputs to the original model for the same input.
        """
        original = self._make_shared_weight_model()
        duplicated = copy.deepcopy(original)

        n_duplicates = duplicate_shared_initializers(duplicated.graph)
        assert n_duplicates > 0

        dummy_input = utils.make_dummy_input(original)
        providers = ["CPUExecutionProvider"]

        original_output = OrtInferenceSession(original, providers).run(
            None, dummy_input
        )
        duplicated_output = OrtInferenceSession(duplicated, providers).run(
            None, dummy_input
        )

        assert len(original_output) == len(duplicated_output)
        for orig, dup in zip(original_output, duplicated_output):
            np.testing.assert_array_equal(orig, dup)

    def test_duplicate_shared_initializers_copy_is_independent(self):
        """
        The duplicated initializer added via graph.initializer.add().CopyFrom()
        must be a real, independent copy.
        """
        original = self._make_shared_weight_model()
        duplicated = copy.deepcopy(original)

        n_duplicates = duplicate_shared_initializers(duplicated.graph)
        assert n_duplicates == 1

        graph = duplicated.graph
        orig_init = next(
            init for init in graph.initializer if init.name == "shared.weight"
        )
        dup_init = next(
            init for init in graph.initializer if init.name != "shared.weight"
        )

        orig_array_before = numpy_helper.to_array(orig_init).copy()
        assert np.array_equal(numpy_helper.to_array(dup_init), orig_array_before)

        # Mutate the original in place; the duplicate must be unaffected.
        orig_init.raw_data = numpy_helper.from_array(orig_array_before + 1).raw_data
        assert np.array_equal(numpy_helper.to_array(dup_init), orig_array_before)

    def test_duplicate_shared_initializers_no_shared_params_after_duplication(self):
        """
        After calling duplicate_shared_initializers, find_shared_param_names must
        return an empty set — no initializer should be referenced by more than
        one node.
        """
        model = ONNXModel(self._make_shared_weight_model())
        conn_graph = ConnectedGraph(model)

        # Confirm the model has shared params before duplication
        assert find_shared_param_names(conn_graph), (
            "Expected shared params in the original model"
        )

        duplicate_shared_initializers(model.model.graph)

        # Re-build connected graph on the modified model
        conn_graph_after = ConnectedGraph(model)
        assert not find_shared_param_names(conn_graph_after), (
            "Expected no shared params after duplicate_shared_initializers"
        )

    @pytest.mark.parametrize("shared_conv_weight", [True, False])
    @pytest.mark.parametrize("shared_bn_weight", [True, False])
    @pytest.mark.parametrize("shared_stat", [True, False])
    def test_duplicate_shared_initializers_with_batchnorm_model(
        self, shared_conv_weight, shared_bn_weight, shared_stat
    ):
        """
        Verify output equivalence and absence of shared params after duplication
        on the richer shared-tensor BN model used in test_bn_fold.
        """
        original = models_for_tests.shared_tensor_batchnorm_model_with_identities(
            shared_conv_weight=shared_conv_weight,
            shared_bn_weight=shared_bn_weight,
            shared_stat=shared_stat,
        )
        duplicated = copy.deepcopy(original)
        duplicate_shared_initializers(duplicated.graph)

        dummy_input = utils.make_dummy_input(original)
        providers = ["CPUExecutionProvider"]

        original_output = OrtInferenceSession(original, providers).run(
            None, dummy_input
        )
        duplicated_output = OrtInferenceSession(duplicated, providers).run(
            None, dummy_input
        )

        assert len(original_output) == len(duplicated_output)
        for orig, dup in zip(original_output, duplicated_output):
            np.testing.assert_array_equal(orig, dup)

        conn_graph_after = ConnectedGraph(ONNXModel(duplicated))
        assert not find_shared_param_names(conn_graph_after), (
            "Expected no shared params after duplicate_shared_initializers"
        )

    def test_duplicate_shared_bias(self, tmp_path):
        """
        Shared bias (not weight) between two Conv nodes must also be duplicated.
        Uses conv_model_with_shared_bias from models_for_tests.
        """
        original = models_for_tests.conv_model_with_shared_bias(tmp_path)
        duplicated = copy.deepcopy(original)

        n_duplicates = duplicate_shared_initializers(duplicated.graph)
        assert n_duplicates > 0

        dummy_input = utils.make_dummy_input(original)
        providers = ["CPUExecutionProvider"]

        original_output = OrtInferenceSession(original, providers).run(
            None, dummy_input
        )
        duplicated_output = OrtInferenceSession(duplicated, providers).run(
            None, dummy_input
        )

        assert len(original_output) == len(duplicated_output)
        for orig, dup in zip(original_output, duplicated_output):
            np.testing.assert_array_equal(orig, dup)

    def test_duplicate_shared_weight_and_bias_parallel_branches(self):
        """
        ParallelConvSharedWeights has two Conv nodes that share both weight and
        bias.  After duplication both tensors must be independent and the model
        output must be unchanged.
        """
        torch_model = models_for_tests.ParallelConvSharedWeights()
        original = models_for_tests._convert_to_onnx_no_fold(
            torch_model, torch.randn(2, 10, 24, 24)
        ).model
        duplicated = copy.deepcopy(original)

        n_duplicates = duplicate_shared_initializers(duplicated.graph)
        assert n_duplicates > 0

        dummy_input = utils.make_dummy_input(original)
        providers = ["CPUExecutionProvider"]

        original_output = OrtInferenceSession(original, providers).run(
            None, dummy_input
        )
        duplicated_output = OrtInferenceSession(duplicated, providers).run(
            None, dummy_input
        )

        assert len(original_output) == len(duplicated_output)
        for orig, dup in zip(original_output, duplicated_output):
            np.testing.assert_array_equal(orig, dup)

        conn_graph_after = ConnectedGraph(ONNXModel(duplicated))
        assert not find_shared_param_names(conn_graph_after)

    def test_duplicate_tensor_shared_by_three_nodes(self):
        """
        A single initializer referenced by three nodes must produce two copies
        (one per extra usage) and the model output must remain identical.
        """
        CHANNELS = 4
        weight = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="triple.weight",
        )
        nodes = [
            helper.make_node(
                "Conv", inputs=["input", weight.name], outputs=["out1"], name="conv_1"
            ),
            helper.make_node(
                "Conv", inputs=["out1", weight.name], outputs=["out2"], name="conv_2"
            ),
            helper.make_node(
                "Conv", inputs=["out2", weight.name], outputs=["output"], name="conv_3"
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "TripleSharedWeight",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            initializer=[weight],
        )
        original = models_for_tests.make_model(graph)
        onnx.checker.check_model(original)
        duplicated = copy.deepcopy(original)

        n_duplicates = duplicate_shared_initializers(duplicated.graph)
        assert n_duplicates == 2  # two extra copies needed for 3 usages

        # All three nodes must now reference distinct initializer names
        init_names = {init.name for init in duplicated.graph.initializer}
        for node in duplicated.graph.node:
            assert node.input[1] in init_names
        node_weights = [node.input[1] for node in duplicated.graph.node]
        assert len(set(node_weights)) == 3, "Each node must own a unique weight copy"

        dummy_input = utils.make_dummy_input(original)
        providers = ["CPUExecutionProvider"]
        original_output = OrtInferenceSession(original, providers).run(
            None, dummy_input
        )
        duplicated_output = OrtInferenceSession(duplicated, providers).run(
            None, dummy_input
        )
        for orig, dup in zip(original_output, duplicated_output):
            np.testing.assert_array_equal(orig, dup)

    def test_duplicate_shared_initializers_is_idempotent_on_non_shared_model(self):
        """
        Calling duplicate_shared_initializers on a model with no shared initializers
        must return 0 and leave the model unchanged.
        """
        original = models_for_tests.build_dummy_model()
        n_init_before = len(original.graph.initializer)

        n_duplicates = duplicate_shared_initializers(original.graph)

        assert n_duplicates == 0
        assert len(original.graph.initializer) == n_init_before

    def test_duplicate_shared_initializers_empty_graph(self):
        """
        duplicate_shared_initializers must return 0 and not raise on an empty graph
        (no nodes, no initializers).
        """
        graph = helper.make_graph(
            [],
            "EmptyGraph",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            outputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
            initializer=[],
        )
        model = models_for_tests.make_model(graph)
        assert duplicate_shared_initializers(model.graph) == 0

    def test_duplicate_shared_initializers_name_collision(self):
        """
        If a copy name (e.g. 'w_copy_1') already exists as an initializer,
        the function must still produce a valid model with unique names and
        correct outputs.
        """
        CHANNELS = 4
        weight = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="w",
        )
        # Pre-existing initializer whose name collides with the auto-generated copy name
        weight_copy_1 = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="w_copy_1",
        )
        nodes = [
            helper.make_node(
                "Conv", inputs=["input", weight.name], outputs=["out1"], name="conv_1"
            ),
            helper.make_node(
                "Conv", inputs=["out1", weight.name], outputs=["out2"], name="conv_2"
            ),
            helper.make_node(
                "Conv",
                inputs=["out2", weight_copy_1.name],
                outputs=["output"],
                name="conv_3",
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "NameCollisionModel",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            initializer=[weight, weight_copy_1],
        )
        original = models_for_tests.make_model(graph)
        onnx.checker.check_model(original)
        duplicated = copy.deepcopy(original)

        duplicate_shared_initializers(duplicated.graph)

        # All node inputs must still resolve to a valid initializer
        init_names = {init.name for init in duplicated.graph.initializer}
        for node in duplicated.graph.node:
            assert node.input[1] in init_names, (
                f"Node '{node.name}' references unknown initializer '{node.input[1]}'"
            )

        dummy_input = utils.make_dummy_input(original)
        providers = ["CPUExecutionProvider"]
        original_output = OrtInferenceSession(original, providers).run(
            None, dummy_input
        )
        duplicated_output = OrtInferenceSession(duplicated, providers).run(
            None, dummy_input
        )
        for orig, dup in zip(original_output, duplicated_output):
            np.testing.assert_array_equal(orig, dup)

    @staticmethod
    def _make_identity_passthrough_model(num_consumers: int = 2) -> onnx.ModelProto:
        """
        Build a model where one initializer is shared by ``num_consumers`` Conv
        nodes via a single Identity passthrough:

            input -+-> Conv(weight_id) -+
                   |                    +-> Add -> ...
                   +-> Conv(weight_id) -+

        where ``weight_id`` is the output of ``Identity(weight)``.
        """
        CHANNELS = 4
        weight = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="weight",
        )
        nodes = [
            helper.make_node(
                "Identity",
                inputs=["weight"],
                outputs=["weight_id"],
                name="identity_weight",
            )
        ]
        for i in range(num_consumers):
            nodes.append(
                helper.make_node(
                    "Conv",
                    inputs=["model_input", "weight_id"],
                    outputs=[f"conv_{i}.output"],
                    name=f"conv_{i}",
                )
            )
        prev = "conv_0.output"
        for i in range(1, num_consumers):
            nodes.append(
                helper.make_node(
                    "Add",
                    inputs=[prev, f"conv_{i}.output"],
                    outputs=[f"add_{i}.output"],
                    name=f"add_{i}",
                )
            )
            prev = f"add_{i}.output"

        graph = helper.make_graph(
            nodes,
            "IdentityPassthroughModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    prev, TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            initializer=[weight],
        )
        model = models_for_tests.make_model(graph)
        onnx.checker.check_model(model)
        return model

    @staticmethod
    def _make_identity_chain_model(
        chain_length: int, num_consumers: int
    ) -> onnx.ModelProto:
        """
        Build a model where ``weight`` flows through ``chain_length`` Identity
        nodes (length 0 = direct), then fans out into ``num_consumers`` Conv
        nodes whose outputs are summed into a single graph output.
        """
        CHANNELS = 4
        weight = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="weight",
        )
        nodes: List[onnx.NodeProto] = []
        weight_tensor = "weight"
        for i in range(chain_length):
            out = f"weight_id{i + 1}"
            nodes.append(
                helper.make_node("Identity", [weight_tensor], [out], name=f"id_{i + 1}")
            )
            weight_tensor = out
        for i in range(num_consumers):
            nodes.append(
                helper.make_node(
                    "Conv",
                    inputs=["model_input", weight_tensor],
                    outputs=[f"conv_{i}.output"],
                    name=f"conv_{i}",
                )
            )
        prev = "conv_0.output"
        for i in range(1, num_consumers):
            nodes.append(
                helper.make_node(
                    "Add",
                    inputs=[prev, f"conv_{i}.output"],
                    outputs=[f"add_{i}.output"],
                    name=f"add_{i}",
                )
            )
            prev = f"add_{i}.output"

        graph = helper.make_graph(
            nodes,
            "IdentityChainModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    prev, TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            initializer=[weight],
        )
        model = models_for_tests.make_model(graph)
        onnx.checker.check_model(model)
        return model

    @staticmethod
    def _make_mixed_direct_and_identity_model() -> onnx.ModelProto:
        """
        ``weight`` is consumed directly by one Conv and via an Identity by
        another Conv.  Both consumers must be treated as sharing ``weight``.
        """
        CHANNELS = 4
        weight = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="weight",
        )
        nodes = [
            helper.make_node("Identity", ["weight"], ["weight_id"], name="id_w"),
            helper.make_node(
                "Conv",
                inputs=["model_input", "weight"],
                outputs=["conv_direct.output"],
                name="conv_direct",
            ),
            helper.make_node(
                "Conv",
                inputs=["model_input", "weight_id"],
                outputs=["conv_via_id.output"],
                name="conv_via_id",
            ),
            helper.make_node(
                "Add",
                inputs=["conv_direct.output", "conv_via_id.output"],
                outputs=["model_output"],
                name="add",
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "MixedSharingModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            initializer=[weight],
        )
        model = models_for_tests.make_model(graph)
        onnx.checker.check_model(model)
        return model

    @staticmethod
    def _assert_identity_dup_correct(
        original: onnx.ModelProto, expected_n_duplicates: int
    ):
        """
        Common assertions for Identity-sharing duplication tests.

        ORT may apply different fusion / constant-folding rules when consumers
        bypass an Identity vs traverse it; that can introduce ~1e-6 FP drift
        even though duplicating an initializer is mathematically bit-exact.
        Hence ``assert_allclose`` rather than ``assert_array_equal``.
        """
        duplicated = copy.deepcopy(original)
        n_duplicates = duplicate_shared_initializers(duplicated.graph)
        assert n_duplicates == expected_n_duplicates

        conv_inputs = [
            node.input[1] for node in duplicated.graph.node if node.op_type == "Conv"
        ]
        assert len(set(conv_inputs)) == len(conv_inputs), (
            "Conv consumers must reference distinct tensors after duplication"
        )

        dummy_input = utils.make_dummy_input(original)
        providers = ["CPUExecutionProvider"]
        original_output = OrtInferenceSession(original, providers).run(
            None, dummy_input
        )
        duplicated_output = OrtInferenceSession(duplicated, providers).run(
            None, dummy_input
        )
        for orig, dup in zip(original_output, duplicated_output):
            np.testing.assert_allclose(orig, dup, rtol=1e-5, atol=1e-5)

    def test_duplicate_shared_initializers_identity_passthrough(self):
        """Single Identity feeding two Conv consumers — basic regression test
        for the bug that the original `duplicate_shared_initializers` missed."""
        self._assert_identity_dup_correct(
            self._make_identity_passthrough_model(num_consumers=2),
            expected_n_duplicates=1,
        )

    def test_duplicate_shared_initializers_identity_passthrough_three_consumers(self):
        """Single Identity feeding three Conv consumers — verifies n-1 copies."""
        self._assert_identity_dup_correct(
            self._make_identity_passthrough_model(num_consumers=3),
            expected_n_duplicates=2,
        )

    def test_duplicate_shared_initializers_chained_identity(self):
        """Identity -> Identity -> two Conv consumers — verifies recursive
        traversal through multi-level Identity chains."""
        self._assert_identity_dup_correct(
            self._make_identity_chain_model(chain_length=2, num_consumers=2),
            expected_n_duplicates=1,
        )

    def test_duplicate_shared_initializers_mixed_direct_and_identity(self):
        """Mix of direct reference and Identity passthrough must be merged
        into a single shared-initializer view."""
        self._assert_identity_dup_correct(
            self._make_mixed_direct_and_identity_model(),
            expected_n_duplicates=1,
        )

    def test_duplicate_shared_initializers_shape_consumer_is_skipped(self):
        """
        A Shape consumer only reads tensor metadata, not values, and must not
        be treated as a real consumer.  When the only "extra" consumer is Shape,
        the initializer must NOT be duplicated.
        """
        CHANNELS = 4
        weight = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="weight",
        )
        nodes = [
            helper.make_node(
                "Conv",
                inputs=["model_input", "weight"],
                outputs=["conv.output"],
                name="conv",
            ),
            helper.make_node(
                "Shape",
                inputs=["weight"],
                outputs=["weight_shape"],
                name="shape_w",
            ),
            helper.make_node(
                "Identity",
                inputs=["conv.output"],
                outputs=["model_output"],
                name="id_out",
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "WeightWithShapeConsumer",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            initializer=[weight],
        )
        model = models_for_tests.make_model(graph)
        onnx.checker.check_model(model)

        n_duplicates = duplicate_shared_initializers(model.graph)
        assert n_duplicates == 0, (
            "Shape consumer must not trigger duplication of the shared initializer"
        )

    @staticmethod
    def _make_qdq_model(share_scale_across_pairs: bool = False) -> onnx.ModelProto:
        """
        Build a minimal QDQ model.

        When ``share_scale_across_pairs=False`` (default):
            A single QDQ pair wrapping a Conv weight:

                weight → QuantizeLinear(scale, zp) → DequantizeLinear(scale, zp) → Conv

        When ``share_scale_across_pairs=True``:
            Two independent QDQ pairs that happen to share the same scale/zp:

                weight  → QLinear1(scale, zp) → DQLinear1(scale, zp) → Conv1
                weight2 → QLinear2(scale, zp) → DQLinear2(scale, zp) → Conv2
        """
        CHANNELS = 4
        weight = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="weight",
        )
        scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), name="scale")
        zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="zp")

        nodes = [
            helper.make_node(
                "QuantizeLinear",
                inputs=["weight", "scale", "zp"],
                outputs=["weight_q"],
                name="q1",
            ),
            helper.make_node(
                "DequantizeLinear",
                inputs=["weight_q", "scale", "zp"],
                outputs=["weight_dq"],
                name="dq1",
            ),
            helper.make_node(
                "Conv",
                inputs=["model_input", "weight_dq"],
                outputs=["model_output"],
                name="conv1",
            ),
        ]
        initializers = [weight, scale, zp]

        if share_scale_across_pairs:
            weight2 = numpy_helper.from_array(
                np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
                name="weight2",
            )
            nodes += [
                helper.make_node(
                    "QuantizeLinear",
                    inputs=["weight2", "scale", "zp"],
                    outputs=["weight2_q"],
                    name="q2",
                ),
                helper.make_node(
                    "DequantizeLinear",
                    inputs=["weight2_q", "scale", "zp"],
                    outputs=["weight2_dq"],
                    name="dq2",
                ),
                helper.make_node(
                    "Conv",
                    inputs=["model_input", "weight2_dq"],
                    outputs=["branch2_output"],
                    name="conv2",
                ),
                helper.make_node(
                    "Add",
                    inputs=["model_output", "branch2_output"],
                    outputs=["final_output"],
                    name="add",
                ),
            ]
            initializers.append(weight2)

        final_out = "final_output" if share_scale_across_pairs else "model_output"
        graph = helper.make_graph(
            nodes,
            "QDQModel",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    final_out, TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            initializer=initializers,
        )
        model = models_for_tests.make_model(graph)
        onnx.checker.check_model(model)
        return model

    def test_duplicate_shared_initializers_qdq_scale_not_duplicated(self):
        """
        Given a QDQ pair where scale and zero_point are shared between
        QuantizeLinear and DequantizeLinear:

            weight → QuantizeLinear(scale, zp) → DequantizeLinear(scale, zp) → Conv

        When: duplicate_shared_initializers is called
        Then: scale and zp must NOT be duplicated — Q and DQ of the same pair
              must always refer to the same quantization parameters.
        """
        model = self._make_qdq_model(share_scale_across_pairs=False)

        n_duplicates = duplicate_shared_initializers(model.graph)

        assert n_duplicates == 0, (
            "scale/zp shared within a QDQ pair must not be duplicated"
        )
        init_names = {init.name for init in model.graph.initializer}
        assert "scale" in init_names
        assert "zp" in init_names
        assert "scale_copy_1" not in init_names
        assert "zp_copy_1" not in init_names

    def test_duplicate_shared_initializers_qdq_weight_still_duplicated(self):
        """
        In a QDQ model, a plain weight initializer (not scale/zp) that is
        shared between two Conv nodes must still be duplicated, while the
        QDQ pair's scale and zero_point remain untouched.

        Graph:
            shared_weight → Conv1
            shared_weight → Conv2      ← ordinary shared weight; must be duplicated

            qdq_weight → QuantizeLinear(scale, zp)
                       → DequantizeLinear(scale, zp) → Conv3
                         ↑ scale/zp shared within QDQ pair; must NOT be duplicated
        """
        CHANNELS = 4
        shared_weight = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="shared_weight",
        )
        qdq_weight = numpy_helper.from_array(
            np.random.randn(CHANNELS, CHANNELS, 1, 1).astype(np.float32),
            name="qdq_weight",
        )
        scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), name="scale")
        zp = numpy_helper.from_array(np.array(0, dtype=np.int8), name="zp")

        nodes = [
            # Two Conv nodes directly sharing shared_weight
            helper.make_node(
                "Conv",
                inputs=["model_input", "shared_weight"],
                outputs=["conv1_out"],
                name="conv1",
            ),
            helper.make_node(
                "Conv",
                inputs=["model_input", "shared_weight"],
                outputs=["conv2_out"],
                name="conv2",
            ),
            helper.make_node(
                "Add",
                inputs=["conv1_out", "conv2_out"],
                outputs=["branch1_out"],
                name="add1",
            ),
            # QDQ pair with scale/zp shared between Q and DQ
            helper.make_node(
                "QuantizeLinear",
                inputs=["qdq_weight", "scale", "zp"],
                outputs=["qdq_weight_q"],
                name="q1",
            ),
            helper.make_node(
                "DequantizeLinear",
                inputs=["qdq_weight_q", "scale", "zp"],
                outputs=["qdq_weight_dq"],
                name="dq1",
            ),
            helper.make_node(
                "Conv",
                inputs=["model_input", "qdq_weight_dq"],
                outputs=["conv3_out"],
                name="conv3",
            ),
            helper.make_node(
                "Add",
                inputs=["branch1_out", "conv3_out"],
                outputs=["model_output"],
                name="add2",
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "QDQWithSharedWeight",
            inputs=[
                helper.make_tensor_value_info(
                    "model_input", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "model_output", TensorProto.FLOAT, [1, CHANNELS, 8, 8]
                )
            ],
            initializer=[shared_weight, qdq_weight, scale, zp],
        )
        model = models_for_tests.make_model(graph)
        onnx.checker.check_model(model)

        n_duplicates = duplicate_shared_initializers(model.graph)

        # shared_weight is used by two Conv nodes → must be duplicated once
        assert n_duplicates == 1
        init_names = {init.name for init in model.graph.initializer}
        assert "shared_weight_copy_1" in init_names

        # Conv1 and Conv2 must now reference distinct initializers
        conv_inputs = [
            node.input[1]
            for node in model.graph.node
            if node.op_type == "Conv" and node.name in ("conv1", "conv2")
        ]
        assert conv_inputs[0] != conv_inputs[1]

        # scale and zp must NOT have been duplicated
        assert "scale_copy_1" not in init_names
        assert "zp_copy_1" not in init_names

    def test_duplicate_shared_initializers_qdq_cross_pair_scale_excluded(self):
        """
        When the same scale/zp initializer is shared across two different QDQ
        pairs, it is currently excluded from duplication.

        Graph:
            weight  → QLinear1(scale, zp) → DQLinear1(scale, zp) → Conv1
            weight2 → QLinear2(scale, zp) → DQLinear2(scale, zp) → Conv2
        """
        model = self._make_qdq_model(share_scale_across_pairs=True)

        n_duplicates = duplicate_shared_initializers(model.graph)

        assert n_duplicates == 0, "Cross-pair shared scale/zp is excluded"
        init_names = {init.name for init in model.graph.initializer}
        assert "scale_copy_1" not in init_names
        assert "zp_copy_1" not in init_names


class TestORTInferenceSession:
    """
    Test OrtInferenceSession class in utils
    """

    def test_user_provided_directory(self):
        """
        Test user provided directory
        """
        model = models_for_tests.build_dummy_model()

        with tempfile.TemporaryDirectory() as tmp_dir:
            session = utils.OrtInferenceSession(
                model=model, providers=["CPUExecutionProvider"], path=tmp_dir
            )
            assert session is not None
            assert session.model_dir is None

    def test_session_managed_directory(self):
        """
        Test Session managed directory
        """
        model = models_for_tests.build_dummy_model()

        session = utils.OrtInferenceSession(
            model=model, providers=["CPUExecutionProvider"]
        )
        assert session.model_dir is not None
        assert os.path.exists(session.model_dir)
        assert session is not None

        model_dir = session.model_dir

        del session

        # Ensure temp directory is deleted after session manager is deleted
        gc.collect()
        assert not os.path.exists(model_dir)
