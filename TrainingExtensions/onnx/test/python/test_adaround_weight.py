# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


"""Unit tests for Adaround Weights"""

import copy
from unittest.mock import patch

import ml_dtypes
import numpy as np
import onnxruntime as ort
import torch
import pytest
from onnx import numpy_helper
import onnx

import aimet_onnx
from aimet_onnx import apply_adaround, QuantizationSimModel
from aimet_onnx.adaround.utils import AdaroundSupportedModules, ModelData
from aimet_onnx.common.onnx._utils import to_array
from aimet_onnx.utils import make_dummy_input, ParamUtils
from .models import models_for_tests
from .models.models_for_tests import (
    conv_prelu_model,
    ParallelConvSharedWeights,
    _convert_to_onnx_no_fold,
)
from .utils import tmp_dir


class TestAdaround:
    """
    AdaRound Weights Unit Test Cases
    """

    @pytest.mark.parametrize(
        "providers",
        (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    def test_apply_adaround(self, providers):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")
        np.random.seed(0)
        torch.manual_seed(0)
        model = models_for_tests.single_residual_model()
        dummy_input = {"input": np.random.rand(1, 3, 32, 32).astype(np.float32)}

        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=providers,
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
        )
        sim.compute_encodings([dummy_input])
        out_before_ada = sim.session.run(None, dummy_input)
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)
        out_after_ada = sim.session.run(None, dummy_input)
        assert not np.array_equal(out_before_ada[0], out_after_ada[0])

        sim.remove_quantizers(sim.model.model)
        for node in sim.model.nodes():
            if node.op_type in AdaroundSupportedModules:
                assert sim.qc_quantize_op_dict[node.input[1]]._is_encoding_frozen

    @pytest.mark.parametrize("pre_calibrate", [True, False])
    @pytest.mark.parametrize(
        "model",
        [
            models_for_tests.single_residual_model().model,
            models_for_tests.depthwise_conv_model().model,
            models_for_tests.add_matmul_model(),
            models_for_tests.model_with_split_matmul(),
        ],
    )
    @pytest.mark.parametrize(
        "providers",
        (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]),
    )
    def test_apply_adaround_2(self, providers, model, pre_calibrate):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")

        inputs = [make_dummy_input(model) for _ in range(2)]
        sim = QuantizationSimModel(copy.deepcopy(model), providers=providers)
        graph_outputs = sim.model.graph().output
        is_enabled = {name: q.enabled for name, q in sim.qc_quantize_op_dict.items()}

        adaroundable_ops = [
            op
            for op in sim.connected_graph.ordered_ops
            if op.type in ("Conv", "MatMul", "Gemm")
        ]
        weight_tensors = {
            t.name: copy.deepcopy(onnx.numpy_helper.to_array(t.tensor))
            for op in adaroundable_ops
            for t, param_type in op.parameters.values()
            if param_type == "weight"
        }

        assert weight_tensors

        if pre_calibrate:
            sim.compute_encodings(inputs)

        apply_adaround(sim, inputs, num_iterations=5)

        # Quantizer enabled state must be restored after adaround
        for name, q in sim.qc_quantize_op_dict.items():
            assert q.enabled == is_enabled[name]

        # Only optimized weights should have frozen encodings
        for name, quantizer in sim.qc_quantize_op_dict.items():
            assert quantizer.is_encoding_frozen() == (name in weight_tensors)

        # Optimized weight should not be equal to original weight
        for name, old_weight in weight_tensors.items():
            new_weight = onnx.numpy_helper.to_array(
                ParamUtils.get_param_by_name(sim.model.model, name)
            )
            assert not np.all(old_weight == new_weight)

        assert graph_outputs == sim.model.graph().output

    @pytest.mark.skip_on_windows_arm64(
        "onnxruntime_extensions is not available on Windows ARM64"
    )
    @pytest.mark.parametrize(
        "providers",
        (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    def test_apply_adaround_for_custom_op(self, providers, tmp_dir):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")
        from onnxruntime_extensions import get_library_path

        model = models_for_tests.custom_add_model()
        onnx_library = get_library_path()
        np.random.seed(0)
        dummy_input = {"input": np.random.rand(1, 3, 64, 64).astype(np.float32)}
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=providers,
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
            user_onnx_libs=[onnx_library],
            path=tmp_dir,
        )
        model_data = ModelData(sim)
        orig_weight = torch.from_numpy(
            numpy_helper.to_array(
                model_data.module_to_info["conv"].params["weight"].tensor
            )
        )
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)
        model_data = ModelData(sim)
        updated_weight = torch.from_numpy(
            numpy_helper.to_array(
                model_data.module_to_info["conv"].params["weight"].tensor
            )
        )
        assert not torch.equal(orig_weight, updated_weight)
        sim.compute_encodings([dummy_input])
        sim.remove_quantizers(sim.model.model)
        for node in sim.model.nodes():
            if node.op_type in AdaroundSupportedModules:
                assert sim.qc_quantize_op_dict[node.input[1]]._is_encoding_frozen

    @pytest.mark.parametrize(
        "model, input_shape",
        [
            (models_for_tests.weight_gemm_model(10, 20, True), (1, 10)),
            (models_for_tests.weight_gemm_model(10, 20, False), (1, 10)),
            (models_for_tests.weight_matmul_model(10, 20), (1, 10, 10)),
        ],
    )
    @pytest.mark.parametrize(
        "providers",
        (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    def test_adaround_matmul_gemm(self, model, input_shape, tmpdir, providers):
        if "CUDAExecutionProvider" in providers and not torch.cuda.is_available():
            pytest.skip("Cuda not available")

        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=providers,
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
        )

        dummy_input = {"input": np.random.rand(*input_shape).astype(np.float32)}
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)
        sim.remove_quantizers(sim.model.model)
        for node in sim.model.nodes():
            if node.op_type in AdaroundSupportedModules:
                assert sim.qc_quantize_op_dict[node.input[1]]._is_encoding_frozen

    @pytest.mark.parametrize(
        "model, input_shape", [(models_for_tests.dynamic_matmul_model(1), (1, 10))]
    )
    def test_adaround_dynamic_matmul(self, model, input_shape, tmpdir):
        """
        AdaRound should not error-out if there is a dynamic matmul
        """
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=["CPUExecutionProvider"],
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
        )
        dummy_input = {"input": np.random.rand(*input_shape).astype(np.float32)}
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)

    @pytest.mark.skip_on_windows_arm64("onnxsim is not available on Windows ARM64")
    @pytest.mark.parametrize(
        "model, input_shape", [(models_for_tests.simplifiable_model(1), (1, 10))]
    )
    def test_adaround_simplifiable_model(self, model, input_shape, tmpdir):
        """
        AdaRound should not error-out for models which need simplification
        """
        from onnxsim import simplify

        model, _ = simplify(model)
        dummy_input = {"input": np.random.rand(*input_shape).astype(np.float32)}
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=["CPUExecutionProvider"],
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
        )
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)

    @pytest.mark.parametrize(
        "model_factory, input_shape",
        [
            (models_for_tests.pointwise_conv1d, (1, 10, 32)),
            (models_for_tests.pointwise_conv3d, (1, 10, 8, 8, 8)),
            (models_for_tests.pointwise_convtranspose1d, (1, 10, 32)),
            (models_for_tests.pointwise_convtranspose3d, (1, 10, 8, 4, 3)),
            (models_for_tests.padded_convtranspose2d, (1, 10, 32, 32)),
        ],
    )
    def test_adaround_convNd_model(self, model_factory, input_shape, tmpdir):
        """
        AdaRound should not error-out for non-2d Conv/ConvTranspose layers
        """
        model = model_factory(input_shape)

        dummy_input = {"input": np.random.rand(*input_shape).astype(np.float32)}
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=["CPUExecutionProvider"],
            param_type=aimet_onnx.int4,
            activation_type=aimet_onnx.int16,
        )
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)

    @pytest.mark.parametrize(
        "config",
        [
            # model, ops_to_optimize input to pass, expected result
            (
                models_for_tests.single_residual_model().model,
                [],
                [
                    "/conv1/Conv",
                    "/conv2/Conv",
                    "/conv3/Conv",
                    "/conv4/Conv",
                    "/fc/Gemm",
                ],
            ),
            (
                models_for_tests.single_residual_model().model,
                ["/conv1/Conv"],
                ["/conv1/Conv"],
            ),
            (
                models_for_tests.single_residual_model().model,
                ["/conv2/Conv"],
                ["/conv2/Conv"],
            ),
            (
                models_for_tests.single_residual_model().model,
                [
                    "/conv1/Conv",
                    "/conv4/Conv",
                    "/conv2/Conv",
                    "/conv3/Conv",
                ],
                ["/conv1/Conv", "/conv4/Conv", "/conv2/Conv", "/conv3/Conv"],
            ),
        ],
    )
    def test_whitelist_functionality(self, config):
        model, whitelist_ops, expected = config
        inputs = [make_dummy_input(model) for _ in range(2)]
        sim = QuantizationSimModel(
            copy.deepcopy(model),
            providers=["CPUExecutionProvider"],
        )

        param_to_op_name_dict = {}
        for cg_op in sim.connected_graph.get_all_ops().values():
            if cg_op.type in AdaroundSupportedModules:
                param_to_op_name_dict[cg_op.inputs[1].name] = cg_op.name

        ops_processed = []

        def mock_adaround_module(module, *args, **kwargs):
            ops_processed.append(param_to_op_name_dict[module.params["weight"].name])

        with patch(
            "aimet_onnx.adaround.adaround_optimizer.AdaroundOptimizer.adaround_module",
            mock_adaround_module,
        ):
            apply_adaround(
                sim, inputs, num_iterations=5, nodes_to_include=whitelist_ops
            )

            print([name for name in ops_processed])
            assert ops_processed.sort() == expected.sort()

    def test_adaround_skips_shared_weights(self):
        torch.manual_seed(10)
        model = _convert_to_onnx_no_fold(
            ParallelConvSharedWeights(), torch.randn(2, 10, 24, 24)
        )
        inputs = [make_dummy_input(model.model) for _ in range(2)]
        sim = QuantizationSimModel(
            copy.deepcopy(model.model), providers=["CPUExecutionProvider"]
        )

        from aimet_onnx.utils import find_shared_param_names

        shared_param_names = find_shared_param_names(sim.connected_graph)

        shared_weight_names = set()
        non_shared_weight_names = set()
        for op in sim.connected_graph.ordered_ops:
            if op.type not in AdaroundSupportedModules:
                continue
            for product, param_type in op.parameters.values():
                if param_type != "weight":
                    continue
                if product.name in shared_param_names:
                    shared_weight_names.add(product.name)
                else:
                    non_shared_weight_names.add(product.name)

        assert shared_weight_names
        assert non_shared_weight_names

        apply_adaround(sim, inputs, num_iterations=5)

        for name in shared_weight_names:
            assert not sim.qc_quantize_op_dict[name].is_encoding_frozen()
        for name in non_shared_weight_names:
            assert sim.qc_quantize_op_dict[name].is_encoding_frozen()

    def test_activation_with_param(self):
        if not torch.cuda.is_available():
            pytest.skip("Cuda not available")

        model = conv_prelu_model().model
        inputs = [make_dummy_input(model) for _ in range(2)]
        sim = QuantizationSimModel(
            copy.deepcopy(model), providers=["CUDAExecutionProvider"]
        )
        apply_adaround(sim, inputs, 10)
        # check adaround went through fine
        assert sim.qc_quantize_op_dict["conv1.weight"]._is_encoding_frozen == True

    def test_adaround_optimizes_model_with_shared_bias(self, tmp_path):
        onnx_model = models_for_tests.conv_model_with_shared_bias(tmp_path)
        sim = QuantizationSimModel(onnx_model)
        dummy_input = make_dummy_input(onnx_model)
        sim.compute_encodings([dummy_input])
        out_before_ada = sim.session.run(None, dummy_input)

        # Check output changes after adaround (weights should be updated)
        apply_adaround(sim, [dummy_input for _ in range(2)], 5)
        out_after_ada = sim.session.run(None, dummy_input)
        assert not np.array_equal(out_before_ada[0], out_after_ada[0])

        # Check weight encodings get frozen
        conv1_weight = sim.connected_graph.ordered_ops[0].inputs[1].name
        conv2_weight = sim.connected_graph.ordered_ops[1].inputs[1].name
        assert sim.qc_quantize_op_dict[conv1_weight].is_encoding_frozen()
        assert sim.qc_quantize_op_dict[conv2_weight].is_encoding_frozen()

    @pytest.mark.cuda
    @pytest.mark.parametrize("io_dtype", [torch.float32, torch.bfloat16])
    def test_adaround_bfloat16_model(self, io_dtype):
        """
        Given: Model computing in bfloat16, with float32 or bfloat16 graph inputs/outputs
        When: Apply AdaRound
        Then: Every weight stays bfloat16, its encoding is frozen, and its rounding
              decisions agree with those AdaRound makes on the same weights in float32
        """
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            pytest.skip("CUDA EP not available")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        rng = np.random.default_rng(0)
        fp32_inputs = [
            {"input": rng.standard_normal((64, 16)).astype(np.float32)}
            for _ in range(4)
        ]
        weight_names = ("fc1.weight", "fc2.weight")

        def adaround(model, inputs):
            sim = QuantizationSimModel(
                model, param_type="int4", activation_type="int16", providers=providers
            )
            weights = {
                init.name: to_array(init).astype(np.float64)
                for init in sim.model.model.graph.initializer
            }
            np.random.seed(0)
            torch.manual_seed(0)
            apply_adaround(sim, inputs, num_iterations=1000)
            return sim, weights

        # Round the float32 model's weights to bfloat16 so both models start equal
        fp32_model = models_for_tests.simple_mlp_model(torch.float32)
        for init in fp32_model.graph.initializer:
            rounded = to_array(init).astype(ml_dtypes.bfloat16).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(rounded, init.name))
        bf16_inputs = (
            fp32_inputs
            if io_dtype == torch.float32
            else [
                {"input": item["input"].astype(ml_dtypes.bfloat16)}
                for item in fp32_inputs
            ]
        )
        fp32_sim, _ = adaround(fp32_model, fp32_inputs)
        bf16_sim, original = adaround(
            models_for_tests.simple_mlp_model(torch.bfloat16, io_dtype=io_dtype),
            bf16_inputs,
        )

        def rounded_up(sim, name):
            """Whether the sim's weight quantizer rounds each weight up."""
            init = ParamUtils.get_param_by_name(sim.model.model, name)
            delta = np.array(
                [e.delta for e in sim.qc_quantize_op_dict[name].get_encodings()]
            )[:, None]
            steps = to_array(init).astype(np.float64) / delta
            return np.round(steps) > np.floor(original[name] / delta)

        for name in weight_names:
            init = ParamUtils.get_param_by_name(bf16_sim.model.model, name)
            assert init.data_type == onnx.TensorProto.BFLOAT16, name
            assert bf16_sim.qc_quantize_op_dict[name].is_encoding_frozen(), name
            agreement = np.mean(
                rounded_up(bf16_sim, name) == rounded_up(fp32_sim, name)
            )
            # bfloat16 activations shift a few borderline decisions
            assert agreement > 0.98, (name, agreement)

        """
        When: Calibrate activations after AdaRound
        Then: The sim runs end-to-end on bfloat16 inputs
        """
        bf16_sim.compute_encodings(bf16_inputs)


def dataloader(input_shape: tuple, batch_size=2):
    class DataLoader:
        """
        Example of a Dataloader which can be used for running AMPv2
        """

        def __init__(self, batch_size: int, input_shape: tuple):
            """
            :param batch_size: batch size for data loader
            """
            self.batch_size = batch_size
            self.input_shape = input_shape

        def __iter__(self):
            """Iterates over dataset"""
            dummy_input = np.random.rand(*self.input_shape).astype(np.float32)
            yield dummy_input

        def __len__(self):
            return 4

    dummy_dataloader = DataLoader(batch_size=batch_size, input_shape=input_shape)
    return dummy_dataloader
