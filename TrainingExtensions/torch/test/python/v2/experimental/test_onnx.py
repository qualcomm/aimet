# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import copy
import os
import itertools
import json
import pathlib
from packaging import version
import onnxruntime as ort
import pytest
import contextlib
import numpy as np
import torch
from torch.onnx import _constants
import onnx
from onnx import helper, TensorProto
import onnx_ir
import tempfile
from unittest.mock import patch

from ..models_ import test_models

from aimet_torch.common.quantsim_config.utils import get_path_for_per_tensor_config
import aimet_torch.v2.quantization as Q
from aimet_torch.quantization.float.quantizer import _float_quantize_dequantize
from aimet_torch.quantization.float.encoding import _MXFP4Encoding, _NVFP4Encoding
from aimet_torch.quantization.float._finfo import (
    _finfo,
    _float4_e2m1fn,
    _float8_e5m2,
    _float8_e5m2fnuz,
    _float8_e4m3fn,
    _float8_e4m3fnuz,
)
from aimet_torch.v2.quantsim.quantsim import QuantizationSimModel
from aimet_torch.onnx import (
    _concretize_int32_bias_quantizers,
    _derive_data_movement_op_encodings,
)
from torchvision.models import mobilenet_v3_small as _mobilenet_v3_small
from torchvision.models.resnet import _resnet, BasicBlock, ResNet
from aimet_torch.experimental.onnx._export import (
    export as _export,
    _get_all_constants,
)
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.v2.utils import patch_attr, remove_activation_quantizers
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim.config_utils import (
    set_blockwise_quantization_for_weights,
    set_grouped_blockwise_quantization_for_weights,
)
import aimet_torch
from aimet_torch.onnx import _to_onnx
from aimet_torch.nn.modules import custom as aimet_ops
from aimet_torch.nn import (
    QuantizedConv1d,
    QuantizedConv2d,
    QuantizedConv3d,
    QuantizedConvTranspose1d,
    QuantizedConvTranspose2d,
    QuantizedConvTranspose3d,
    QuantizedEmbedding,
    QuantizedLinear,
)


def _get_qdq_encoding_map(onnx_model):
    """Build a mapping from activation tensor name to (scale, zero_point) arrays.

    For each QuantizeLinear node, maps its *input* tensor to (scale, zp).
    For each DequantizeLinear node, maps its *output* tensor to (scale, zp).
    This covers both sides of a QDQ pair and any standalone Q or DQ.
    """
    constants = _get_all_constants(onnx_model)
    encoding_map = {}
    for node in onnx_model.graph.node:
        if node.op_type == "QuantizeLinear":
            scale = onnx.numpy_helper.to_array(constants[node.input[1]])
            zp = onnx.numpy_helper.to_array(constants[node.input[2]])
            encoding_map[node.input[0]] = (scale, zp)
        elif node.op_type == "DequantizeLinear":
            scale = onnx.numpy_helper.to_array(constants[node.input[1]])
            zp = onnx.numpy_helper.to_array(constants[node.input[2]])
            encoding_map[node.output[0]] = (scale, zp)
    return encoding_map


@pytest.fixture(autouse=True, params=range(1))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)


@pytest.mark.parametrize(
    "qtzr_cls", [Q.affine.Quantize, Q.affine.QuantizeDequantize, Q.affine.Dequantize]
)
@pytest.mark.parametrize(
    "input_shape, scale_shape, block_size",
    [
        ([], [], None),  # per-tensor
        ((100, 100), (1,), None),  # per-tensor
        ((100, 100), [], None),  # per-tensor
        ((100, 100), (100, 1), None),  # per-channel
        ((100, 100), (100, 1), (1, 100)),  # per-channel
        ((100, 100), (100, 50), (1, 2)),  # blockwise
        ((100, 100), (50, 100), (2, 1)),  # blockwise
        ((100, 100), (50, 50), (2, 2)),  # blockwise
        ((100, 100), (50, 50), (-1, -1)),  # blockwise
    ],
)
@pytest.mark.parametrize("symmetric", [True, False])
def test_quantize_torch_ort_equal(
    qtzr_cls, input_shape, scale_shape, block_size, symmetric
):
    """
    When: Export a quantizer with torch.onnx.export
    """
    x = torch.randn(input_shape)
    qtzr = qtzr_cls(scale_shape, 8, symmetric, block_size=block_size)
    with qtzr.compute_encodings():
        _ = qtzr(x)

    with tempfile.TemporaryDirectory() as dirname:
        full_path = os.path.join(dirname, "qtzr.onnx")

        with open(full_path, "wb") as f:
            _export(
                qtzr, x, f, input_names=["input"], output_names=["output"], dynamo=False
            )

        with torch.no_grad():
            y = qtzr(x)

        """
        Then: The saved onnx model should pass onnx model checker
        """
        model = onnx.load_model(full_path)
        onnx.checker.check_model(model)

        """
        Then: The saved onnx model should contain exactly one graph node in "aimet" domain
              with proper name and attributes
        """
        nodes = [node for node in model.graph.node if node.domain == "aimet"]
        assert len(nodes) == 1
        (node,) = nodes

        assert (
            node.name == "/quantize"
            if qtzr_cls is Q.affine.Quantize
            else "/quantize_dequantize"
        )

        if block_size:
            assert node.attribute[0].name == "block_size"
            assert node.attribute[0].ints == list(
                np.array(input_shape) // np.array(scale_shape)
            )
        else:
            assert not any(attr.name == "block_size" for attr in node.attribute)

        if qtzr_cls != Q.affine.Dequantize:
            assert node.attribute[bool(block_size) + 0].name == "qmax"
            assert node.attribute[bool(block_size) + 0].i == (127 if symmetric else 255)
            assert node.attribute[bool(block_size) + 1].name == "qmin"
            assert node.attribute[bool(block_size) + 1].i == (-128 if symmetric else 0)

        """
        Then: The saved onnx model should contain exactly one graph node in "aimet" domain
              with proper scale and offset values
        """
        constants = _get_all_constants(model)
        assert node.input[1] in constants
        assert node.input[2] in constants
        onnx_scale = torch.tensor(onnx.numpy_helper.to_array(constants[node.input[1]]))
        onnx_offset = torch.tensor(onnx.numpy_helper.to_array(constants[node.input[2]]))
        if scale_shape == []:
            onnx_scale.squeeze_(0)
            onnx_offset.squeeze_(0)
        assert torch.equal(onnx_scale, qtzr.get_scale())
        assert torch.equal(onnx_offset, qtzr.get_offset())

        """
        Then: The saved onnx model should produce the same output with the original quantizer
              given the same input
        """
        sess = ort.InferenceSession(full_path, providers=["CPUExecutionProvider"])
        (out,) = sess.run(None, {"input": x.numpy()})
        assert torch.equal(torch.from_numpy(out), y)


@pytest.mark.parametrize(
    "input_shape, scale_shape, block_size",
    [
        ([], [], None),  # per-tensor
        ((100, 100), (1,), None),  # per-tensor
        ((100, 100), [], None),  # per-tensor
        ((100, 100), (100, 1), None),  # per-channel
        ((100, 100), (100, 1), (1, 100)),  # per-channel
        ((100, 100), (100, 50), (1, 2)),  # blockwise
        ((100, 100), (50, 100), (2, 1)),  # blockwise
        ((100, 100), (50, 50), (2, 2)),  # blockwise
        ((100, 100), (50, 50), (-1, -1)),  # blockwise
    ],
)
@pytest.mark.parametrize("symmetric", [True, False])
def test_dequantize_torch_ort_equal(input_shape, scale_shape, block_size, symmetric):
    """
    When: Export dequantize with torch.onnx.export
    """

    class Dequantize(torch.nn.Module):
        def forward(self, x: Q.QuantizedTensor):
            return x.dequantize()

    x = torch.randn(input_shape)
    qtzr = Q.affine.Quantize(scale_shape, 8, symmetric, block_size=block_size)
    with qtzr.compute_encodings():
        x = qtzr(x)

    with tempfile.TemporaryDirectory() as dirname:
        full_path = os.path.join(dirname, "qtzr.onnx")

        with open(full_path, "wb") as f:
            _export(
                Dequantize(),
                x,
                f,
                input_names=["input"],
                output_names=["output"],
                dynamo=False,
            )

        with torch.no_grad():
            y = x.dequantize()

        """
        Then: The saved onnx model should pass onnx model checker
        """
        model = onnx.load_model(full_path)
        onnx.checker.check_model(model)

        """
        Then: The saved onnx model should contain exactly one graph node in "aimet" domain
              with proper name and attributes
        """
        nodes = [node for node in model.graph.node if node.domain == "aimet"]
        assert len(nodes) == 1
        (node,) = nodes

        assert node.name == "/dequantize"

        if block_size:
            assert node.attribute[0].name == "block_size"
            assert node.attribute[0].ints == list(
                np.array(input_shape) // np.array(scale_shape)
            )
        else:
            assert not any(attr.name == "block_size" for attr in node.attribute)

        """
        Then: The saved onnx model should produce the same output with the original quantizer
              given the same input
        """
        sess = ort.InferenceSession(full_path, providers=["CPUExecutionProvider"])
        (out,) = sess.run(None, {"input": x.numpy()})
        assert torch.equal(torch.from_numpy(out), y)


@pytest.fixture(scope="module")
def mobilenet_v3_small():
    torch.manual_seed(0)
    return _mobilenet_v3_small().eval()


@pytest.fixture(scope="module")
def resnet_tiny() -> ResNet:
    torch.manual_seed(0)
    return _resnet(BasicBlock, [1, 1, 1, 1], None, True).eval()


@pytest.fixture(scope="module")
def _resnet_tiny_sim_singleton(resnet_tiny):
    dummy_input = torch.randn(1, 3, 30, 30)
    model = prepare_model(resnet_tiny)
    fold_all_batch_norms(model, None, dummy_input)
    sim = QuantizationSimModel(
        resnet_tiny,
        dummy_input=dummy_input,
        default_param_bw=8,
        default_output_bw=16,
    )

    # Set fc to lpbq
    set_grouped_blockwise_quantization_for_weights(
        sim,
        [torch.nn.Linear],
        bitwidth=4,
        symmetric=True,
        decompressed_bw=8,
        block_size=64,
    )
    return sim


@pytest.fixture(scope="function")
def resnet_tiny_sim(_resnet_tiny_sim_singleton) -> QuantizationSimModel:
    return copy.deepcopy(_resnet_tiny_sim_singleton)


@torch.no_grad()
@pytest.mark.parallel
@pytest.mark.parametrize("dynamo", [False, True])
@pytest.mark.parametrize("encoding_version", ["0.6.1", "1.0.0", "2.0.0"])
@pytest.mark.parametrize("export_int32_bias", [False, True])
@pytest.mark.parametrize("fold_param_quantizers", [False, True])
def test_quantsim_export_resnet(
    tmp_path: pathlib.Path,
    resnet_tiny_sim: QuantizationSimModel,
    encoding_version: str,
    fold_param_quantizers: bool,
    export_int32_bias: bool,
    dynamo: bool,
):
    """
    When: Export quantized torchvision model using quantsim.export
    """
    x = torch.randn(1, 3, 224, 224)
    sim = resnet_tiny_sim
    sim.compute_encodings(lambda model: model(x))

    # Compute original pytorch model output with qdq weights
    with (
        _concretize_int32_bias_quantizers(sim.model, x)
        if export_int32_bias
        else contextlib.nullcontext()
    ):
        expected_param_encodings = {
            f"{module_name}.{param_name}": (
                qtzr.get_encodings()
                ._hint_input_shape(qmodule.get_parameter(param_name).shape)
                .to_qnn_encoding_dict(encoding_version)
            )
            for module_name, qmodule in sim.named_qmodules()
            for param_name, qtzr in qmodule.param_quantizers.items()
            if isinstance(qtzr, Q.affine.AffineQuantizerBase)
        }

        expected_activation_encodings = {}
        expected_activation_encodings.update(
            {
                f"{module_name}.input_quantizers.{i}": qtzr.get_encodings().to_qnn_encoding_dict(
                    encoding_version
                )
                for module_name, qmodule in sim.named_qmodules()
                for i, qtzr in enumerate(qmodule.input_quantizers)
                if isinstance(qtzr, Q.affine.AffineQuantizerBase)
            }
        )
        expected_activation_encodings.update(
            {
                f"{module_name}.output_quantizers.{i}": qtzr.get_encodings().to_qnn_encoding_dict(
                    encoding_version
                )
                for module_name, qmodule in sim.named_qmodules()
                for i, qtzr in enumerate(qmodule.output_quantizers)
                if isinstance(qtzr, Q.affine.AffineQuantizerBase)
            }
        )

        with remove_activation_quantizers(sim.model):
            expected_out = sim.model(x)

    if fold_param_quantizers:
        sim.fold_param_quantizers()

    onnx_path = tmp_path / "torchvision_model.onnx"
    encodings_path = tmp_path / "torchvision_model.encodings"

    sim.onnx.export(
        x,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        export_int32_bias=export_int32_bias,
        dynamo=dynamo,
        encoding_version=encoding_version,
    )

    """
    Then: The saved onnx model should pass onnx model checker
    """
    onnx_model = onnx.load_model(onnx_path)
    onnx.checker.check_model(onnx_model)

    """
    Then: Input/Output names should be strictly honored
    """
    assert list(x.name for x in onnx_model.graph.input) == ["input"]
    assert list(y.name for y in onnx_model.graph.output) == ["output"]

    with open(encodings_path) as f:
        onnx_encodings = json.load(f)

    onnx_weight_names = set(
        convfc.input[1]
        for convfc in onnx_model.graph.node
        if convfc.op_type in ("Conv", "Gemm")
    )
    onnx_bias_names = set(
        convfc.input[2]
        for convfc in onnx_model.graph.node
        if convfc.op_type in ("Conv", "Gemm") and len(convfc.input) > 2
    )
    quantized_param_names = (
        onnx_weight_names | onnx_bias_names if export_int32_bias else onnx_weight_names
    )

    """
    Then: The onnx encodings should have the same number of encodings
          as the number of quantizers in the original pytorch model
    """
    if encoding_version < "2.0.0":
        assert len(onnx_encodings["param_encodings"]) == len(quantized_param_names)
        # Exported encodings can contain MORE encodings than quantsim
        # due to data movement op's output encodings that are generated
        # on-the-fly during export
        assert len(onnx_encodings["activation_encodings"]) >= len(
            expected_activation_encodings
        )
    else:
        # Exported encodings can contain MORE encodings than quantsim
        # due to data movement op's output encodings that are generated
        # on-the-fly during export
        assert len(onnx_encodings["encodings"]) >= (
            len(expected_activation_encodings) + len(quantized_param_names)
        )

    """
    Then: The onnx encodings should have the same scale and offset value
          as the values of quantizers in the original pytorch model
    """
    if encoding_version == "0.6.1":
        for name, e in onnx_encodings["param_encodings"].items():
            if name in expected_param_encodings:
                assert e == expected_param_encodings[name]
            else:
                assert any(
                    len(e) == len(expected)
                    and e[i]["scale"] == expected[i]["scale"]
                    and e[i]["offset"] == expected[i]["offset"]
                    and e[i]["bitwidth"] == expected[i]["bitwidth"]
                    for expected in expected_param_encodings.values()
                    for i in range(len(e))
                )
        for e in onnx_encodings["activation_encodings"].values():
            assert any(
                e[0]["scale"] == expected[0]["scale"]
                and e[0]["offset"] == expected[0]["offset"]
                and e[0]["bitwidth"] == expected[0]["bitwidth"]
                for expected in expected_activation_encodings.values()
            )
    elif encoding_version == "1.0.0":
        for e in onnx_encodings["param_encodings"]:
            name = e.pop("name")
            if name in expected_param_encodings:
                assert e == expected_param_encodings[name]
            else:
                assert any(
                    e["scale"] == expected["scale"]
                    and e["offset"] == expected["offset"]
                    and e["bw"] == expected["bw"]
                    for expected in expected_param_encodings.values()
                )

        for e in onnx_encodings["activation_encodings"]:
            assert any(
                e["scale"] == expected["scale"]
                and e["offset"] == expected["offset"]
                and e["bw"] == expected["bw"]
                for expected in expected_activation_encodings.values()
            )
    elif encoding_version == "2.0.0":
        expected_encodings = expected_param_encodings | expected_activation_encodings

        for e in onnx_encodings["encodings"]:
            name = e.pop("name")
            if name in expected_encodings:
                expected = expected_encodings[name]
                if name in expected_param_encodings and "axis" in expected:
                    weight_dim = sim.model.get_parameter(name).dim()
                    # Make positive
                    expected["axis"] = (expected["axis"] + weight_dim) % weight_dim

                assert e == expected
                continue

            assert any(
                e.get("output_dtype") == expected.get("output_dtype")
                and e.get("y_scale") == expected.get("y_scale")
                and e.get("y_zero_point") == expected.get("y_zero_point")
                and e.get("per_channel_float_scale")
                == expected.get("per_channel_float_scale")
                and e.get("per_block_int_scale") == expected.get("per_block_int_scale")
                for expected in expected_encodings.values()
            )
    else:
        raise RuntimeError(f"Unexpected encoding veresion: {encoding_version}")

    """
    Then: The exported onnx model should produce output close enough to
          the original pytorch model with qdq weights
    """
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {"input": x.numpy()})

    assert torch.allclose(torch.from_numpy(out), expected_out, atol=1e-5)


@pytest.mark.parallel
@pytest.mark.parametrize("dynamo", [False, True])
@pytest.mark.parametrize("fold_param_quantizers", [False, True])
@pytest.mark.parametrize("export_int32_bias", [True, False])
def test_quantsim_export_onnx_qdq_resnet(
    tmp_path: pathlib.Path,
    resnet_tiny_sim: QuantizationSimModel,
    export_int32_bias: bool,
    fold_param_quantizers: bool,
    dynamo: bool,
):
    """
    When: Export quantized torchvision model using quantsim.export
    """
    x = torch.randn(1, 3, 224, 224)
    sim = resnet_tiny_sim
    sim.compute_encodings(lambda model: model(x))

    with (
        _concretize_int32_bias_quantizers(sim.model, x)
        if export_int32_bias
        else contextlib.nullcontext()
    ):
        expected_out = sim.model(x)
        activation_qdq_nodes = [
            qtzr
            for _, qmodule in sim.named_qmodules()
            for qtzr in itertools.chain(
                qmodule.input_quantizers, qmodule.output_quantizers
            )
            if isinstance(qtzr, Q.affine.AffineQuantizerBase)
        ]

    if fold_param_quantizers:
        sim.fold_param_quantizers()

    onnx_path = tmp_path / "torchvision_model.onnx"
    aimet_torch.onnx.export(
        sim,
        x,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        export_int32_bias=export_int32_bias,
        dynamo=dynamo,
    )

    """
    Then: The saved onnx model should pass onnx model checker
    """
    onnx_model = onnx.load_model(onnx_path)
    onnx.checker.check_model(onnx_model)

    """
    Then: Input/Output names should be strictly honored
    """
    assert list(x.name for x in onnx_model.graph.input) == ["input"]
    assert list(y.name for y in onnx_model.graph.output) == ["output"]

    """
    Then: Model should contain expected number of DequantizedLinear nodes
    """
    onnx_dq_nodes = [
        node for node in onnx_model.graph.node if node.op_type == "DequantizeLinear"
    ]
    # Exported onnx qdq model can contain MORE qdq nodes than quantsim
    # as data movement op's output encodings that are generated
    # on-the-fly during export
    onnx_weight_names = set(
        convfc.input[1]
        for convfc in onnx_model.graph.node
        if convfc.op_type in ("Conv", "Gemm")
    )
    onnx_bias_names = set(
        convfc.input[2]
        for convfc in onnx_model.graph.node
        if convfc.op_type in ("Conv", "Gemm") and len(convfc.input) > 2
    )
    quantized_param_names = (
        onnx_weight_names | onnx_bias_names if export_int32_bias else onnx_weight_names
    )
    assert len(onnx_dq_nodes) >= len(activation_qdq_nodes) + len(quantized_param_names)

    """
    Then: All model input/outputs should be associated with QDQ
    """
    input_names = set(inp.name for inp in onnx_model.graph.input)
    output_names = set(out.name for out in onnx_model.graph.output)
    for node in onnx_model.graph.node:
        if node.input and node.input[0] in input_names:
            assert node.op_type == "QuantizeLinear"
            input_names.remove(node.input[0])
        if node.output and node.output[0] in output_names:
            assert node.op_type == "DequantizeLinear"
            output_names.remove(node.output[0])
    assert not input_names
    assert not output_names

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {"input": x.numpy()})

    # Allow off-by-3 error
    atol = sim.model.fc.output_quantizers[0].get_scale().item() * 3
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


@pytest.mark.skip()
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_non_float32_qdq_export(tmp_path, dtype):
    x = torch.randn(1, 3, 32, 32).to(dtype)
    model = test_models.SingleResidual().to(dtype)

    sim = QuantizationSimModel(model, x, default_param_bw=8, default_output_bw=8)

    sim.compute_encodings(lambda model: model(x))
    onnx_path = os.path.join(tmp_path, "model.onnx")
    with pytest.raises(RuntimeError):
        aimet_torch.onnx.export(
            sim,
            x,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )


@pytest.mark.parametrize("target_opset", range(_constants.ONNX_MIN_OPSET, 22))
@pytest.mark.parametrize(
    "param_bw, act_bw, per_channel, minimum_required_opset",
    [
        (4, 8, False, 21),
        (4, 16, False, 21),
        (8, 8, False, 10),
        (8, 16, False, 21),
        (16, 16, False, 21),
        (4, 8, False, 21),
        (4, 16, True, 21),
        (8, 8, True, 13),
        (8, 16, True, 21),
        (16, 16, True, 21),
    ],
)
def test_minimum_opset(
    param_bw: int,
    act_bw: int,
    per_channel: bool,
    minimum_required_opset: int,
    target_opset: int,
):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(10, 10, 3),
        torch.nn.ReLU(),
    )
    x = torch.randn(1, 10, 224, 224)
    config_file = "htp_v81" if per_channel else get_path_for_per_tensor_config()
    sim = QuantizationSimModel(
        model,
        x,
        default_param_bw=param_bw,
        default_output_bw=act_bw,
        config_file=config_file,
    )
    sim.compute_encodings(lambda model: model(x))

    expected_out = sim.model(x)
    atol = 1 * sim.model[-1].output_quantizers[0].get_scale().item()

    with tempfile.TemporaryDirectory() as tmpdir:
        full_path = os.path.join(tmpdir, "model.onnx")

        if 9 <= target_opset <= _constants.ONNX_MAX_OPSET:
            # sim.onnx.export (onnx + json export) should always work
            sim.onnx.export(
                x,
                f=full_path,
                opset_version=target_opset,
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                dynamo=False,
            )

        if target_opset < minimum_required_opset:
            """
            When: target opset version < minimum required version
            Then: Throw runtime error
            """
            with pytest.raises(RuntimeError):
                aimet_torch.onnx.export(
                    sim,
                    x,
                    f=full_path,
                    opset_version=target_opset,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                    dynamo=False,
                )
            return

        """
        When: aimet_torch.onnx.export with specific target opset version
        """
        aimet_torch.onnx.export(
            sim.model,
            x,
            f=full_path,
            opset_version=target_opset,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )

        """
        Then: Exported onnx model's opset should be equal to the target opset version
        """
        onnx_qdq_model = onnx.load_model(full_path)
        assert onnx_qdq_model.opset_import[0].version == target_opset

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        sess = ort.InferenceSession(
            onnx_qdq_model.SerializeToString(),
            providers=["CPUExecutionProvider"],
            sess_options=sess_options,
        )
        (out,) = sess.run(None, {"input": x.detach().numpy()})
        assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"opset_version": 22},
        {"export_params": False},
        {"keep_initializers_as_inputs": True},
        {"dynamo": True},
        {"do_constant_folding": False},
        {"export_modules_as_functions": True},
        {"operator_export_type": torch.onnx.OperatorExportTypes.ONNX_ATEN},
    ],
)
def test_unsupported_args(kwargs):
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    x = torch.zeros(10, 10)
    sim = QuantizationSimModel(model, x)

    if "dynamo" not in kwargs:
        kwargs["dynamo"] = False

    with pytest.raises((ValueError, RuntimeError, NotImplementedError)):
        aimet_torch.onnx.export(sim.model, x, f=os.devnull, **kwargs)


@pytest.mark.parametrize("dynamo", [False, True])
def test_non_standard_quantizer(dynamo: bool):
    """
    When: Export model with non-standard-bitwidth quantizer
    Then: Should throw RuntimeError
    """
    model = torch.nn.Sequential(torch.nn.Linear(16, 16))
    x = torch.zeros(16, 16)
    sim = QuantizationSimModel(model, x)
    sim.model[0].param_quantizers["weight"].bitwidth = 9

    with pytest.raises(RuntimeError):
        aimet_torch.onnx.export(sim.model, x, f=os.devnull, dynamo=dynamo)


@pytest.mark.parametrize("dynamo", [False, True])
def test_data_movement_op_encoding_generation(dynamo: bool):
    """
    Given: Model with data movement ops
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x):
            x = self.conv(x)
            x = x.reshape(1, -1)
            return x[:, -10:]

    """
    When Export to onnx QDQ
    """
    model = Model()
    x = torch.randn(1, 3, 224, 224)
    sim = QuantizationSimModel(model, x)
    sim.compute_encodings(lambda model: model(x))

    with tempfile.TemporaryDirectory() as tmpdir:
        full_path = os.path.join(tmpdir, "model.onnx")
        aimet_torch.onnx.export(
            sim.model,
            x,
            full_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=dynamo,
        )
        onnx_model = onnx.load_model(full_path)

    with open("/tmp/onnx_reshape_qdq.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    """
    Then: All model input/outputs should be associated with QDQ
    """
    input_names = set(inp.name for inp in onnx_model.graph.input)
    output_names = set(out.name for out in onnx_model.graph.output)
    for node in onnx_model.graph.node:
        if node.input and node.input[0] in input_names:
            assert node.op_type == "QuantizeLinear"
            input_names.remove(node.input[0])
        if node.output and node.output[0] in output_names:
            assert node.op_type == "DequantizeLinear"
            output_names.remove(node.output[0])
    assert not input_names
    assert not output_names

    """
    Then: ORT output should be EQUAL with/without data movement op output QDQ
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        full_path = os.path.join(tmpdir, "model.onnx")
        with patch(
            "aimet_torch.onnx._derive_data_movement_op_encodings", lambda *_: {}
        ):
            aimet_torch.onnx.export(
                sim.model,
                x,
                full_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                dynamo=dynamo,
            )
        onnx_model_ = onnx.load_model(full_path)
        # patch sanity check
        assert len(onnx_model.graph.node) > len(onnx_model_.graph.node)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(), sess_options=sess_options
    )
    sess_ = ort.InferenceSession(
        onnx_model_.SerializeToString(), sess_options=sess_options
    )

    for _ in range(10):
        x = torch.randn(5, 3, 224, 224).detach().numpy()
        (output,) = sess.run(None, {"input": x})
        (output_,) = sess_.run(None, {"input": x})
        assert np.all(output == output_)


def test_data_movement_op_encoding_generation_edge_case():
    """
    Given:
                                                          +--> QDQ
      input -> Relu -+-> Reshape -> QDQ --> Add -> Split -+
                     +-> Sigmoid ------------^            +--> ...
    """
    model = helper.make_model(
        opset_imports=[helper.make_operatorsetid("", 21)],
        graph=helper.make_graph(
            name="reshape_with_multiple_consumers",
            inputs=[
                helper.make_tensor_value_info(
                    "input", TensorProto.FLOAT, shape=[3, 1024]
                ),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "split_output_0", TensorProto.FLOAT, shape=[1, 3, 512]
                ),
                helper.make_tensor_value_info(
                    "split_output_1", TensorProto.FLOAT, shape=[1, 3, 512]
                ),
            ],
            nodes=[
                helper.make_node(
                    "Relu",
                    inputs=["input"],
                    outputs=["relu_output"],
                    name="relu",
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["shape"],
                    name="shape",
                    value_ints=[1, 3, 1024],
                ),
                helper.make_node(
                    "Reshape",
                    inputs=["relu_output", "shape"],
                    outputs=["reshape_output"],
                    name="reshape",
                ),
                helper.make_node(
                    "Sigmoid",
                    inputs=["relu_output"],
                    outputs=["sigmoid_output"],
                    name="sigmoid",
                ),
                helper.make_node(
                    "Add",
                    inputs=["reshape_output", "sigmoid_output"],
                    outputs=["add_output"],
                    name="add",
                ),
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=["splits"],
                    name="Constant_0",
                    value_ints=[512, 512],
                ),
                helper.make_node(
                    "Split",
                    inputs=["add_output", "splits"],
                    outputs=["split_output_0", "split_output_1"],
                    axis=-1,
                    name="split",
                ),
            ],
        ),
    )
    onnx.checker.check_model(model, True)

    """
    When: Call _derive_data_movement_op_encodings
    Then: Output encodings should not be reused for input quantization
    """
    new_encodings = _derive_data_movement_op_encodings(
        model,
        {
            "reshape_output": Q.affine.AffineEncoding(
                torch.ones(()), torch.zeros(()), qmin=0, qmax=255, symmetry=False
            ).to_qnn_encoding_dict("2.0.0"),
            "split_output_0": Q.affine.AffineEncoding(
                torch.ones(()), torch.zeros(()), qmin=0, qmax=255, symmetry=False
            ).to_qnn_encoding_dict("2.0.0"),
        },
    )

    assert not new_encodings


@pytest.mark.parametrize("dynamo", [False, True])
def test_back_to_back_qdq(tmp_path: pathlib.Path, dynamo: bool):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.softmax = torch.nn.Softmax()

        def forward(self, x):
            x = self.linear(x)
            return self.softmax(x)

    """
    Given: Sim that contains back-to-back qdq
    """
    input = torch.randn(100, 10)
    model = Model()
    sim = aimet_torch.QuantizationSimModel(
        model,
        input,
        default_param_bw=8,
        default_output_bw=8,
        config_file="htp_v81",
    )
    sim.model.softmax.input_quantizers[0] = Q.affine.QuantizeDequantize(
        shape=(), bitwidth=16, symmetric=False
    )

    sim.compute_encodings(lambda model: model(input))

    """
    When: Export to onnx QDQ
    Then: Raises NotImplementedError
    """
    aimet_torch.onnx.export(
        sim.model,
        input,
        tmp_path / "qdq_model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
        dynamo=dynamo,
    )

    # TODO: Uncomment this when AIMET begins to support exporting back-to-back QDQ
    """
    Then: onnx graph should look like this:

        weight -> QDQ ---V
        input --> QDQ -> Gemm -> QDQ ----> QDQ -> Softmax -> QDQ -> output
        bias_q -> DQ ----^     (8-bit)   (16-bit)
    """
    onnx_model = onnx.load_model(tmp_path / "qdq_model.onnx")
    num_dq = len(
        [dq for dq in onnx_model.graph.node if dq.op_type == "DequantizeLinear"]
    )
    assert num_dq == 6, f"Expected 6 DequantizeLinear nodes, but got {num_dq}"

    """
    Given: Sim that contains redundant back-to-back qdq
    """
    input = torch.randn(100, 10)
    model = Model()
    sim = aimet_torch.QuantizationSimModel(
        model,
        input,
        default_param_bw=8,
        default_output_bw=8,
        config_file="htp_v81",
    )
    sim.model.softmax.input_quantizers[0] = copy.deepcopy(
        sim.model.linear.output_quantizers[0]
    )
    sim.compute_encodings(lambda model: model(input))

    """
    When: Export to onnx QDQ
    Then:
      1. Should be exported normally
      2. The redundant back-to-back QDQs should be consolidated into one QDQ

        weight -> QDQ ---V
        input --> QDQ -> Gemm -> QDQ -------> QDQ -> Softmax -> QDQ -> output
        bias_q -> DQ ----^       <-consolidated->
    """
    aimet_torch.onnx.export(
        sim.model,
        input,
        tmp_path / "qdq_model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
        dynamo=dynamo,
    )
    onnx_model = onnx.load_model(tmp_path / "qdq_model.onnx")
    num_dq = len(
        [dq for dq in onnx_model.graph.node if dq.op_type == "DequantizeLinear"]
    )
    assert num_dq == 5, f"Expected 5 DequantizeLinear nodes, but got {num_dq}"

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": input.detach().numpy()})

    with torch.no_grad():
        expected_out = sim.model(input)

    atol = sim.model.softmax.output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


@torch.no_grad()
@pytest.mark.parametrize("opset_version", [19, 21])
@pytest.mark.parametrize("prequantize_constants", [False, True])
def test_export_external_data(
    opset_version: int,
    prequantize_constants: bool,
    tmp_path: pathlib.Path,
):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 100, bias=False)
            self.scale = torch.nn.Parameter(torch.ones(100))

        def forward(self, x):
            return self.linear(x) * self.scale

    x = torch.randn(1, 10)
    model = Model()
    sim = QuantizationSimModel(model, x, config_file="htp_v81")
    sim.compute_encodings(lambda model: model(x))

    onnx_path = os.path.join(tmp_path, "qdq_model.onnx")

    """
    When: Call sim.onnx.export with external_data=True
    Then: All encoding should be exported correctly
    """
    sim.onnx.export(
        x,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=True,
        external_data=True,
        opset_version=opset_version,
        encoding_version="2.0.0",
    )

    assert os.path.exists(os.path.join(tmp_path, "qdq_model.onnx.data"))
    with open(os.path.join(tmp_path, "qdq_model.encodings")) as f:
        encodings = json.load(f)["encodings"]

    quantizers = [
        q for q in sim.model.modules() if isinstance(q, Q.affine.AffineQuantizerBase)
    ]

    for e in encodings:
        y_scale = e["y_scale"]
        assert any(
            np.allclose(y_scale, q.get_scale().numpy().flatten()) for q in quantizers
        )

    """
    When: Call aimet_torch.onnx.export with external_data=True
    Then: ONNX model should produce same output as sim
    """
    aimet_torch.onnx.export(
        sim,
        x,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset_version,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        prequantize_constants=prequantize_constants,
        dynamo=True,
        external_data=True,
    )
    assert os.path.exists(os.path.join(tmp_path, "qdq_model.onnx.data"))

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": x.detach().numpy()})

    with torch.no_grad():
        expected_out = sim.model(x)

    atol = sim.model.linear.output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


@torch.no_grad()
def test_fold_linear_weight_transpose_preserves_external_data(tmp_path: pathlib.Path):
    """
    _fold_linear_weight_transpose creates a transposed weight copy via
    onnx.numpy_helper.from_array, which always materializes it as inline raw_data
    even when the source weight was external. This test checks that copy is offloaded
    back to external data.

    dynamo=True + external_data=True forces the weight external regardless of size
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 20, bias=False)

        def forward(self, x):
            return self.linear(x)

    x = torch.randn(1, 4, 10)
    model = Model()
    sim = QuantizationSimModel(model, x, config_file="htp_v81")
    sim.compute_encodings(lambda m: m(x))

    onnx_path = os.path.join(tmp_path, "qdq_model.onnx")
    aimet_torch.onnx.export(
        sim,
        x,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
        dynamo=True,
        external_data=True,
    )

    # external_data=True must produce an external data file.
    assert os.path.exists(onnx_path + ".data")

    onnx_model = onnx.load(onnx_path, load_external_data=False)

    # Match the folded weight by its transposed [10, 20] shape (nn.Linear stores it
    # as [20, 10]): this both picks the right tensor and guards against the test
    # passing vacuously, since an unfolded graph has no [10, 20] weight to find.
    folded_weights = [
        i for i in onnx_model.graph.initializer if list(i.dims) == [10, 20]
    ]
    assert len(folded_weights) == 1, (
        "Expected exactly one transposed [10, 20] weight -- "
        "_fold_linear_weight_transpose path not exercised; "
        f"got {[(i.name, list(i.dims)) for i in folded_weights]}"
    )
    (folded_weight,) = folded_weights

    # The core fix: the folded weight must remain external data (not re-inlined).
    assert folded_weight.data_location == onnx.TensorProto.EXTERNAL, (
        f"Folded weight '{folded_weight.name}' should remain external data"
    )
    assert not folded_weight.HasField("raw_data"), (
        f"Folded weight '{folded_weight.name}' should have no inline raw_data"
    )


@pytest.mark.parametrize("dynamo", [False, True])
def test_output_split(tmp_path, dynamo: bool):
    """
    Given:
      Model with an output that is split into multiple consumers:

      Op1 ------+-----------> (output)
                |
                +---> Op2 --> ...

    When: Export to onnx QDQ
    Then: Should export successfully as below

      Op1 ---> QDQ ---------> (output)
                |
                +---> Op2 --> ...
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
            self.softmax = torch.nn.Softmax()

        def forward(self, x):
            y = self.linear(x)
            return y, self.softmax(y)

    model = Model()
    x = torch.randn(100, 10)
    sim = aimet_torch.QuantizationSimModel(model, x, config_file="htp_v81")
    sim.compute_encodings(lambda model: model(x))

    aimet_torch.onnx.export(
        sim.model,
        x,
        f=tmp_path / "model.onnx",
        dynamo=dynamo,
        input_names=["input"],
        output_names=["output1", "output2"],
    )
    onnx_model = onnx.load_model(tmp_path / "model.onnx")
    onnx.checker.check_model(onnx_model)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out1, out2) = sess.run(None, {"input": x.detach().numpy()})
    with torch.no_grad():
        expected_out1, expected_out2 = sim.model(x)

    atol1 = sim.model.linear.output_quantizers[0].get_scale().item()
    atol2 = sim.model.softmax.output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out1), expected_out1, atol=atol1)
    assert torch.allclose(torch.from_numpy(out2), expected_out2, atol=atol2)


@torch.no_grad()
@pytest.mark.parallel
@pytest.mark.parametrize("prequantize_constants", [False, True])
@pytest.mark.parametrize(
    "compile, dynamo",
    [
        (False, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize("zero_point_shift", [0.0, 0.5])
def test_quantsim_export_int2(
    tmp_path: pathlib.Path,
    zero_point_shift: float,
    dynamo: bool,
    compile: bool,
    prequantize_constants: bool,
):
    """
    When: Export quantized model with int2 weights using sim.onnx.export
    Then: The exported weight encoding's y_zero_point should be equal to -zero_point_shift
    """
    if compile and version.parse(torch.__version__) < version.parse("2.11.0.dev"):
        pytest.skip(
            reason="Exporting torch.compile-d model is only supported in torch >= 2.11.0"
        )

    model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3))
    x = torch.randn(1, 3, 32, 32)
    sim = QuantizationSimModel(model, x, default_param_bw=2)
    sim.model[0].param_quantizers["weight"].zero_point_shift = zero_point_shift
    sim.compute_encodings(lambda model: model(x))

    if compile:
        sim.model = torch.compile(sim.model)

    sim.onnx.export(
        x,
        tmp_path / "int2_conv.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
        encoding_version="2.0.0",
    )

    with open(tmp_path / "int2_conv.encodings") as f:
        encodings = json.load(f)["encodings"]

    weight_encoding = next(
        e
        for e in encodings
        if e["name"] == ("_orig_mod.0.weight" if compile else "0.weight")
    )
    y_zero_point = weight_encoding.get("y_zero_point", 0)
    assert np.all(np.array(y_zero_point) == -zero_point_shift)

    aimet_torch.onnx.export(
        sim.model,
        x,
        tmp_path / "int2_conv_qdq.onnx",
        opset_version=25,
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
        prequantize_constants=prequantize_constants,
    )
    onnx_qdq_model = onnx.load_model(tmp_path / "int2_conv_qdq.onnx")
    onnx.checker.check_model(onnx_qdq_model)

    q_nodes = [
        node for node in onnx_qdq_model.graph.node if node.op_type == "QuantizeLinear"
    ]
    dq_nodes = [
        node for node in onnx_qdq_model.graph.node if node.op_type == "DequantizeLinear"
    ]
    if prequantize_constants:
        assert len(q_nodes) == 2
        assert len(dq_nodes) == 4
    else:
        assert len(q_nodes) == 3
        assert len(dq_nodes) == 4

    for node in onnx_qdq_model.graph.node:
        if node.op_type != "DequantizeLinear":
            continue

        scale_name, zp_name = node.input[1:3]

        scale_array = onnx.numpy_helper.to_array(
            next(
                init
                for init in onnx_qdq_model.graph.initializer
                if init.name == scale_name
            )
        )
        zp_array = onnx.numpy_helper.to_array(
            next(
                init
                for init in onnx_qdq_model.graph.initializer
                if init.name == zp_name
            )
        )
        if node.output == "0.weight_qdq":
            expected_scale = (
                sim.model[0].weight.encoding.scale
                if fold_param_quantizers
                else sim.model[0].param_quantizers["weight"].get_scale()
            )
            expected_zp = 0
        elif node.input == "input_qdq":
            expected_scale = sim.model[0].input_quantizers[0].get_scale()
            expected_zp = -sim.model[0].input_quantizers[0].get_offset()
        elif node.output == "output":
            expected_scale = sim.model[0].output_quantizers[0].get_scale()
            expected_zp = -sim.model[0].output_quantizers[0].get_offset()
        else:
            continue

        assert torch.allclose(
            torch.from_numpy(scale_array).reshape(expected_scale.shape),
            expected_scale,
        )
        assert np.all(zp_array == expected_zp)

    if zero_point_shift == 0.0:
        return

    """
    When: Export model with absorbed zero_point_shift using aimet_torch.onnx.export
    Then:
      1. The exported weight tensor should only consist of {-3, -1, 1, 3}
      2. The exported onnx model should produce same output as sim
    """
    out = sim.model(x)
    aimet_torch.onnx._absorb_zero_point_shift(sim.model)
    out2 = sim.model(x)
    assert torch.equal(out, out2)

    if compile:
        weight_qtzr = sim.model._orig_mod[0].param_quantizers["weight"]
        weight = sim.model._orig_mod[0].weight
    else:
        weight_qtzr = sim.model[0].param_quantizers["weight"]
        weight = sim.model[0].weight

    w_int4 = weight_qtzr(weight).quantize()
    assert torch.all((w_int4 == -3) | (w_int4 == -1) | (w_int4 == 1) | (w_int4 == 3))

    aimet_torch.onnx.export(
        sim.model,
        x,
        tmp_path / "int2_conv_qdq.onnx",
        opset_version=21,
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
        prequantize_constants=prequantize_constants,
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        tmp_path / "int2_conv_qdq.onnx", sess_options=sess_options
    )
    (out_onnx,) = sess.run(None, {"input": x.numpy()})

    if compile:
        atol = sim.model._orig_mod[0].output_quantizers[0].get_scale().item()
    else:
        atol = sim.model[0].output_quantizers[0].get_scale().item()

    assert torch.allclose(torch.from_numpy(out_onnx), out2, atol=atol)


@torch.no_grad()
@pytest.mark.parametrize("dynamo", [False, True])
@pytest.mark.parametrize("lpbq", [False, True])
def test_1x1_conv_bq(tmp_path: pathlib.Path, lpbq: bool, dynamo: bool):
    """
    When: Export quantized model with 1x1 conv using aimet_torch.onnx.export
    Then: The exported onnx model should produce output close enough to
          the original pytorch model with qdq weights
    """
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, bias=False)
    )
    dummy_input = torch.randn(1, 16, 100, 100)

    sim = aimet_torch.QuantizationSimModel(model, dummy_input=dummy_input)
    if lpbq:
        set_grouped_blockwise_quantization_for_weights(
            sim,
            [torch.nn.Conv2d],
            bitwidth=4,
            symmetric=True,
            decompressed_bw=8,
            block_size=4,
        )
    else:
        set_blockwise_quantization_for_weights(
            sim, [torch.nn.Conv2d], bitwidth=4, symmetric=True, block_size=4
        )

    sim.compute_encodings(lambda model: model(dummy_input))
    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        tmp_path / "lpbq_conv1x1.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
        opset_version=21,
    )

    out_sim = sim.model(dummy_input)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        tmp_path / "lpbq_conv1x1.onnx", sess_options=sess_options
    )
    (out_onnx,) = sess.run(None, {"input": dummy_input.numpy()})
    atol = sim.model[0].output_quantizers[0].get_scale().item()

    assert torch.allclose(torch.from_numpy(out_onnx), out_sim, atol=atol)


@torch.no_grad()
@pytest.mark.parametrize("dynamo", [False, True])
def test_duplicate_qdq_input(tmp_path, dynamo: bool):
    """
    Given: Same input tensor associated with multiple QDQ nodes

                     +-----> aimet::QuantizeDequantize ------> out0
        Relu --------+
                ↑    +-----> aimet::QuantizeDequantize ------> out1
                |
           "/Relu_output_0"

    When: Export to onnx QDQ
    Then: There should be no tensor that feeds into multiple QuantizeLinear/DequantizeLinear nodes
          in the exported onnx graph

                                  "/Relu_output_0_dup_0"
                                      ↓
                     +-----> Identity -> QuantizeLinear -> DequantizeLinear ------> out0
        Relu --------+
                ↑    +-----> Identity -> QuantizeLinear -> DequantizeLinear ------> out1
                |                     ↑
           "/Relu_output_0"       "/Relu_output_0_dup_1"
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.qdq2 = Q.affine.QuantizeDequantize(
                (), qmin=0, qmax=255, symmetric=False
            )
            self.qdq0 = Q.affine.QuantizeDequantize(
                (), qmin=0, qmax=255, symmetric=False
            )
            self.qdq1 = Q.affine.QuantizeDequantize(
                (), qmin=0, qmax=255, symmetric=False
            )

        def forward(self, x):
            x = torch.nn.functional.relu(x)
            y0 = self.qdq0(x)
            y1 = self.qdq1(x)
            return y0.flatten(), y1.flatten()

    model = Model()
    x = torch.randn(1, 10)
    model.qdq0.set_range(-1.0, 1.0)
    model.qdq1.set_range(0.0, 1.0)
    aimet_torch.onnx.export(
        model,
        (x,),
        tmp_path / "duplicate_qdq_input.onnx",
        input_names=["input"],
        output_names=["output_0", "output_1"],
        dynamo=dynamo,
    )

    onnx_model = onnx.load(tmp_path / "duplicate_qdq_input.onnx")
    onnx.checker.check_model(onnx_model)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(), sess_options=sess_options
    )
    ort_out0, ort_out1 = sess.run(None, {"input": x.numpy()})
    sim_out0, sim_out1 = model(x)
    assert np.allclose(
        ort_out0, sim_out0.detach().numpy(), atol=model.qdq0.get_scale().item()
    )
    assert np.allclose(
        ort_out1, sim_out1.detach().numpy(), atol=model.qdq1.get_scale().item()
    )


@pytest.mark.skipif(
    not Q.affine.backends.triton.is_available(),
    reason="Triton backend not available",
)
@pytest.mark.parametrize("dynamo", [False, True])
@pytest.mark.parametrize("export_int32_bias", [False, True])
@pytest.mark.parametrize("fold_param_quantizers", [False, True])
def test_triton(
    tmp_path: pathlib.Path,
    dynamo: bool,
    export_int32_bias: bool,
    fold_param_quantizers: bool,
):
    """
    When: Export to onnx QDQ with torch_builtins and triton backends
    Then: The exported onnx models should be identical
    """
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 3, 3),
        torch.nn.ReLU(),
    )
    dummy_input = torch.randn(1, 3, 32, 32)
    sim = aimet_torch.QuantizationSimModel(model, dummy_input, config_file="htp_v81")
    sim.compute_encodings(lambda model: model(dummy_input))

    if fold_param_quantizers:
        sim.fold_param_quantizers()

    with Q.affine.set_backend("torch_builtins"):
        aimet_torch.onnx.export(
            sim.model,
            dummy_input,
            tmp_path / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=21,
            dynamo=dynamo,
            export_int32_bias=export_int32_bias,
        )
        torch_builtin_export = onnx.load(tmp_path / "model.onnx")

    with Q.affine.set_backend("triton"):
        aimet_torch.onnx.export(
            sim.model,
            dummy_input,
            tmp_path / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=21,
            dynamo=dynamo,
            export_int32_bias=export_int32_bias,
        )
        triton_export = onnx.load(tmp_path / "model.onnx")

    assert torch_builtin_export == triton_export


@pytest.mark.parametrize("force_activation_as", ["unsigned", "signed", None])
def test_activation_uint(tmp_path: pathlib.Path, force_activation_as: str | None):
    """
    Given: Model with symmetric activation encoding
    When: Export to onnx QDQ
    Then: All activation encodings in the exported onnx model should be uint
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mm = aimet_torch.nn.modules.custom.MatMul()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x, y):
            output = self.mm(x, y)
            return self.linear(output)

    dummy_input = (torch.randn(10, 10), torch.randn(10, 10))
    sim = QuantizationSimModel(
        Model(), dummy_input, default_output_bw=16, config_file="htp_v81"
    )
    # sanity check
    assert not sim.model.mm.input_quantizers[0].symmetric
    assert sim.model.mm.input_quantizers[1].symmetric
    assert not sim.model.mm.output_quantizers[0].symmetric

    sim.compute_encodings(lambda m: m(*dummy_input))

    sim.onnx.export(
        dummy_input,
        tmp_path / "model.onnx",
        input_names=["x", "y"],
        output_names=["output"],
        dynamo=False,
        encoding_version="2.0.0",
        force_activation_as=force_activation_as,
        export_int32_bias=True,
    )
    with open(tmp_path / "model.encodings", "r") as f:
        encodings = json.load(f)

    for enc in encodings["encodings"]:
        if enc["name"] == "linear.weight":
            expected_dtype = "int8"
        elif enc["name"] == "linear.bias":
            expected_dtype = "int32"
        elif force_activation_as == "unsigned":
            expected_dtype = "uint16"
        elif force_activation_as == "signed":
            expected_dtype = "int16"
        else:
            expected_dtype = "int16" if enc["name"] == "y" else "uint16"
        assert enc["output_dtype"] == expected_dtype, enc

    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        tmp_path / "model.onnx",
        opset_version=21,
        input_names=["x", "y"],
        output_names=["output"],
        dynamo=False,
        force_activation_as=force_activation_as,
        export_int32_bias=True,
    )

    onnx_model = onnx.load(tmp_path / "model.onnx")
    onnx.checker.check_model(onnx_model)

    initializers = {init.name: init for init in onnx_model.graph.initializer}
    for node in onnx_model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            zero_point = node.input[2]

            if node.input[0].startswith("linear.weight"):
                expected_dtype = onnx.TensorProto.INT8
            elif node.input[0].startswith("linear.bias"):
                expected_dtype = onnx.TensorProto.INT32
            elif force_activation_as == "unsigned":
                expected_dtype = TensorProto.UINT16
            elif force_activation_as == "signed":
                expected_dtype = TensorProto.INT16
            else:
                expected_dtype = (
                    TensorProto.INT16
                    if node.input[0] in ("y", "y_q")
                    else TensorProto.UINT16
                )

            assert initializers[zero_point].data_type == expected_dtype

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(), sess_options=sess_options
    )
    (ort_out,) = sess.run(
        None, {"x": dummy_input[0].numpy(), "y": dummy_input[1].numpy()}
    )
    sim_out = sim.model(*dummy_input)
    assert np.allclose(
        ort_out,
        sim_out.detach().numpy(),
        atol=sim.model.mm.output_quantizers[0].get_scale().item(),
    )


@torch.no_grad()
@pytest.mark.parallel
@pytest.mark.parametrize("dynamo", [False])
@pytest.mark.parametrize("prequantize_constants", [True, False])
@pytest.mark.parametrize("fold_param_quantizers", [True, False])
@pytest.mark.parametrize(
    "finfo",
    [
        _float8_e5m2,
        _float8_e5m2fnuz,
        _float8_e4m3fn,
        _float8_e4m3fnuz,
        _float4_e2m1fn,
    ],
)
@pytest.mark.parametrize(
    "shape, block_size, channel_axis, block_axis",
    [
        [(), None, None, None],  # per-tensor
        [(1,), None, None, None],  # per-tensor
        [(10,), None, 1, None],  # per-channel
        [(1, 10), None, 1, None],  # per-channel
        [(10, 1), None, 0, None],  # per-channel
        [(10, 2), (-1, 5), 0, 1],  # blockwise
    ],
)
def test_export_float8_and_float4(
    shape: tuple[int, ...],
    finfo: _finfo,
    block_size: tuple[int, ...] | None,
    channel_axis: int | None,
    block_axis: int | None,
    fold_param_quantizers: bool,
    prequantize_constants: bool,
    dynamo: bool,
    tmp_path: pathlib.Path,
):
    """
    When: Export quantized model with float8 encodings using sim.onnx.export
    Then: The exported encodings should match
    """
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    x = torch.randn(10, 10)

    sim = aimet_torch.QuantizationSimModel(model, x)
    sim.model[0].input_quantizers[0] = Q.float.FloatQuantizeDequantize(*finfo)
    sim.model[0].output_quantizers[0] = Q.float.FloatQuantizeDequantize(*finfo)
    sim.model[0].param_quantizers["weight"] = Q.float.FloatQuantizeDequantize(
        *finfo,
        shape=shape,
        block_size=block_size,
    )
    sim.compute_encodings(lambda model: model(x))

    if fold_param_quantizers:
        sim.fold_param_quantizers()

    for encoding_version in ["0.6.1", "1.0.0"]:
        # Old encoding versions can't support float8/float4 encodings
        with pytest.raises(RuntimeError):
            sim.onnx.export(
                (x,),
                tmp_path / "float8_linear.onnx",
                opset_version=19,
                input_names=["input"],
                output_names=["output"],
                dynamo=dynamo,
                encoding_version=encoding_version,
            )

    sim.onnx.export(
        (x,),
        tmp_path / f"{finfo.to_str()}_linear.onnx",
        opset_version=19,
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
        encoding_version="2.0.0",
    )

    with open(tmp_path / f"{finfo.to_str()}_linear.encodings") as f:
        encodings = json.load(f)["encodings"]

    _, expected_dtype = (
        helper.tensor_dtype_to_string(finfo.to_onnx_dtype()).lower().split(".")
    )
    for e in encodings:
        assert e["output_dtype"] == expected_dtype

        if e["name"] == "input":
            assert e.keys() == {"name", "y_scale", "output_dtype"}
            assert e["y_scale"] == sim.model[0].input_quantizers[0].get_scale().item()
        elif e["name"] == "output":
            assert e.keys() == {"name", "y_scale", "output_dtype"}
            assert e["y_scale"] == sim.model[0].output_quantizers[0].get_scale().item()
        elif e["name"] == "0.weight":
            if not shape or all(s == 1 for s in shape):
                assert e.keys() == {"name", "y_scale", "output_dtype"}
            elif block_size is None:
                assert e.keys() == {
                    "name",
                    "y_scale",
                    "output_dtype",
                    "axis",
                }
                assert e["axis"] == (
                    block_axis if block_axis is not None else channel_axis
                )
            else:
                assert e.keys() == {
                    "name",
                    "y_scale",
                    "output_dtype",
                    "axis",
                    "block_size",
                }
                assert e["axis"] == 1
                assert e["block_size"] == 5

            weight_scale = (
                sim.model[0].weight.encoding.scale
                if fold_param_quantizers
                else sim.model[0].param_quantizers["weight"].get_scale()
            )
            assert torch.equal(torch.tensor(e["y_scale"]).reshape(shape), weight_scale)

    aimet_torch.onnx.export(
        sim.model,
        (x,),
        tmp_path / f"{finfo.to_str()}_linear_qdq.onnx",
        opset_version=(
            23 if finfo == _float4_e2m1fn else 19 if block_size is None else 21
        ),
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
        prequantize_constants=prequantize_constants,
    )
    onnx_qdq_model = onnx.load_model(tmp_path / f"{finfo.to_str()}_linear_qdq.onnx")
    onnx.checker.check_model(onnx_qdq_model)

    q_nodes = [
        node for node in onnx_qdq_model.graph.node if node.op_type == "QuantizeLinear"
    ]
    dq_nodes = [
        node for node in onnx_qdq_model.graph.node if node.op_type == "DequantizeLinear"
    ]

    if prequantize_constants:
        assert len(q_nodes) == 2
        assert len(dq_nodes) == 3
    else:
        assert len(q_nodes) == len(dq_nodes) == 3

    for node in onnx_qdq_model.graph.node:
        if node.op_type != "DequantizeLinear":
            continue

        scale_name, zp_name = node.input[1:3]

        zp_array = onnx.numpy_helper.to_array(
            next(
                init
                for init in onnx_qdq_model.graph.initializer
                if init.name == zp_name
            )
        )
        assert (zp_array == 0).all()

        scale_array = onnx.numpy_helper.to_array(
            next(
                init
                for init in onnx_qdq_model.graph.initializer
                if init.name == scale_name
            )
        )
        if node.output == "0.weight_qdq":
            expected_scale = (
                sim.model[0].weight.encoding.scale
                if fold_param_quantizers
                else sim.model[0].param_quantizers["weight"].get_scale()
            )
        elif node.input == "input_qdq":
            expected_scale = sim.model[0].input_quantizers[0].get_scale()
        elif node.output == "output":
            expected_scale = sim.model[0].output_quantizers[0].get_scale()
        else:
            continue

        assert torch.allclose(
            torch.from_numpy(scale_array).reshape(expected_scale.shape),
            expected_scale,
        )

    if prequantize_constants:
        weight_q = onnx.numpy_helper.to_array(
            next(
                init
                for init in onnx_qdq_model.graph.initializer
                if init.name == "0.weight_q"
            )
        )
        weight = sim.model[0].weight

        if isinstance(weight, Q.DequantizedTensor):
            expected_weight_q = weight.quantize().detach().numpy()
        else:
            weight_qtzr = sim.model[0].param_quantizers["weight"]
            expected_weight_q = weight_qtzr(weight).quantize().detach().numpy()

        assert np.all(weight_q == expected_weight_q.astype(weight_q.dtype))
        # without downcasting, weight_q and expected_weight_q can slightly differ
        # like 128.0 vs. 127.999 due to floating point precision
        assert np.allclose(weight_q, expected_weight_q)

    if finfo == _float4_e2m1fn:
        # Onnxruntime doesn't support float4 yet
        return

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_qdq_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": x.detach().numpy()})
    expected_out = sim.model(x)
    assert torch.allclose(torch.from_numpy(out), expected_out)


@pytest.mark.parametrize("scheme", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("dynamo", [True, False])
def test_export_fp4_int8(tmp_path: pathlib.Path, scheme: str, dynamo: bool):
    """
    Given: Model with float4 DequantizedTensor weight
    When: Create quantsim with per-channel W8 and export to onnx QDQ
    Then: Exported onnx QDQ should have float4 encoding for the first weight QDQ
          and int8 encoding for the second weight QDQ, and both encodings should
          match the sim quantizers' encodings.
          Exported weight should be on fp4 grid, not int8 grid.
    """
    model = torch.nn.Linear(64, 64)
    x = torch.randn(64, 64)
    sim = aimet_torch.QuantizationSimModel(model, x, default_param_bw=8)

    if scheme == "mxfp4":
        # MXFP4 e8m0 scale
        sim.model.set_weight_quantizer_to_mxfp4_int8(block_size=16)
    else:
        # NVFP4 quantized scale & meta-scale
        scale_q = (torch.randint(1, 100, (64, 4)) / 100).to(torch.float8_e4m3fn)
        meta_scale = torch.tensor(0.1)
        sim.model.set_weight_quantizer_to_nvfp4_int8(scale_q, meta_scale)

    sim.compute_encodings(lambda model: model(x))

    sim.onnx.export(
        (x,),
        str(tmp_path / "float4_int8.onnx"),
        opset_version=23,
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
        encoding_version="2.1.0",
    )
    onnx.checker.check_model(onnx.load(tmp_path / "float4_int8.onnx"))

    with open(tmp_path / "float4_int8.encodings") as f:
        encodings = {e["name"]: e for e in json.load(f)["encodings"]}

    assert encodings.keys() == {
        "input",
        "output",
        "weight",
        "bias",
        (
            "float_quantize_dequantize_alias"
            if dynamo
            else "/weight/FloatQuantizeDequantize_output_0_alias"
        ),
    }

    fp4_weight_encoding = encodings["weight"]
    expected_fp4_encoding = sim.model.weight.encoding

    if scheme == "mxfp4":
        assert isinstance(expected_fp4_encoding, _MXFP4Encoding)
        assert torch.equal(
            torch.tensor(fp4_weight_encoding["y_scale"]).reshape(64, 4),
            expected_fp4_encoding.scale,
        )
    else:
        assert isinstance(expected_fp4_encoding, _NVFP4Encoding)
        assert torch.equal(
            torch.tensor(fp4_weight_encoding["y_scale"]["x"]).reshape(64, 4),
            scale_q.to(torch.float32),
        )
        assert torch.equal(
            torch.tensor(fp4_weight_encoding["y_scale"]["x_scale"]),
            expected_fp4_encoding.meta_scale,
        )

    assert fp4_weight_encoding["output_dtype"] == "float4e2m1"
    assert "y_zero_point" not in fp4_weight_encoding
    assert fp4_weight_encoding.get("axis") == 1
    assert fp4_weight_encoding.get("block_size") == 16

    int8_weight_encoding = encodings[
        "float_quantize_dequantize_alias"
        if dynamo
        else "/weight/FloatQuantizeDequantize_output_0_alias"
    ]
    expected_int8_encoding = sim.model.param_quantizers["weight"].get_encodings()

    assert int8_weight_encoding["output_dtype"] == "int8"
    assert torch.equal(
        torch.tensor(int8_weight_encoding["y_scale"]).reshape(64, 1),
        expected_int8_encoding.scale,
    )
    assert "y_zero_point" not in int8_weight_encoding
    assert int8_weight_encoding.get("axis") == 0
    assert "block_size" not in int8_weight_encoding

    aimet_torch.onnx.export(
        sim.model,
        (x,),
        tmp_path / "float4_int8_qdq.onnx",
        opset_version=23,
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
    )
    onnx_qdq_model = onnx.load_model(tmp_path / "float4_int8_qdq.onnx")
    onnx.checker.check_model(onnx_qdq_model)
    producers = {
        output: node for node in onnx_qdq_model.graph.node for output in node.output
    }
    consumers = {}
    for node in onnx_qdq_model.graph.node:
        for input in node.input:
            consumers.setdefault(input, []).append(node)

    constants = _get_all_constants(onnx_qdq_model)
    (fp4_q,) = consumers["weight"]
    scale_name, zp_name = fp4_q.input[1:3]

    if scheme == "mxfp4":
        scale_array = onnx.numpy_helper.to_array(constants[scale_name])
        assert torch.allclose(
            torch.from_numpy(scale_array).reshape(64, 4), expected_fp4_encoding.scale
        )
    else:
        dq = producers[scale_name]
        quantized_scale_name, meta_scale_name = dq.input[0:2]
        quantized_scale_array = onnx.numpy_helper.to_array(
            constants[quantized_scale_name]
        ).astype(np.float32)
        assert torch.allclose(
            torch.from_numpy(quantized_scale_array).reshape(64, 4),
            scale_q.to(torch.float32),
        )
        meta_scale_array = onnx.numpy_helper.to_array(constants[meta_scale_name])
        assert torch.allclose(
            torch.from_numpy(meta_scale_array), expected_fp4_encoding.meta_scale
        )

    zp_array = onnx.numpy_helper.to_array(constants[zp_name])
    assert (zp_array == 0).all()

    (int8_q,) = consumers["weight_qdq"]
    scale_name, zp_name = int8_q.input[1:3]
    scale_array = onnx.numpy_helper.to_array(constants[scale_name])
    assert torch.allclose(
        torch.from_numpy(scale_array).reshape(64, 1),
        expected_int8_encoding.scale,
    )
    zp_array = onnx.numpy_helper.to_array(constants[zp_name])
    assert (zp_array == 0).all()

    onnx_weight = onnx.numpy_helper.to_array(constants["weight"])
    onnx_weight = torch.from_numpy(onnx_weight)
    assert torch.allclose(
        expected_fp4_encoding.quantize_dequantize(onnx_weight), onnx_weight
    )


def test_control_flow_op_export(tmp_path: pathlib.Path):
    """
    Given: Model with control flow op (If, Loop, and Scan)
    When: Export to onnx QDQ
    Then: Should export successfully and the exported onnx model should be valid
    """

    class Model(torch.nn.Module):
        """
        torch.Tensor.squeeze(0) is a conditional operator which only takes
        effect if axis 0 is singleton axis. Assuming axis 0 is dynamic
        batch dimension, torch.Tensor.squeeze(0) will be exported to onnx as
        control flow operator If which looks like this:

        output = If(
            condition=Eq(Shape(input)[0], 1),
            then_branch=Squeeze(input, axes=0),
            else_branch=Identity(input),
        )
        """

        def __init__(self):
            super(Model, self).__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x):
            x = self.linear(x)
            return x.squeeze(0)

    model = Model()
    input = torch.randn(10, 10)
    sim = aimet_torch.QuantizationSimModel(model, input)
    sim.compute_encodings(lambda model: model(input))

    aimet_torch.onnx.export(
        sim.model,
        input,
        tmp_path / "squeeze.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )
    onnx_qdq_model = onnx.load(tmp_path / "squeeze.onnx")
    # Sanity check
    assert any(node.op_type == "If" for node in onnx_qdq_model.graph.node)
    onnx.checker.check_model(onnx_qdq_model)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_qdq_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": input.detach().numpy()})
    expected_out = sim.model(input)
    atol = sim.model.linear.output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


def test_concat(tmp_path: pathlib.Path):
    """
    Given: Concat with only output encoding but without input encoding
    When: Export to onnx QDQ
    Then: Exported concat inputs must reuse the output encoding
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = torch.nn.Conv2d(6, 6, 3, padding=1)

        def forward(self, img, input_uv):
            out = torch.cat((img, input_uv), dim=1)
            return self.conv(out)

    model = Model()
    img = torch.randn(1, 3, 224, 224)
    input_uv = torch.randn(1, 3, 224, 224)
    sim = aimet_torch.QuantizationSimModel(model, (img, input_uv))
    sim.compute_encodings(lambda model: model(img, input_uv))
    sim.onnx.export(
        (img, input_uv),
        tmp_path / "concat.onnx",
        input_names=["img", "input_uv"],
        output_names=["output"],
        dynamo=False,
        encoding_version="2.0.0",
    )

    with open(tmp_path / "concat.encodings") as f:
        encodings = {enc.pop("name"): enc for enc in json.load(f)["encodings"]}

    assert encodings.keys() == {
        "img",
        "input_uv",
        "/Concat_output_0",
        "conv.weight",
        "conv.bias",
        "output",
    }
    assert encodings["input_uv"] == encodings["img"] == encodings["/Concat_output_0"]


@pytest.mark.skipif(
    version.parse(torch.__version__) >= version.parse("2.12.0"),
    reason="Deduplication slowdown issue was resolved in PyTorch 2.12.0",
)
def test_disable_C_jit_pass_onnx_deduplicate_initializers(tmp_path: pathlib.Path):
    """
    Given: Model with shared parameters
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear1 = torch.nn.Linear(10, 10)
            self.linear2 = torch.nn.Linear(10, 10)
            # Make linear2's weight and bias share the same initializer with linear1
            self.linear2.weight = self.linear1.weight
            self.linear2.bias = self.linear1.bias

        def forward(self, x):
            return self.linear2(self.linear1(x))

    model = Model()
    x = torch.randn(1, 10)

    """
    When: Export to onnx QDQ with C-jit pass "onnx_deduplicate_initializers" disabled
    Then:
      1) Export should work normally
      2) The exported onnx model should have duplicated initializers for shared parameters
      3) The exported model should produce same output as sim
    """
    sim = aimet_torch.QuantizationSimModel(model, x)
    sim.compute_encodings(lambda model: model(x))

    # Temporarily patch the threshold to 0 to disable onnx_deduplicate_initializers pass
    with patch_attr(
        aimet_torch.experimental.onnx._export,
        "_LARGE_MODEL_THRESHOLD_NUM_NN_PARAMETER_OBJECTS",
        0,
    ):
        aimet_torch.onnx.export(
            sim.model,
            x,
            tmp_path / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
    onnx_qdq_model = onnx.load(tmp_path / "model.onnx")
    onnx.checker.check_model(onnx_qdq_model)

    initializer_names = set(init.name for init in onnx_qdq_model.graph.initializer)
    assert initializer_names >= {
        "linear1.weight",
        "linear1.bias_q",
        "linear1.weight_scale",
        "linear1.weight_zero_point",
        "linear1.bias_scale",
        "linear1.bias_zero_point",
        "linear2.weight",
        "linear2.bias_q",
        "linear2.weight_scale",
        "linear2.weight_zero_point",
        "linear2.bias_scale",
        "linear2.bias_zero_point",
    }

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_qdq_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": x.detach().numpy()})
    expected_out = sim.model(x)
    atol = sim.model.linear2.output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


def test_export_creates_directory_if_not_exists(tmp_path):
    """
    Given: A quantized model
    When: Export to a path where the parent directory does not exist
    Then: The directory should be created automatically and export should succeed
    """
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    x = torch.randn(1, 10)

    sim = aimet_torch.QuantizationSimModel(model, x)
    sim.compute_encodings(lambda model: model(x))

    # Create a nested path where intermediate directories don't exist
    nested_dir = tmp_path / "nested" / "subdir" / "deep"
    onnx_path = nested_dir / "model.onnx"

    assert not nested_dir.exists()

    aimet_torch.onnx.export(
        sim.model,
        x,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    assert nested_dir.exists()
    assert onnx_path.exists()

    # Verify the exported model is valid
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


@pytest.mark.parametrize("activation_bw", [8, 16])
@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: test_models.ModelWithPreparedConstRescale(3.0, divide=True),
        lambda: test_models.ModelWithPreparedConstRescale(3.0, divide=False),
        lambda: test_models.MatMulRescaleAddModel(divide=True),
        lambda: test_models.MatMulRescaleAddModel(divide=False),
        lambda: test_models.StandalonePreparedConstRescale(3.0, divide=True),
        lambda: test_models.StandalonePreparedConstRescale(3.0, divide=False),
        lambda: test_models.ModelWithFunctionalDiv(),
        lambda: test_models.DivWithDataMovement(),
        lambda: test_models.RescaleModelWithSharedScaleFactor(),
    ],
)
def test_aimet_torch_export_with_propagated_rescale_encodings(
    tmp_path, model_factory, activation_bw
):
    """
    Given: Model with constant scalar Mul/Div op with no output quantizer
    """
    model = model_factory()
    dummy_input = model.dummy_input()
    sim = QuantizationSimModel(
        model, dummy_input, default_output_bw=activation_bw, config_file="htp_v81"
    )
    # Disable rescale output quantizers (may be input quantizer of subsequent op)
    for module in sim.qmodules():
        if isinstance(module, (aimet_ops.Divide, aimet_ops.Multiply)):
            module.output_quantizers[0] = None
            module.input_quantizers[1] = None
        else:
            module.input_quantizers[0] = None
    sim.compute_encodings(lambda m: m(*dummy_input))
    sim_output = sim.model(*dummy_input)
    """
    When: Export model to onnx QDQ
    """
    fname = os.path.join(tmp_path, "model.onnx")
    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        fname,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        opset_version=21,
    )
    """
    Then: (1) Exported model is a valid onnx model
    """
    onnx_model = onnx.load(fname)
    onnx.checker.check_model(onnx_model)
    """
    Then: (2) onnx QDQ model output matches sim output within 1 output scale tolerance
    """
    encoding_map = _get_qdq_encoding_map(onnx_model)
    constants = _get_all_constants(onnx_model)
    output_name = onnx_model.graph.output[0].name
    model_output_scale, _ = encoding_map[output_name]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    onnx_input = {
        inp.name: dummy_input[i].detach().numpy()
        for i, inp in enumerate(onnx_model.graph.input)
    }
    (ort_out,) = sess.run(None, onnx_input)
    assert np.allclose(
        ort_out, sim_output.detach().numpy(), atol=model_output_scale.item()
    )
    """
    Then: (3) A QDQ op is inserted at the Mul/Div output
    """
    rescale_node = next(
        node for node in onnx_model.graph.node if node.op_type in ("Mul", "Div")
    )
    rescale_input = rescale_node.input[0]
    rescale_output = rescale_node.output[0]
    assert rescale_output in encoding_map
    """
    Then: (4) The scale of the inserted QDQ matches input_scale * scale_factor
    """
    assert rescale_input in encoding_map
    input_scale, input_zp = encoding_map[rescale_input]
    output_scale, output_zp = encoding_map[rescale_output]
    producers = {out: n for n in onnx_model.graph.node for out in n.output}
    # Propagate through QDQ to get constant factor
    const_factor_name = producers[producers[rescale_node.input[1]].input[0]].input[0]
    const_factor = onnx.numpy_helper.to_array(constants[const_factor_name])
    exp_out_scale = (
        input_scale / const_factor
        if rescale_node.op_type == "Div"
        else input_scale * const_factor
    )
    assert np.isclose(output_scale, exp_out_scale, rtol=1e-5)
    """
    Then: (5) The zero point and dtype of the inserted QDQ match the input encoding
    """
    assert np.array_equal(output_zp, input_zp)
    assert output_zp.dtype == input_zp.dtype
    """
    Then: (6) The scale factor has a QDQ encoding and incurs no quantization noise
    """
    assert const_factor_name in encoding_map
    factor_scale_arr, factor_zp_arr = encoding_map[const_factor_name]
    assert factor_zp_arr == 0
    assert np.round(const_factor / factor_scale_arr) * factor_scale_arr == const_factor


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("activation_bw", [8, 16])
@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: test_models.ModelWithPreparedConstRescale(3.0, divide=True),
        lambda: test_models.ModelWithPreparedConstRescale(3.0, divide=False),
        lambda: test_models.MatMulRescaleAddModel(divide=True),
        lambda: test_models.MatMulRescaleAddModel(divide=False),
        lambda: test_models.StandalonePreparedConstRescale(3.0, divide=True),
        lambda: test_models.StandalonePreparedConstRescale(3.0, divide=False),
        lambda: test_models.ModelWithFunctionalDiv(),
        lambda: test_models.DivWithDataMovement(),
        lambda: test_models.ModelWithReversedMulOrdering(),
        lambda: test_models.RescaleModelWithSharedScaleFactor(),
    ],
)
def test_sim_onnx_export_with_propagated_rescale_encodings(
    tmp_path, model_factory, activation_bw, dtype
):
    """
    Given: Model with constant scalar Mul/Div op with no output quantizer
    """
    model = model_factory().to(dtype)
    dummy_input = tuple(t.to(dtype) for t in model.dummy_input())
    sim = QuantizationSimModel(model, dummy_input, default_output_bw=activation_bw)
    # Disable rescale output quantizers
    for module in sim.qmodules():
        if isinstance(module, (aimet_ops.Divide, aimet_ops.Multiply)):
            module.output_quantizers[0] = None
            module.input_quantizers[1] = None
        else:
            module.input_quantizers[0] = None
    sim.compute_encodings(lambda m: m(*dummy_input))
    """
    When: Export model via sim.onnx.export
    """
    fname = os.path.join(tmp_path, "model.onnx")
    encoding_path = os.path.join(tmp_path, "model.encodings")
    sim.onnx.export(
        dummy_input,
        fname,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        encoding_version="2.0.0",
    )
    """
    Then: (1) Exported model is a valid onnx model
    """
    onnx_model = onnx.load(fname)
    onnx.checker.check_model(onnx_model)
    with open(encoding_path) as f:
        encodings = json.load(f)
    """
    Then: (2) The encoding file contains an encoding for the Mul/Div output tensor
    """
    rescale_node = next(
        node for node in onnx_model.graph.node if node.op_type in ("Mul", "Div")
    )
    rescale_output_name = rescale_node.output[0]
    encoding_dict = {enc["name"]: enc for enc in encodings["encodings"]}
    assert rescale_output_name in encoding_dict
    """
    Then: (3) The propagated encoding scale matches input_scale * scale_factor
    """
    constants = _get_all_constants(onnx_model)
    inp_idx, scale_idx = (0, 1) if rescale_node.input[1] in constants else (1, 0)
    input_encoding = encoding_dict[rescale_node.input[inp_idx]]
    output_encoding = encoding_dict[rescale_output_name]
    const_factor_name = rescale_node.input[scale_idx]
    const_factor = onnx.numpy_helper.to_array(constants[const_factor_name])
    if rescale_node.op_type == "Div":
        expected_scale = input_encoding["y_scale"] / const_factor.item()
    else:
        expected_scale = input_encoding["y_scale"] * const_factor.item()
    assert np.isclose(output_encoding["y_scale"], expected_scale, rtol=1e-5)
    """
    Then: (4) The zero point is preserved in the output encoding
    """
    assert output_encoding.get("y_zero_point") == input_encoding.get("y_zero_point")
    """
    Then: (5) The dtype is preserved in the output encoding
    """
    assert output_encoding.get("dtype") == input_encoding.get("dtype")
    """
    Then: (6) The encoding file contains an encoding for the scale factor
    """
    assert const_factor_name in encoding_dict
    factor_encoding = encoding_dict[const_factor_name]
    """
    Then: (7) The factor encoding has the same bitwidth as input/output encodings
    """
    assert factor_encoding.get("dtype") == input_encoding.get("dtype")
    """
    Then: (8) The factor encoding incurs no quantization noise:
    """
    factor_scale = factor_encoding["y_scale"]
    assert factor_encoding.get("y_zero_point", 0) == 0
    q_float = np.round(const_factor.item() / factor_scale) * factor_scale
    assert np.isclose(q_float, np.round(q_float), atol=1e-6)


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: test_models.ModelWithPreparedConstRescale(-3.0, divide=True),
        lambda: test_models.ModelWithPreparedConstRescale(-2.0, divide=False),
        lambda: test_models.ModelWithPreparedConstRescale(0.0, divide=True),
        lambda: test_models.ModelWithPreparedConstRescale(0.0, divide=False),
        lambda: test_models.ModelWithPreparedConstRescale(float("inf"), divide=True),
        lambda: test_models.ModelWithPreparedConstRescale(float("nan"), divide=True),
        lambda: test_models.RescaleWithVectorConst(divide=True),
        lambda: test_models.RescaleWithVectorConst(divide=False),
        lambda: test_models.ModelWithDynamicRescale(divide=True),
        lambda: test_models.ModelWithDynamicRescale(divide=False),
    ],
)
def test_no_propagation_for_unsafe_rescale(tmp_path, model_factory):
    """
    Given: Model with Mul/Div op where propagation would be unsafe
           (negative constant, zero constant, or non-scalar constant, dynamic constant)
    When: Export model to onnx QDQ
    Then: No QDQ is inserted at the Mul/Div output (propagation is skipped)
    """
    model = model_factory()
    dummy_input = model.dummy_input()
    sim = QuantizationSimModel(model, dummy_input)
    # Disable rescale output quantizers
    for module in sim.model.modules():
        if isinstance(module, (aimet_ops.Divide, aimet_ops.Multiply)):
            module.output_quantizers[0] = None
    sim.compute_encodings(lambda m: m(*dummy_input))
    fname = os.path.join(tmp_path, "model.onnx")
    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        fname,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        opset_version=21,
    )
    onnx_model = onnx.load(fname)
    encoding_map = _get_qdq_encoding_map(onnx_model)
    rescale_nodes = [
        node for node in onnx_model.graph.node if node.op_type in ("Mul", "Div")
    ]
    for rescale_node in rescale_nodes:
        assert rescale_node.output[0] not in encoding_map


def test_no_propagation_when_output_already_quantized(tmp_path):
    """
    Given: Model with Mul/Div where the output already has a quantizer
    When: Export model to onnx QDQ
    Then: The existing output quantizer is used, not a derived one
    """
    model = test_models.ModelWithPreparedConstRescale(2.0, divide=True)
    dummy_input = model.dummy_input()
    sim = QuantizationSimModel(model, dummy_input)
    sim.compute_encodings(lambda m: m(*dummy_input))
    sim.model.rescale.output_quantizers[0] = Q.affine.QuantizeDequantize(
        (), bitwidth=8, symmetric=False
    )
    sim.model.rescale.output_quantizers[0].set_range(0, 255.0)
    fname = os.path.join(tmp_path, "model.onnx")
    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        fname,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        opset_version=21,
    )
    onnx_model = onnx.load(fname)
    onnx.checker.check_model(onnx_model)
    encoding_map = _get_qdq_encoding_map(onnx_model)
    # The Div output should have the overriden scale of 1.0,
    rescale_node = next(
        node for node in onnx_model.graph.node if node.op_type in ("Mul", "Div")
    )
    assert rescale_node.output[0] in encoding_map
    scale, offset = encoding_map[rescale_node.output[0]]
    assert scale == 1
    assert offset == 0


def test_no_propagation_when_no_input_encoding(tmp_path):
    """
    Given: Model with Div where the input to the Div has no quantizer encoding
    When: Export model to onnx QDQ
    Then: No QDQ is inserted at the Div output (no input encoding to propagate from)
    """
    model = test_models.StandalonePreparedConstRescale(3.0, divide=True)
    dummy_input = model.dummy_input()
    sim = QuantizationSimModel(model, dummy_input)
    # Disable both input and output quantizers on the Divide
    for module in sim.model.modules():
        if isinstance(module, (aimet_ops.Divide, aimet_ops.Multiply)):
            module.output_quantizers[0] = None
            module.input_quantizers[0] = None
    sim.compute_encodings(lambda m: m(*dummy_input))
    fname = os.path.join(tmp_path, "model.onnx")
    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        fname,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        opset_version=21,
    )
    onnx_model = onnx.load(fname)
    encoding_map = _get_qdq_encoding_map(onnx_model)
    rescale_node = next(
        node for node in onnx_model.graph.node if node.op_type in ("Mul", "Div")
    )
    assert rescale_node.output[0] not in encoding_map


def test_concat_forward_propagation(tmp_path: pathlib.Path):
    """
    Given: Model with a Concat with all inputs sharing the same encoding
           or Split with all outputs sharing the same encoding

      --> QDQ ->                                          -> QDQ ->
                Concat --------> ... or ... -------> Split
      --> QDQ ->                                          -> QDQ ->

    When: Export to onnx QDQ
    Then: The exported onnx QDQ should reuse the input encoding for the concat outputs

      --> QDQ ->                                           -> QDQ ->
                Concat -> QDQ -> ... or ... -> QDQ -> Split
      --> QDQ ->                                           -> QDQ ->
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)

        def forward(self, input):
            in1, in2 = torch.split(input, 3, dim=1)
            out1 = self.conv(in1)
            out2 = self.conv(in2)
            return torch.cat((out1, out2), dim=1)

    model = Model()
    x = torch.randn(1, 6, 224, 224)
    sim = aimet_torch.QuantizationSimModel(model, x)
    sim.compute_encodings(lambda model: model(x))

    aimet_torch.onnx.export(
        sim.model,
        (x,),
        tmp_path / "concat.onnx",
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
    )
    onnx_model = onnx.load(tmp_path / "concat.onnx")
    onnx.checker.check_model(onnx_model)
    producers = {out: node for node in onnx_model.graph.node for out in node.output}
    assert producers["output"].op_type == "DequantizeLinear"

    consumers = {}
    for node in onnx_model.graph.node:
        for input in node.input:
            consumers.setdefault(input, []).append(node)
    (input_consumer,) = consumers["input"]
    assert input_consumer.op_type == "QuantizeLinear"

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": x.detach().numpy()})
    expected_out = sim.model(x)
    atol = sim.model.conv.output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


def test_exported_qdq_matches_sim_with_lossy_rescale_quantization(tmp_path):
    """
    Given: Model with a rescale factor that undergoes lossy quantization
    When: Export to onnx QDQ
    Then: The exported onnx QDQ should match sim output when executed
    """
    model = test_models.ModelWithPreparedConstRescale(2.0, divide=True)
    dummy_input = model.dummy_input()
    sim = QuantizationSimModel(model, dummy_input, default_output_bw=8)
    sim.model.rescale.output_quantizers[0] = None
    sim.compute_encodings(lambda m: m(*dummy_input))
    # Rescale will get clipped to 1.5 in QDQ
    clip_val = 1.5
    sim.model.rescale.input_quantizers[1] = Q.affine.QuantizeDequantize(
        (), bitwidth=8, symmetric=False
    )
    sim.model.rescale.input_quantizers[1].set_range(0, clip_val)
    sim_output = sim.model(*dummy_input)
    fname = os.path.join(tmp_path, "model.onnx")
    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        fname,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        opset_version=21,
    )
    onnx_model = onnx.load(fname)
    onnx.checker.check_model(onnx_model)

    # Verify that QDQ output matches sim output
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    onnx_input = {
        inp.name: dummy_input[i].detach().numpy()
        for i, inp in enumerate(onnx_model.graph.input)
    }
    (ort_out,) = sess.run(None, onnx_input)
    assert np.allclose(
        ort_out,
        sim_output.detach().numpy(),
        atol=sim.model.linear_2.output_quantizers[0].get_scale().item(),
    )


def test_encoding_metadata(tmp_path: pathlib.Path):
    """
    Given: A quantized model
    When: Export
    Then: The exported encoding should contain metadata with correct encoding version and AIMET version
    """
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    x = torch.randn(1, 10)

    sim = aimet_torch.QuantizationSimModel(model, x)
    sim.compute_encodings(lambda model: model(x))

    for encoding_version in ["0.6.1", "1.0.0", "2.0.0", "2.1.0"]:
        sim.onnx.export(
            (x,),
            tmp_path / "model.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
            encoding_version=encoding_version,
        )

        encodings = json.load(open(tmp_path / "model.encodings"))
        assert encodings["version"] == encoding_version
        assert encodings["producer"] == {
            "package": "aimet-torch",
            "version": aimet_torch.__version__,
        }

    aimet_torch.onnx.export(
        sim.model,
        (x,),
        tmp_path / "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    onnx_qdq_model = onnx.load(tmp_path / "model.onnx")
    (prop,) = onnx_qdq_model.metadata_props
    assert prop.key == "producer"
    assert prop.value == f"aimet-torch {aimet_torch.__version__}"


@pytest.mark.parametrize("dynamo", [True, False])
def test_shared_weight_export(tmp_path: pathlib.Path, dynamo: bool):
    """
    Given: Model with shared weight and identical encodings
    When: Export to onnx QDQ
    Then: The shared weight should be associated with exactly one QDQ node
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10, bias=False),
        torch.nn.Linear(10, 10, bias=False),
    )
    model[1].weight = model[0].weight

    x = torch.randn(10, 10)
    sim = aimet_torch.QuantizationSimModel(model, x)
    sim.compute_encodings(lambda model: model(x))
    aimet_torch.onnx.export(
        sim.model,
        (x,),
        tmp_path / "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamo=dynamo,
    )
    onnx_model = onnx.load(tmp_path / "model.onnx")
    onnx.checker.check_model(onnx_model)

    q_nodes = [
        node for node in onnx_model.graph.node if node.op_type == "QuantizeLinear"
    ]
    assert len([q.input[0] for q in q_nodes]) == 4
    # Shouldn't contain Transpose
    op_types = set(node.op_type for node in onnx_model.graph.node)
    assert op_types == {"QuantizeLinear", "DequantizeLinear", "MatMul"} or op_types == {
        "QuantizeLinear",
        "DequantizeLinear",
        "Gemm",
    }

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": x.detach().numpy()})
    expected_out = sim.model(x)
    atol = sim.model[1].output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


@pytest.mark.parametrize("dynamo", [True, False])
def test_lmhead_weight_sharing_export(tmp_path: pathlib.Path, dynamo: bool):
    qembedding = QuantizedEmbedding(num_embeddings=10, embedding_dim=12)
    qlinear = QuantizedLinear(in_features=12, out_features=10)
    qlinear.weight = qembedding.weight
    qembedding.param_quantizers["weight"] = Q.affine.QuantizeDequantize(
        shape=(),
        qmin=-128,
        qmax=127,
        symmetric=True,
    )
    qlinear.param_quantizers["weight"] = Q.affine.QuantizeDequantize(
        shape=(),
        qmin=-128,
        qmax=127,
        symmetric=True,
    )
    qembedding.compute_param_encodings()
    qlinear.compute_param_encodings()
    model = torch.nn.Sequential(qembedding, qlinear)

    x = torch.randint(0, 10, (2, 10))
    aimet_torch.onnx.export(
        model,
        (x,),
        tmp_path / "model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
        dynamo=dynamo,
    )
    onnx_model = onnx.load(tmp_path / "model.onnx")
    op_types = set(node.op_type for node in onnx_model.graph.node)
    assert op_types == {"QuantizeLinear", "DequantizeLinear", "MatMul", "Add", "Gather"}

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": x.detach().numpy()})
    expected_out = model(x)
    assert torch.allclose(torch.from_numpy(out), expected_out)


def test_unhashable_input():
    """
    Given: Built-in module that takes an unhashable input (e.g. list or dict)
    When: Export to onnx QDQ
    Then: Export should succeed and the exported model should be valid
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv1d(3, 3, 3)
            self.reshape = aimet_ops.Reshape()

        def forward(self, x):
            x = self.conv(x)
            x = self.reshape(x, [-1])
            return x

    model = Model()
    dummy_input = torch.randn(1, 3, 224)
    sim = aimet_torch.QuantizationSimModel(model, dummy_input)
    sim.compute_encodings(lambda model: model(dummy_input))
    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": dummy_input.detach().numpy()})
    expected_out = sim.model(dummy_input)
    atol = sim.model.conv.output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


@pytest.mark.parametrize(
    "conv_cls, padding_mode",
    [
        *itertools.product(
            [QuantizedConv1d, QuantizedConv2d, QuantizedConv3d],
            ["zeros", "reflect", "replicate", "circular"],
        ),
        (QuantizedConvTranspose1d, "zeros"),
        (QuantizedConvTranspose2d, "zeros"),
        (QuantizedConvTranspose3d, "zeros"),
    ],
)
def test_pad_conv_export(conv_cls, padding_mode: str, tmp_path: pathlib.Path):
    """
    When: Export Conv with various padding modes to ONNX QDQ
    Then: All input/output of Conv and Pad (if any) should be associated with QDQ nodes
    """
    qconv = conv_cls(3, 3, 3, padding=1, padding_mode=padding_mode)
    qconv.input_quantizers[0] = Q.affine.QuantizeDequantize(
        (), qmin=0, qmax=255, symmetric=False
    )
    qconv.output_quantizers[0] = Q.affine.QuantizeDequantize(
        (), qmin=0, qmax=255, symmetric=False
    )
    qconv.param_quantizers["weight"] = Q.affine.QuantizeDequantize(
        (), qmin=-128, qmax=127, symmetric=True
    )
    input = torch.randn(1, 3, *(24 for _ in range(qconv.weight.ndim - 2)))

    with torch.no_grad(), qconv.compute_encodings():
        _ = qconv(input)

    aimet_torch.onnx.export(
        qconv,
        (input,),
        tmp_path / "conv.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
    )

    onnx_model = onnx.load(tmp_path / "conv.onnx")
    onnx.checker.check_model(onnx_model)

    producers = {out: node for node in onnx_model.graph.node for out in node.output}
    consumers = {}
    for node in onnx_model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    conv_node = next(
        node
        for node in onnx_model.graph.node
        if node.op_type in ("Conv", "ConvTranspose")
    )
    pad_node = next(
        (node for node in onnx_model.graph.node if node.op_type == "Pad"), None
    )

    for inp in conv_node.input:
        producer = producers[inp]
        assert producer.op_type == "DequantizeLinear"
    (consumer,) = consumers[conv_node.output[0]]
    assert consumer.op_type == "QuantizeLinear"

    if pad_node:
        assert padding_mode in ("reflect", "replicate")  # sanity check
        producer = producers[pad_node.input[0]]
        assert producer.op_type == "DequantizeLinear"
        (consumer,) = consumers[conv_node.output[0]]
        assert consumer.op_type == "QuantizeLinear"

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    (out,) = sess.run(None, {"input": input.detach().numpy()})
    expected_out = qconv(input)
    atol = qconv.output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


def test_qmha_export(tmp_path: pathlib.Path):
    """
    Given: A quantized multi-head attention module
    When: Export to onnx QDQ
    Then: The exported onnx model should be valid and should have QDQ nodes around all computation nodes
    """
    embed_dim = 8
    num_heads = 2
    L, N, E = 4, 2, embed_dim

    mha = torch.nn.MultiheadAttention(embed_dim, num_heads)

    query = torch.randn(L, N, E)
    key = torch.randn(L, N, E)
    value = torch.randn(L, N, E)
    dummy_input = (query, key, value)

    sim = aimet_torch.QuantizationSimModel(mha, dummy_input)
    sim.compute_encodings(lambda model: model(*dummy_input))

    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        tmp_path / "mha_decomposed.onnx",
        input_names=["query", "key", "value"],
        output_names=["output", "attn_weights"],
        dynamo=False,
    )

    onnx_model = onnx.load(tmp_path / "mha_decomposed.onnx")
    onnx.checker.check_model(onnx_model)

    producers = {out: node for node in onnx_model.graph.node for out in node.output}
    consumers = {}

    for node in onnx_model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    for node in onnx_model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear", "Constant"):
            continue

        for inp in node.input:
            producer = producers.get(inp)
            if producer:
                assert (
                    producer.op_type == "DequantizeLinear"
                    or producer.op_type == "Constant"
                    or (producer.op_type, node.op_type)
                    in [
                        ("Transpose", "MatMul"),
                        ("MatMul", "Add"),
                    ]
                )

        for out in node.output:
            for consumer in consumers.get(out, []):
                assert consumer.op_type == "QuantizeLinear" or (
                    (node.op_type, consumer.op_type)
                    in [
                        ("Transpose", "MatMul"),
                        ("MatMul", "Add"),
                    ]
                )

    input_names = set(inp.name for inp in onnx_model.graph.input)
    output_names = set(out.name for out in onnx_model.graph.output)
    for node in onnx_model.graph.node:
        if node.input and node.input[0] in input_names:
            assert node.op_type == "QuantizeLinear"
            input_names.remove(node.input[0])
        if node.output and node.output[0] in output_names:
            assert node.op_type == "DequantizeLinear"
            output_names.remove(node.output[0])
    assert not input_names
    assert not output_names

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    out, attn_weights = sess.run(
        None,
        {
            "query": query.detach().numpy(),
            "key": key.detach().numpy(),
            "value": value.detach().numpy(),
        },
    )
    expected_out, expected_attn_weights = sim.model(*dummy_input)
    atol = sim.model.out_proj.output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)

    atol = sim.model.mean.output_quantizers[0].get_scale().item()
    assert torch.allclose(
        torch.from_numpy(attn_weights), expected_attn_weights, atol=atol
    )


@pytest.mark.parametrize("dynamo", [True, False])
@pytest.mark.parametrize(
    "qtzr_cls", [Q.affine.QuantizeDequantize, Q.float.FloatQuantizeDequantize]
)
@pytest.mark.parametrize(
    "shape, block_size",
    [
        [(), None],  # per-tensor
        [(12, 1), None],  # per-channel with axis=0
        [(10,), None],  # per-channel with axis=1
        [(1, 10), None],  # per-channel with axis=1
        [(12, 2), (1, 5)],  # per-channel with channel_axis=0, block_axis=1
        [(2, 10), (6, 1)],  # per-channel with channel_axis=1, block_axis=0
    ],
)
def test_qlinear_onnx_export(
    tmp_path: pathlib.Path, qtzr_cls, shape, block_size, dynamo: bool
):
    if dynamo and qtzr_cls is Q.float.FloatQuantizeDequantize:
        pytest.skip("Exporting float quantizer with dynamo is not implemented yet")

    model = torch.nn.Linear(in_features=10, out_features=12)
    sim = QuantizationSimModel(model, torch.randn(10))
    qlinear = sim.model
    aimet_torch.utils.remove_activation_quantizers(qlinear)

    if qtzr_cls is Q.affine.QuantizeDequantize:
        qlinear.param_quantizers["weight"] = Q.affine.QuantizeDequantize(
            shape=shape,
            qmin=-128,
            qmax=127,
            symmetric=True,
            block_size=block_size,
        )
    else:
        qlinear.param_quantizers["weight"] = Q.float.FloatQuantizeDequantize(
            dtype=torch.float8_e5m2,
            shape=shape,
            block_size=block_size,
        )

    qlinear.compute_param_encodings()

    # Test with 1-3D input
    for ndim in range(1, 4):
        x = torch.randn(10)

        while x.ndim < ndim:
            x = x.unsqueeze(0)

        if qtzr_cls is Q.affine.QuantizeDequantize:
            sim.onnx.export(
                (x,),
                tmp_path / "qlinear.onnx",
                input_names=["input"],
                output_names=["output"],
                dynamo=dynamo,
                encoding_version="1.0.0",
            )
            with open(tmp_path / "qlinear.encodings") as f:
                encodings = json.load(f)
            assert len(encodings["param_encodings"]) == 1

        aimet_torch.onnx.export(
            qlinear,
            (x,),
            tmp_path / "qlinear.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=21,
            dynamo=dynamo,
        )
        onnx_model = onnx.load(tmp_path / "qlinear.onnx")
        op_types = set(node.op_type for node in onnx_model.graph.node)

        # Shouldn't contain Transpose
        if ndim == 2:
            assert op_types == {"QuantizeLinear", "DequantizeLinear", "Gemm"}
        else:
            assert op_types == {"QuantizeLinear", "DequantizeLinear", "MatMul", "Add"}

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        sess = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=["CPUExecutionProvider"],
            sess_options=sess_options,
        )
        (out,) = sess.run(None, {"input": x.detach().numpy()})
        expected_out = qlinear(x)
        assert torch.allclose(torch.from_numpy(out), expected_out)


def test_export_with_branch_input_quantizer(tmp_path):
    """
    Given: A tensor consumed by ops both with and without input quantizers
    When: Export the model to onnx QDQ
    Then: Ops without input quantizers should receive the unquantized tensor
    """
    model = test_models.ModelWithBranch()
    (dummy_input,) = model.dummy_input()
    sim = QuantizationSimModel(model, dummy_input)
    remove_activation_quantizers(sim.model)
    sim.model.linear2.input_quantizers[0] = Q.affine.QuantizeDequantize(
        (), bitwidth=8, symmetric=False
    )
    sim.compute_encodings(lambda model: model(dummy_input))
    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        tmp_path / "branch_model.onnx",
    )

    onnx_model = onnx.load(tmp_path / "branch_model.onnx")
    producers = {node.output[0]: node for node in onnx_model.graph.node}
    add_node = next(node for node in onnx_model.graph.node if node.op_type == "Add")
    # Add node only receives unquantized tensors in torch
    for tensor in add_node.input:
        assert producers[tensor].op_type != "DequantizeLinear"


@pytest.mark.parallel
@pytest.mark.parametrize("dynamo", [True, False])
def test_export_with_shared_weight(tmp_path, dynamo: bool):
    """
    Given:
        W -+-> QDQ --------------> consumer_1
           +-> Identity -> QDQ' -> consumer_2

        where QDQ and QDQ' share the same encoding

    When: Export to onnx QDQ
    Then: The exported onnx model should have a single QDQ node for the shared weight

        W -> QDQ --+-> consumer_1
                   +-> consumer_2
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.norm1 = torch.nn.LayerNorm(10, bias=False)
            self.norm2 = torch.nn.LayerNorm(10, bias=False)
            with torch.no_grad():
                self.norm2.weight.copy_(self.norm1.weight)

        def forward(self, x):
            return self.norm2(self.norm1(x))

    sim = QuantizationSimModel(Model(), torch.randn(1, 3, 10, 10))
    sim.compute_encodings(lambda model: model(torch.randn(1, 3, 10, 10)))
    aimet_torch.onnx.export(
        sim.model,
        torch.randn(1, 3, 10, 10),
        tmp_path / "shared_weight_model_qdq.onnx",
        dynamo=dynamo,
    )
    onnx_model = onnx.load(tmp_path / "shared_weight_model_qdq.onnx")

    producers = {node.output[0]: node for node in onnx_model.graph.node}
    q_nodes = [
        node for node in onnx_model.graph.node if node.op_type == "QuantizeLinear"
    ]
    assert len(q_nodes) == 4  # shared weight, input, norm1 output, norm2 output
    for node in onnx_model.graph.node:
        if node.op_type == "QuantizeLinear":
            # There shouldn't be any back-to-back QDQ
            producer = producers.get(node.input[0])
            assert not (producer and producer.op_type == "DequantizeLinear")

    # There shouldn't be any Identity nodes in the exported model, e.g.
    #     W -> QDQ -> Identity -> QDQ -+-> consumer_1
    #                                  +-> consumer_2
    assert not [node for node in onnx_model.graph.node if node.op_type == "Identity"]


class MaskedSoftmax(torch.nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.amin = aimet_torch.nn.modules.custom.AMin()
        self.equal = aimet_torch.nn.modules.custom.Equal()
        self.where = aimet_torch.nn.modules.custom.Where()
        self.add = aimet_torch.nn.modules.custom.Add()
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        mask_val = self.add(self.amin(input, [-1], keepdims=True), -20.0)
        return self.softmax(
            self.where(self.equal(mask, 0.0), input, mask_val),
        )


# TODO(#7434): Remove this test case once aimet-torch supports MaskedSoftmax as supergroup
def test_masked_softmax_temporary_workaround(tmp_path: pathlib.Path):
    """
    Given: MaskedSoftmax subgraph manually set to supergroup by user
    When: Export
    Then: Intermediate outputs of the subgraph should not be quantized
    """
    masked_softmax = MaskedSoftmax()
    qk = torch.randn(1, 3, 3, 3)
    mask = torch.tensor(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ]
    ).reshape(1, 1, 3, 3)

    sim = aimet_torch.QuantizationSimModel(
        masked_softmax,
        (qk, mask),
        default_output_bw=16,
        config_file="htp_v81",
    )
    aimet_torch.utils.remove_output_quantizers(sim.model.amin)
    aimet_torch.utils.remove_output_quantizers(sim.model.equal)
    aimet_torch.utils.remove_output_quantizers(sim.model.where)
    aimet_torch.utils.remove_output_quantizers(sim.model.add)
    aimet_torch.utils.remove_input_quantizers(sim.model.softmax)
    sim.compute_encodings(lambda model: model(qk, mask))
    sim.onnx.export(
        (qk, mask),
        tmp_path / "masked_softmax.onnx",
        input_names=["qk", "mask"],
        output_names=["output"],
        dynamo=False,
        encoding_version="2.0.0",
    )
    with open(tmp_path / "masked_softmax.encodings") as f:
        encodings = json.load(f)

    assert {e["name"] for e in encodings["encodings"]} == {
        "qk",
        "output",
    }


@pytest.mark.parametrize("dynamo", [True, False])
def test_rotary_embedding_export(tmp_path: pathlib.Path, dynamo: bool):
    """
    When: Exporting custom RotaryEmbedding module to onnx QDQ
    Then: Intermediate outputs should not be quantized
    """
    rope = aimet_ops.RotaryEmbedding(
        interleaved=False, rotary_embedding_dim=4, head_size=8
    )
    dummy_input = (
        torch.randn(1, 1, 2, 8),
        torch.randn(1, 2, 2),
        torch.randn(1, 2, 2),
    )
    sim = QuantizationSimModel(rope, dummy_input)
    sim.compute_encodings(lambda m: m(*dummy_input))
    aimet_torch.onnx.export(
        sim.model,
        dummy_input,
        str(tmp_path / "rotary_embedding.onnx"),
        input_names=["input", "cos", "sin"],
        output_names=["output"],
        dynamo=dynamo,
    )
    onnx_model = onnx.load_model(tmp_path / "rotary_embedding.onnx")
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    (ort_out,) = sess.run(
        None,
        {
            "input": dummy_input[0].detach().numpy(),
            "cos": dummy_input[1].detach().numpy(),
            "sin": dummy_input[2].detach().numpy(),
        },
    )
    expected_out = sim.model(*dummy_input)
    assert torch.allclose(torch.from_numpy(ort_out), expected_out)

    dq_nodes = [dq for dq in onnx_model.graph.node if dq.op_type == "DequantizeLinear"]
    assert len(dq_nodes) == 4
    assert dq_nodes[0].output[0] == "input_qdq"
    assert dq_nodes[1].output[0] == "cos_qdq"
    assert dq_nodes[2].output[0] == "sin_qdq"
    assert dq_nodes[3].output[0] == "output"


@pytest.mark.parametrize("dynamo", [True, False])
def test_rotary_embedding_export_encodings(tmp_path: pathlib.Path, dynamo: bool):
    """
    When: Exporting custom RotaryEmbedding module to onnx
    Then: RoPE output encodings should not have _UnsafeBarrier in the tensor name
    """

    class RoPEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = aimet_ops.RotaryEmbedding(
                interleaved=False, rotary_embedding_dim=4, head_size=8
            )
            self.add = aimet_ops.Add()
            self.sub = aimet_ops.Subtract()

        def forward(self, x, cos_cache, sin_cache):
            rope_out = self.rope(self.add(x, 2), cos_cache, sin_cache)
            return self.sub(rope_out, 2)

    rope = RoPEModel()
    dummy_input = (
        torch.randn(1, 1, 2, 8),
        torch.randn(1, 2, 2),
        torch.randn(1, 2, 2),
    )
    sim = QuantizationSimModel(rope, dummy_input)
    sim.compute_encodings(lambda m: m(*dummy_input))
    sim.onnx.export(
        dummy_input,
        str(tmp_path / "rotary_embedding.onnx"),
        input_names=["input", "cos", "sin"],
        output_names=["output"],
        dynamo=dynamo,
        encoding_version="2.0.0",
    )

    with open(tmp_path / "rotary_embedding.encodings") as f:
        encodings = json.load(f)["encodings"]

    encoding_names = {e["name"] for e in encodings}
    expected_names = {"input", "cos", "sin", "output"}

    if dynamo:
        expected_names |= {"concat_1", "add"}
    else:
        expected_names |= {"/rope/Concat_output_0", "/add/Add_output_0"}

    assert encoding_names == expected_names


def test_export_aten_quantize_dequantize(tmp_path: pathlib.Path):
    """
    Given: Model with torch built-in (de-)quantize_per_* ops
    When: Export
    Then: The exported onnx model should have QDQ where torch built-in (de-)quantize_per_* ops were
    """

    @aimet_torch.nn.QuantizationMixin.ignore
    class Model(torch.nn.Module):
        def forward(self, x):
            x = torch.ops.quantized_decomposed.quantize_per_tensor(
                x,
                scale=0.1,
                zero_point=0,
                quant_min=0,
                quant_max=255,
                dtype=torch.uint8,
            )
            x = torch.ops.quantized_decomposed.dequantize_per_tensor(
                x,
                scale=0.1,
                zero_point=0,
                quant_min=0,
                quant_max=255,
                dtype=torch.uint8,
            )
            return x.T

    model = Model()
    dummy_input = torch.randn(10, 10)
    aimet_torch.onnx.export(
        model,
        dummy_input,
        str(tmp_path / "model.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
    )
    model = onnx.load(str(tmp_path / "model.onnx"))
    producers = {out: node for node in model.graph.node for out in node.output}
    consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    assert producers["output"].op_type == "DequantizeLinear"
    (q,) = consumers["input"]
    assert q.op_type == "QuantizeLinear"

    sim = QuantizationSimModel(Model(), dummy_input).onnx.export(
        dummy_input,
        str(tmp_path / "model.onnx"),
        input_names=["input"],
        output_names=["output"],
        encoding_version="2.0.0",
    )

    with open(tmp_path / "model.encodings") as f:
        encodings = json.load(f)["encodings"]

    assert {e["name"] for e in encodings} == {"input", "output"}


@pytest.mark.parametrize(
    "module_cls",
    [
        aimet_ops.Concat,
        # aimet_ops.Where,
    ],
)
def test_multi_input_grid_equivariant_op_encoding_propagation(
    tmp_path: pathlib.Path, module_cls
):
    """
    Given: Multi-input grid-equivariant module (e.g. Concat, Where)
    """
    module = module_cls()

    if module_cls is aimet_ops.Concat:
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)
        inputs = {"x": x, "y": y}
    elif module_cls is aimet_ops.Where:
        x = torch.randn(5, 5) > 0
        y = torch.randn(5, 5)
        z = torch.randn(5, 5)
        inputs = {"x": x, "y": y, "z": z}
    else:
        raise ValueError(f"Unsupported module class: {module_cls}")

    input_names = list(inputs.keys())
    float_input_names = [
        name for name, inp in inputs.items() if inp.is_floating_point()
    ]
    inputs = tuple(inputs.values())
    sim = aimet_torch.QuantizationSimModel(module, inputs)
    sim.compute_encodings(lambda m: m(*inputs))

    # All input quantizers should be tied automatically
    assert len(set(sim.model.input_quantizers)) == 1

    def _export_and_get_encoding(sim):
        sim.onnx.export(
            inputs,
            tmp_path / "model.onnx",
            input_names=input_names,
            output_names=["output"],
            opset_version=21,
            encoding_version="2.0.0",
        )

        with open(tmp_path / "model.encodings") as f:
            encodings = json.load(f)["encodings"]
        return encodings

    """
    When: Export without input quantizer
    Then: Output encoding should be propagated to inputs
    """
    with aimet_torch.utils.remove_input_quantizers(sim.model):
        encodings = _export_and_get_encoding(sim)
        assert {e.pop("name") for e in encodings} == {*float_input_names, "output"}
        assert len(set(tuple(e.items()) for e in encodings)) == 1

    """
    When: Export without output quantizer and with same encoding across all inputs
    Then: Input encodings should be propagated to output
    """
    with aimet_torch.utils.remove_output_quantizers(sim.model):
        encodings = _export_and_get_encoding(sim)
        assert {e.pop("name") for e in encodings} == {*float_input_names, "output"}
        assert len(set(tuple(e.items()) for e in encodings)) == 1

    """
    When: Export without output quantizer with different input encodings
    Then: Input encodings should NOT be propagated to output
    """
    with aimet_torch.utils.remove_output_quantizers(sim.model):
        # Manually untie input quantizers
        sim.model.input_quantizers[-1] = copy.deepcopy(sim.model.input_quantizers[-1])
        sim.model.input_quantizers[-1].set_range(-10, 10)

        encodings = _export_and_get_encoding(sim)
        assert {e.pop("name") for e in encodings} == {*float_input_names}
        assert len(set(tuple(e.items()) for e in encodings)) == len(float_input_names)


@pytest.mark.parametrize("input_ndim", [2, 3])
def test_encoding_version_2_1_0(tmp_path: pathlib.Path, input_ndim: int):
    """
    Given: A quantized model with LPBQ weights
    When: Export to onnx QDQ with encoding version 2.1.0
    Then: The exported encodings should contain LPBQ encoding in 2.1.0 format
    """
    dummy_input = torch.randn(1, 10) if input_ndim == 2 else torch.randn(1, 1, 10)
    sim = QuantizationSimModel(torch.nn.Linear(10, 10), dummy_input)

    set_grouped_blockwise_quantization_for_weights(
        sim,
        [torch.nn.Linear],
        bitwidth=4,
        symmetric=True,
        decompressed_bw=8,
        block_size=2,
    )
    aimet_torch.utils.remove_activation_quantizers(sim.model)
    sim.model.compute_param_encodings()

    sim.onnx.export(
        (dummy_input,),
        tmp_path / "model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
        encoding_version="2.1.0",
        export_int32_bias=False,
    )

    with open(tmp_path / "model.encodings") as f:
        encodings = json.load(f)["encodings"]

    lpbq_enc = sim.model.param_quantizers["weight"].get_encodings()
    expected_meta_scale = lpbq_enc.per_channel_scale.flatten()

    if input_ndim == 2:
        input_name = "weight"
        expected_channel_axis = -2
        expected_block_axis = 1
        expected_scale_q = lpbq_enc.per_block_int_scale.int()
    else:
        # If input_dim > 2, Linear is exported as MatMul + Add with transposed weight
        # Check LPBQ encoding axes were also transposed accordingly
        input_name = "weight_0"
        expected_channel_axis = -1
        expected_block_axis = 0
        expected_scale_q = lpbq_enc.per_block_int_scale.T.int()

    expected_encoding = [
        {
            "name": input_name,
            "y_scale": {
                "x": expected_scale_q.tolist(),
                "x_scale": expected_meta_scale.tolist(),
                "axis": expected_channel_axis,
            },
            "axis": expected_block_axis,
            "block_size": 2,
            "output_dtype": "int4",
        }
    ]

    assert encodings == expected_encoding


def test_export_with_abnormal_axes(tmp_path: pathlib.Path):
    """
    Given: Weight quantized with multiple non-trivial block axes
    When: Export to onnx QDQ
    Then: Should raise error
    """
    dummy_input = torch.randn(1, 10)
    sim = QuantizationSimModel(torch.nn.Linear(10, 10), dummy_input)
    aimet_torch.utils.remove_activation_quantizers(sim.model)
    sim.model.param_quantizers["weight"] = Q.affine.QuantizeDequantize(
        shape=(2, 2),
        qmin=-8,
        qmax=7,
        symmetric=True,
        block_size=(-1, -1),
    )
    sim.model.compute_param_encodings()

    with pytest.raises(RuntimeError, match="Multiple non-trivial block axes found"):
        sim.onnx.export(
            (dummy_input,),
            tmp_path / "model.onnx",
            opset_version=21,
            encoding_version="2.1.0",
        )

    with pytest.raises(RuntimeError, match="Multiple non-trivial block axes found"):
        aimet_torch.onnx.export(
            sim.model,
            (dummy_input,),
            tmp_path / "model.onnx",
            opset_version=21,
        )

    """
    Given: Weight quantized with no channel axis and one non-trivial block axis
    When: Export to onnx QDQ
    Then: Should raise error
    """
    sim.model.param_quantizers["weight"] = Q.affine.QuantizeDequantize(
        shape=(1, 2),
        qmin=-8,
        qmax=7,
        symmetric=True,
        block_size=(-1, -1),
    )
    sim.model.compute_param_encodings()

    with pytest.raises(
        RuntimeError,
        match="Block axis 1 found without a corresponding channel axis",
    ):
        sim.onnx.export(
            (dummy_input,),
            tmp_path / "model.onnx",
            opset_version=21,
            encoding_version="2.1.0",
        )

    with pytest.raises(
        RuntimeError,
        match="Block axis 1 found without a corresponding channel axis",
    ):
        aimet_torch.onnx.export(
            sim.model,
            (dummy_input,),
            tmp_path / "model.onnx",
            opset_version=21,
        )


@pytest.fixture(scope="module")
def nvfp4_sim() -> QuantizationSimModel:
    dummy_input = torch.randn(1, 4)
    sim = QuantizationSimModel(torch.nn.Linear(4, 16), dummy_input)
    aimet_torch.utils.remove_all_quantizers(sim.model)

    quantized_scale = (torch.arange(1, 9).reshape(4, 2) / 8).to(torch.float8_e4m3fn)
    meta_scale = torch.tensor(0.1)
    scale = quantized_scale.to(torch.float32) * meta_scale

    nvfp4_weight = _float_quantize_dequantize(
        sim.model.weight,
        finfo=_float4_e2m1fn,
        scale=scale,
        block_size=(1, 8),
    ).as_subclass(Q.DequantizedTensor)
    nvfp4_weight.encoding = _NVFP4Encoding(
        scale=scale,
        meta_scale=meta_scale,
        block_size=(1, 8),
    )
    sim.model.weight = torch.nn.Parameter(nvfp4_weight)
    return sim


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_nvfp4_encoding(tmp_path: pathlib.Path, dynamo: bool, nvfp4_sim):
    """
    Given: A quantized model with NVFP4 weights
    When: Export to onnx with encoding version 2.1.0
    Then: The exported encodings should contain NVFP4 encoding in 2.1.0 format
    """
    dummy_input = torch.randn(1, 4)
    nvfp4_sim.onnx.export(
        (dummy_input,),
        str(tmp_path / "model.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
        encoding_version="2.1.0",
        export_int32_bias=False,
        dynamo=dynamo,
    )

    with open(tmp_path / "model.encodings") as f:
        encodings = json.load(f)["encodings"]

    nvfp4_enc = nvfp4_sim.model.weight.encoding
    expected_encoding = [
        {
            "name": "weight",
            "y_scale": {
                "x": nvfp4_enc._get_quantized_scale().tolist(),
                "x_scale": nvfp4_enc.meta_scale.item(),
                "input_dtype": "float8e4m3fn",
            },
            "axis": 1,
            "block_size": 8,
            "output_dtype": "float4e2m1",
        }
    ]
    assert encodings == expected_encoding


@pytest.mark.cuda
@pytest.mark.parametrize("dynamo", [True, False])
def test_export_nvfp4_onnx_qdq(tmp_path: pathlib.Path, dynamo: bool, nvfp4_sim):
    """
    Given: A quantized model with NVFP4 weights
    When: Export to onnx QDQ
    Then: The exported onnx QDQ model should produce the same output as the original model
    """
    dummy_input = torch.randn(1, 4)
    aimet_torch.onnx.export(
        nvfp4_sim.model,
        (dummy_input,),
        str(tmp_path / "model.onnx"),
        input_names=["input"],
        output_names=["output"],
        opset_version=25,
        export_int32_bias=False,
        dynamo=dynamo,
    )
    onnx_model = onnx.load(str(tmp_path / "model.onnx"))
    model = onnx_ir.from_proto(onnx_model)

    # Check following subgraph
    #
    #     weight -----> Q -> DQ -> ...
    #    scale_q -> DQ -^----^
    # meta_scale ---^
    weight_q = model.graph.node("weight_q")
    weight_dq = model.graph.node("weight_dq")
    weight_scale_dq = model.graph.node("weight_scale_dq")

    weight = model.graph.initializers["weight"]
    scale_q = model.graph.initializers["weight_scale_q"]
    meta_scale = model.graph.initializers["weight_meta_scale"]
    meta_zp = model.graph.initializers["weight_meta_zero_point"]
    weight_zp = model.graph.initializers["weight_zero_point"]

    assert weight_scale_dq.op_type == "DequantizeLinear"
    assert weight_scale_dq.inputs == (scale_q, meta_scale, meta_zp)
    assert weight_q.inputs == (weight, weight_scale_dq.outputs[0], weight_zp)
    assert weight_dq.inputs == (
        weight_q.outputs[0],
        weight_scale_dq.outputs[0],
        weight_zp,
    )

    if torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(0)
        compute_capability = (device_properties.major, device_properties.minor)
    else:
        compute_capability = (0, 0)

    if compute_capability < (12, 0):
        pytest.skip("ORT only supports float4e2m1 on NVIDIA Blackwell GPUs")

    sess = ort.InferenceSession(
        onnx_model.SerializeToString(), providers=["CUDAExecutionProvider"]
    )
    (out,) = sess.run(None, {"input": dummy_input.detach().numpy()})
    expected_out = nvfp4_sim.model(dummy_input)
    assert torch.allclose(torch.from_numpy(out), expected_out)


def test_int32_encoding_propagation(tmp_path):
    """
    When: Exporting a model with int32 bias to onnx QDQ
    Then: Int32 DQ should not be propagated
    """

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.randn(16))
            self.bias_quantizer = Q.affine.QuantizeDequantize(
                shape=(),
                qmin=-(2**31),
                qmax=2**31 - 1,
                symmetric=True,
            )
            self.bias_quantizer.set_range(-1, 1)

        def forward(self, x):
            bias_qdq = self.bias_quantizer(self.bias)
            return x + bias_qdq.reshape(4, 4)

    model = Model()
    dummy_input = torch.randn(4, 4)
    aimet_torch.onnx.export(
        model,
        (dummy_input,),
        tmp_path / "int32_model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=21,
    )
    onnx_model = onnx.load(tmp_path / "int32_model.onnx")
    onnx.checker.check_model(onnx_model)
    dq_nodes = [dq for dq in onnx_model.graph.node if dq.op_type == "DequantizeLinear"]
    assert len(dq_nodes) == 1
    (dq,) = dq_nodes
    assert dq.input[0] == "bias_q"
