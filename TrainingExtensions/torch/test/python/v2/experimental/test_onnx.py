# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
import copy
import os
import json
import pathlib
from packaging.version import parse
import onnxruntime as ort
import pytest
import contextlib
import numpy as np
import torch
from torch.onnx import _constants
import onnx
from onnx import helper, TensorProto
import tempfile
from unittest.mock import patch
from ..models_ import test_models

from aimet_common.quantsim_config.utils import (
    get_path_for_per_channel_config,
    get_path_for_per_tensor_config,
)
from aimet_common import quantsim as quantsim_common
import aimet_torch.v2 as aimet
import aimet_torch.v2.quantization as Q
from aimet_torch.v2.quantsim.quantsim import QuantizationSimModel
from aimet_torch.onnx import (
    _concretize_int32_bias_quantizers,
    _derive_data_movement_op_encoding,
)
from torchvision.models import resnet18, mobilenet_v3_small
from aimet_torch.v2.experimental.onnx._export import export as _export
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.utils import get_all_quantizers
from aimet_torch.v2.utils import remove_activation_quantizers
from aimet_torch.model_preparer import prepare_model
from aimet_torch.v2.quantsim.config_utils import (
    set_grouped_blockwise_quantization_for_weights,
)
import aimet_torch


@pytest.fixture(autouse=True, params=range(1))
def seed(request):
    seed = request.param
    torch.manual_seed(seed)


@contextlib.contextmanager
def set_encoding_version(version):
    try:
        old_version = quantsim_common.encoding_version
        quantsim_common.encoding_version = version
        yield
    finally:
        quantsim_common.encoding_version = old_version


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
        assert node.attribute[0].name == "block_size"
        assert node.attribute[0].ints == (
            [1]
            if block_size is None
            else list(np.array(input_shape) // np.array(scale_shape))
        )

        if qtzr_cls != Q.affine.Dequantize:
            assert node.attribute[1].name == "qmax"
            assert node.attribute[1].i == (127 if symmetric else 255)
            assert node.attribute[2].name == "qmin"
            assert node.attribute[2].i == (-128 if symmetric else 0)

        """
        Then: The saved onnx model should contain exactly one graph node in "aimet" domain
              with proper scale and offset values
        """
        const_map = {
            node.output[0]: node
            for node in model.graph.node
            if node.op_type == "Constant"
        }
        assert node.input[1] in const_map
        assert node.input[2] in const_map
        onnx_scale = torch.tensor(
            onnx.numpy_helper.to_array(const_map[node.input[1]].attribute[0].t)
        )
        onnx_offset = torch.tensor(
            onnx.numpy_helper.to_array(const_map[node.input[2]].attribute[0].t)
        )
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
        assert node.attribute[0].name == "block_size"
        assert node.attribute[0].ints == (
            [1]
            if block_size is None
            else list(np.array(input_shape) // np.array(scale_shape))
        )

        """
        Then: The saved onnx model should produce the same output with the original quantizer
              given the same input
        """
        sess = ort.InferenceSession(full_path, providers=["CPUExecutionProvider"])
        (out,) = sess.run(None, {"input": x.numpy()})
        assert torch.equal(torch.from_numpy(out), y)


@torch.no_grad()
@pytest.mark.parametrize(
    "model_factory,      input_shape",
    [
        (resnet18, (1, 3, 224, 224)),
        (mobilenet_v3_small, (1, 3, 224, 224)),
    ],
)
def test_export_torchvision_models(model_factory, input_shape):
    """
    When: Export quantized torchvision model
    """
    x = torch.randn(input_shape)
    model = model_factory().eval()
    model = prepare_model(model)
    model = QuantizationSimModel(
        model, x, config_file=get_path_for_per_channel_config()
    ).model

    with aimet.nn.compute_encodings(model):
        model(x)

    y = model(x)

    with tempfile.TemporaryDirectory() as dirname:
        full_path = os.path.join(dirname, "torchvision_model.onnx")

        with open(full_path, "wb") as f:
            _export(
                model,
                x,
                f,
                input_names=["input"],
                output_names=["output"],
                dynamo=False,
            )

        """
        Then: The saved onnx model should pass onnx model checker
        """
        onnx_model = onnx.load_model(full_path)
        onnx.checker.check_model(onnx_model)

        """
        Then: The onnx model should have the same number of quant nodes
              as the number of quantizers in the original pytorch model
        """
        nodes = [node for node in onnx_model.graph.node if node.domain == "aimet"]
        quantizers_in_model = [
            qtzr
            for qtzr_group in get_all_quantizers(model)
            for qtzr in qtzr_group
            if qtzr
        ]
        assert len(nodes) == len(quantizers_in_model)

        """
        Then: The quant nodes in the onnx model should have constant scale and offset values
        """
        const_map = {
            node.output[0]: node
            for node in onnx_model.graph.node
            if node.op_type == "Constant"
        }
        for node in nodes:
            assert node.input[1] in const_map
            assert node.input[2] in const_map

        """
        Then: The onnx model should produce output close enough to the original pytorch model
        """
        sess = ort.InferenceSession(full_path, providers=["CPUExecutionProvider"])
        (out,) = sess.run(None, {"input": x.numpy()})

        # Allow off-by-3 error
        atol = 3 * y.encoding.scale.item()
        assert torch.allclose(torch.from_numpy(out), y, atol=atol)


@torch.no_grad()
@pytest.mark.parametrize("encoding_version", ["0.6.1", "1.0.0", "2.0.0"])
@pytest.mark.parametrize("lpbq", [False, True])
@pytest.mark.parametrize("export_int32_bias", [False, True])
@pytest.mark.parametrize("fold_param_quantizers", [False, True])
@pytest.mark.parametrize(
    "param_dtype, activation_dtype",
    [
        ("int8", "uint8"),
        ("int8", "float16"),
        ("float16", "float16"),
    ],
)
def test_quantsim_export_resnet18(
    encoding_version,
    lpbq: bool,
    fold_param_quantizers: bool,
    export_int32_bias: bool,
    param_dtype: str,
    activation_dtype: str,
):
    """
    When: Export quantized torchvision model using quantsim.export
    """
    x = torch.randn(1, 3, 224, 224)
    model = resnet18().eval()
    model = prepare_model(model)
    fold_all_batch_norms(model, None, x)

    param_kind, param_bw = _parse_type(param_dtype)
    activation_kind, activation_bw = _parse_type(activation_dtype)
    sim = QuantizationSimModel(
        model, x, default_param_bw=param_bw, default_output_bw=activation_bw
    )

    if lpbq:
        set_grouped_blockwise_quantization_for_weights(
            sim,
            [sim.model.fc],
            bitwidth=4,
            symmetric=True,
            decompressed_bw=8,
            block_size=64,
        )

    if param_kind == "float":
        dtype = getattr(torch, param_dtype)
        for qmodule in sim.qmodules():
            for name, qtzr in qmodule.param_quantizers.items():
                if not qtzr:
                    continue
                qmodule.param_quantizers[name] = Q.float.FloatQuantizeDequantize(
                    dtype=dtype
                )

    if activation_kind == "float":
        dtype = getattr(torch, activation_dtype)
        for qmodule in sim.qmodules():
            for i, qtzr in enumerate(qmodule.input_quantizers):
                if not qtzr:
                    continue
                qmodule.input_quantizers[i] = Q.float.FloatQuantizeDequantize(
                    dtype=dtype
                )

        for qmodule in sim.qmodules():
            for i, qtzr in enumerate(qmodule.output_quantizers):
                if not qtzr:
                    continue
                qmodule.output_quantizers[i] = Q.float.FloatQuantizeDequantize(
                    dtype=dtype
                )

    sim.compute_encodings(lambda model: model(x))

    # Compute original pytorch model output with qdq weights
    with (
        _concretize_int32_bias_quantizers(sim.model, x)
        if export_int32_bias
        else contextlib.nullcontext()
    ):
        expected_param_encodings = {
            f"{module_name}.{param_name}": qtzr.get_encodings().to_qnn_encoding_dict(
                encoding_version
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

    with tempfile.TemporaryDirectory() as dirname:
        onnx_path = os.path.join(dirname, "torchvision_model.onnx")
        encodings_path = os.path.join(dirname, "torchvision_model.encodings")

        with set_encoding_version(encoding_version):
            sim.onnx.export(
                x,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                export_int32_bias=export_int32_bias,
                dynamo=False,
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

        """
        Then: The onnx encodings should have the same number of encodings
              as the number of quantizers in the original pytorch model
        """
        if encoding_version < "2.0.0":
            assert len(onnx_encodings["param_encodings"]) == len(
                expected_param_encodings
            )
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
            assert len(onnx_encodings["encodings"]) >= len(
                expected_param_encodings
            ) + len(expected_activation_encodings)

        """
        Then: The onnx encodings should have the same scale and offset value
              as the values of quantizers in the original pytorch model
        """
        if encoding_version == "0.6.1":
            assert onnx_encodings["param_encodings"] == expected_param_encodings

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
                assert e == expected_param_encodings[name]

            for e in onnx_encodings["activation_encodings"]:
                assert any(
                    e["scale"] == expected["scale"]
                    and e["offset"] == expected["offset"]
                    and e["bw"] == expected["bw"]
                    for expected in expected_activation_encodings.values()
                )
        elif encoding_version == "2.0.0":
            expected_encodings = (
                expected_param_encodings | expected_activation_encodings
            )

            for e in onnx_encodings["encodings"]:
                name = e.pop("name")
                if name in expected_encodings:
                    assert e == expected_encodings[name]
                    continue

                assert any(
                    e.get("output_dtype") == expected.get("output_dtype")
                    and e.get("y_scale") == expected.get("y_scale")
                    and e.get("y_zero_point") == expected.get("y_zero_point")
                    and e.get("per_channel_float_scale")
                    == expected.get("per_channel_float_scale")
                    and e.get("per_block_int_scale")
                    == expected.get("per_block_int_scale")
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


def _parse_type(type_str: str) -> tuple[str, int]:
    if type_str.startswith("int"):
        return "int", int(type_str[3:])
    if type_str.startswith("uint"):
        return "uint", int(type_str[4:])
    if type_str.startswith("float"):
        return "float", int(type_str[5:])
    raise RuntimeError


@pytest.mark.parametrize("lpbq", [False])
@pytest.mark.parametrize("fold_param_quantizers", [False, True])
@pytest.mark.parametrize("export_int32_bias", [True, False])
@pytest.mark.parametrize(
    "param_dtype, activation_dtype",
    [
        ("int8", "uint8"),
        ("int8", "uint16"),
        ("int8", "float16"),
        ("float16", "float16"),
    ],
)
def test_quantsim_export_onnx_qdq_resnet18(
    lpbq: bool,
    export_int32_bias: bool,
    fold_param_quantizers: bool,
    param_dtype: str,
    activation_dtype: str,
):
    """
    When: Export quantized torchvision model using quantsim.export
    """
    x = torch.randn(1, 3, 224, 224)
    model = resnet18().eval()
    model = prepare_model(model)
    fold_all_batch_norms(model, None, x)

    param_kind, param_bw = _parse_type(param_dtype)
    activation_kind, activation_bw = _parse_type(activation_dtype)
    sim = QuantizationSimModel(
        model, x, default_param_bw=param_bw, default_output_bw=activation_bw
    )

    if lpbq:
        set_grouped_blockwise_quantization_for_weights(
            sim,
            [sim.model.fc],
            bitwidth=4,
            symmetric=True,
            decompressed_bw=8,
            block_size=64,
        )

    if param_kind == "float":
        dtype = getattr(torch, param_dtype)
        for qmodule in sim.qmodules():
            for name, qtzr in qmodule.param_quantizers.items():
                if not qtzr:
                    continue
                qmodule.param_quantizers[name] = Q.float.FloatQuantizeDequantize(
                    dtype=dtype
                )

    if activation_kind == "float":
        dtype = getattr(torch, activation_dtype)
        for qmodule in sim.qmodules():
            for i, qtzr in enumerate(qmodule.input_quantizers):
                if not qtzr:
                    continue
                qmodule.input_quantizers[i] = Q.float.FloatQuantizeDequantize(
                    dtype=dtype
                )

        for qmodule in sim.qmodules():
            for i, qtzr in enumerate(qmodule.output_quantizers):
                if not qtzr:
                    continue
                qmodule.output_quantizers[i] = Q.float.FloatQuantizeDequantize(
                    dtype=dtype
                )

    sim.compute_encodings(lambda model: model(x))

    with (
        _concretize_int32_bias_quantizers(sim.model, x)
        if export_int32_bias
        else contextlib.nullcontext()
    ):
        expected_out = sim.model(x)
        sim_qdq_nodes = [
            q
            for q in sim.model.modules()
            if isinstance(q, (Q.affine.QuantizeDequantize, Q.affine.Dequantize))
        ]

    if fold_param_quantizers:
        sim.fold_param_quantizers()

    with tempfile.TemporaryDirectory() as dirname:
        onnx_path = os.path.join(dirname, "torchvision_model.onnx")
        aimet_torch.onnx.export(
            sim,
            x,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=21,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            export_int32_bias=export_int32_bias,
            dynamo=False,
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
        assert len(onnx_dq_nodes) >= len(sim_qdq_nodes)

        if activation_kind in ("uint", "int"):
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

    if activation_kind in ("uint", "int"):
        # Allow off-by-3 error
        atol = sim.model.fc.output_quantizers[0].get_scale().item() * 3
    else:
        # Allow off-by-3 error, using float16.eps as a pseudo-scale
        atol = torch.finfo(torch.float16).eps * 3

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


def test_non_standard_quantizer():
    """
    When: Export model with LPBQ quantizer
    Then: Should throw NotImplementedError
    """
    model = torch.nn.Sequential(torch.nn.Linear(16, 16))
    x = torch.zeros(16, 16)
    sim = QuantizationSimModel(model, x)
    set_grouped_blockwise_quantization_for_weights(
        sim, [sim.model[0]], bitwidth=4, symmetric=True, decompressed_bw=8, block_size=4
    )

    with pytest.raises(NotImplementedError):
        aimet_torch.onnx.export(sim.model, x, f=os.devnull, dynamo=False)

    """
    When: Export model with non-standard-bitwidth quantizer
    Then: Should throw RuntimeError
    """
    sim = QuantizationSimModel(model, x)
    sim.model[0].param_quantizers["weight"].bitwidth = 9

    with pytest.raises(RuntimeError):
        aimet_torch.onnx.export(sim.model, x, f=os.devnull, dynamo=False)


def test_data_movement_op_encoding_generation():
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
            dynamo=False,
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
        with patch("aimet_torch.onnx._derive_data_movement_op_encoding", lambda *_: {}):
            aimet_torch.onnx.export(
                sim.model,
                x,
                full_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                dynamo=False,
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
    When: Call _derive_data_movement_op_encoding
    Then: Output encodings should not be reused for input quantization
    """
    new_encodings = _derive_data_movement_op_encoding(
        model,
        {
            "reshape_output": (
                Q.affine.AffineEncoding(
                    torch.ones(()), torch.zeros(()), qmin=0, qmax=255, symmetry=False
                ),
                False,
            ),
            "split_output_0": (
                Q.affine.AffineEncoding(
                    torch.ones(()), torch.zeros(()), qmin=0, qmax=255, symmetry=False
                ),
                False,
            ),
        },
    )

    assert not new_encodings


def test_back_to_back_qdq(tmp_path):
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
    with pytest.raises(NotImplementedError):
        aimet_torch.onnx.export(
            sim.model,
            input,
            "qdq_model.onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=21,
            dynamo=False,
        )

    with pytest.raises(NotImplementedError):
        sim.onnx.export(
            input,
            "qdq_model.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )

    # TODO: Uncomment this when AIMET begins to support exporting back-to-back QDQ
    # """
    # Then: onnx graph should look like this:

    #     weight -> QDQ ---V
    #     input --> QDQ -> Gemm -> QDQ ----> QDQ -> Softmax -> QDQ -> output
    #     bias_q -> DQ ----^     (8-bit)   (16-bit)
    # """
    # onnx_model = onnx.load_model("qdq_model.onnx")
    # num_dq = len([dq for dq in onnx_model.graph.node if dq.op_type == "DequantizeLinear"])
    # assert num_dq == 6, f"Expected 6 DequantizeLinear nodes, but got {num_dq}"

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
        dynamo=False,
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


@pytest.fixture(scope="module")
def large_model():
    return torch.nn.Sequential(
        torch.nn.Linear(2**15, 2**14, bias=False)  # 0.5B parameters = 2GB
    )


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield pathlib.Path(tmpdir).resolve()


@torch.no_grad()
@pytest.mark.parametrize("opset_version", [19, 21])
@pytest.mark.parametrize("prequantize_constants", [False, True])
def test_export_large_model(
    large_model: torch.nn.Module,
    opset_version: int,
    prequantize_constants: bool,
    tmp_path: pathlib.Path,
):
    if opset_version > 21 and parse(torch.__version__) < parse("2.2.0"):
        pytest.skip("Skipping temporarily to stabilize CI")

    """
    Given: model that exceeds 2GB
    """
    x = torch.randn(1, 2**15)
    sim = QuantizationSimModel(
        large_model, x, config_file="htp_quantsim_config_v81_per_channel_linear.json"
    )
    sim.compute_encodings(lambda model: model(x))

    onnx_path = os.path.join(tmp_path, "qdq_model.onnx")

    """
    When: Export encoding with sim.onnx.export
    Then: All encoding should be exported correctly
    """
    with set_encoding_version("2.0.0"):
        sim.onnx.export(
            x,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )

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
    When: Export to onnx QDQ
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
        dynamo=False,
    )

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

    atol = sim.model[-1].output_quantizers[0].get_scale().item()
    assert torch.allclose(torch.from_numpy(out), expected_out, atol=atol)


def test_output_split(tmp_path):
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
        dynamo=False,
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
