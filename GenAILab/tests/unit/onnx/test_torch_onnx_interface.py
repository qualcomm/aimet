# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for running bfloat16 ONNX graphs through the torch interface."""

import io

import onnx
import pytest
import torch

from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import OrtInferenceSession

from GenAILab.qai_hub_lm.backends.onnx.torch_onnx_interface import (
    TorchONNXInterface,
    mock_torch_onnx_inference,
)

CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class _Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4,), 2.0))

    def forward(self, x):
        return x * self.weight


def _onnx_model(dtype: torch.dtype, dynamic: bool = False) -> onnx.ModelProto:
    model = _Scale().to(dtype).eval()
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        (torch.ones(2, 4, dtype=dtype),),
        buf,
        input_names=["x"],
        output_names=["y"],
        opset_version=18,
        dynamo=False,
        # A symbol the output shape resolver cannot map forces ORT allocation.
        dynamic_axes={"x": {0: "n"}, "y": {0: "m"}} if dynamic else None,
    )
    buf.seek(0)
    return onnx.load_model(buf)


def _float_sim(model: onnx.ModelProto) -> QuantizationSimModel:
    """A sim over ``model`` with every quantizer disabled, i.e. float evaluation."""
    sim = QuantizationSimModel(model, providers=CUDA_PROVIDERS)
    for quantizer in sim.qc_quantize_op_dict.values():
        quantizer.enabled = False
    return sim


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestBF16Inference:
    @pytest.mark.parametrize("dynamic", [False, True])
    def test_bf16_io_round_trips_on_cuda(self, dynamic):
        sim = _float_sim(_onnx_model(torch.bfloat16, dynamic))
        x = torch.arange(8, dtype=torch.bfloat16, device="cuda").reshape(2, 4)

        y = mock_torch_onnx_inference(sim.session, x)

        assert y.dtype == torch.bfloat16
        assert y.device.type == "cuda"
        torch.testing.assert_close(y, x * 2)

    def test_interface_reports_bf16(self):
        sim = _float_sim(_onnx_model(torch.bfloat16))
        interface = TorchONNXInterface(sim, config=None)

        assert interface.dtype == torch.bfloat16

        x = torch.ones(2, 4, dtype=torch.float32, device="cuda")
        with pytest.warns(UserWarning, match="casting"):
            y = interface(x)
        assert y.dtype == torch.bfloat16

    def test_remove_quantizers_keeps_model_runnable(self):
        sim = _float_sim(_onnx_model(torch.bfloat16))
        # What the RemoveQuantization recipe does.
        sim.model.model = sim.remove_quantizers(sim.model.model)
        sim._rebuild_session()

        x = torch.ones(2, 4, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(mock_torch_onnx_inference(sim.session, x), x * 2)


def test_bf16_cpu_session_raises_clear_error():
    # ORT's CPU EP cannot even load most bf16 ops (e.g. Mul), so use a graph of
    # just an Identity, which loads, to reach the numpy-path guard.
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.BFLOAT16, [2])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.BFLOAT16, [2])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 18)]
    )
    model.ir_version = 9
    session = OrtInferenceSession(model, ["CPUExecutionProvider"])

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        mock_torch_onnx_inference(session, torch.ones(2, dtype=torch.bfloat16))


def test_fp32_cpu_session_unchanged():
    session = OrtInferenceSession(_onnx_model(torch.float32), ["CPUExecutionProvider"])
    x = torch.ones(2, 4)

    torch.testing.assert_close(mock_torch_onnx_inference(session, x), x * 2)
