# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the `remove_lora_adapters` ONNX graph pass defined in
`aimet_onnx.experimental.lora.lora_graph_passes`.
"""

import os
import tempfile

import numpy as np
import onnx
import onnx_ir
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn

from .conftest import skip_module_on_windows_arm64

skip_module_on_windows_arm64("safetensors is not available on Windows ARM64")

from aimet_onnx.experimental.lora.lora_graph_passes import (
    LORA_CONV_ADAPTER_PATTERN,
    LORA_MATMUL_ADAPTER_PATTERN,
    _find_op_chain,
    remove_lora_adapters,
)


class _TwoLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 8, bias=False)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class _LoraWrapper(nn.Module):
    """PEFT-style wrapper supporting concurrent adapters merging via chained Adds."""

    def __init__(
        self,
        base_layer: nn.Linear,
        num_adapters: int = 2,
        rank: int = 4,
        alpha: float = 8.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.ModuleList(
            nn.Linear(base_layer.in_features, rank, bias=False)
            for _ in range(num_adapters)
        )
        self.lora_B = nn.ModuleList(
            nn.Linear(rank, base_layer.out_features, bias=False)
            for _ in range(num_adapters)
        )
        self.scaling = alpha / rank
        # Non-zero init so each adapter meaningfully perturbs the output;
        # otherwise we couldn't tell removal apart from a no-op.
        for a, b in zip(self.lora_A, self.lora_B):
            nn.init.normal_(a.weight, std=0.1)
            nn.init.normal_(b.weight, std=0.1)

    def forward(self, x):
        out = self.base_layer(x)
        for a, b in zip(self.lora_A, self.lora_B):
            out = out + b(a(x)) * self.scaling
        return out


def test_remove_lora_adapters_restores_base_output_on_two_linear_model():
    """
    End-to-end verification that `remove_lora_adapters` restores pre-LoRA
    behavior on a tiny two-Linear PyTorch model.

    Flow:
      1. Build a model with two Linear layers and capture a reference forward
         pass output.
      2. Wrap one Linear layer with a PEFT-style adapter containing two
         concurrent LoRA branches (which merge via chained Add nodes).
      3. Export the wrapped model to ONNX and confirm the adapter actually
         perturbs the output (guards against a vacuous test).
      4. Strip the LoRA branches using `remove_lora_adapters`.
      5. Run ORT inference on the stripped model and assert the output
         matches the pre-LoRA reference, confirming the pass correctly
         rewires consumers of the terminal Add back to the base branch.
    """
    torch.manual_seed(0)
    model = _TwoLinearModel().eval()
    x = torch.randn(2, 16)

    with torch.no_grad():
        reference_out = model(x).numpy()

    # Wrap linear1 with TWO concurrent LoRA adapters. Wrapping linear1 (rather
    # than linear2) ensures the adapter's terminal Add has a downstream
    # consumer in the exported graph — the pass rewires Add consumers, so an
    # Add sitting directly on a graph output would leave the LoRA branch
    # reachable via the graph output.
    model.linear1 = _LoraWrapper(model.linear1, num_adapters=2, rank=4, alpha=8.0)
    model.eval()

    with torch.no_grad():
        wrapped_out_after = model(x).numpy()
    assert not np.allclose(reference_out, wrapped_out_after), (
        "LoRA adapter had no effect; test would be vacuous."
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "two_linear_lora.onnx")
        torch.onnx.export(
            model,
            (x,),
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,  # legacy exporter preserves module-qualified node names
        )
        onnx_model = onnx.load(onnx_path)

        # The base_layer's terminal op becomes the attach point. With
        # bias=False this is a single MatMul whose name contains "base_layer".
        attach_point = None
        for node in onnx_model.graph.node:
            if (
                node.op_type in ("MatMul", "Gemm")
                and "base_layer" in node.name
                and "lora" not in node.name.lower()
            ):
                attach_point = node.name
                break
        assert attach_point is not None, (
            "Could not locate base_layer attach point node in exported ONNX graph."
        )

        stripped_model = remove_lora_adapters(onnx_model, attach_points=[attach_point])

        lora_nodes = [
            n.name for n in stripped_model.graph.node if "lora" in n.name.lower()
        ]
        assert not lora_nodes, f"LoRA nodes still present after removal: {lora_nodes}"

        sess = ort.InferenceSession(
            stripped_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        stripped_out = sess.run(None, {"input": x.numpy()})[0]

    np.testing.assert_allclose(stripped_out, reference_out, rtol=1e-5, atol=1e-5)
