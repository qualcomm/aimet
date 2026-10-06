# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import json

import onnx
import pytest
import torch
from transformers.models.llama import modeling_llama
from transformers.models.phi3 import modeling_phi3
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.mistral import modeling_mistral
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.internvl import modeling_internvl
import aimet_torch


@pytest.mark.parametrize(
    "rmsnorm_cls",
    [
        modeling_llama.LlamaRMSNorm,
        modeling_phi3.Phi3RMSNorm,
        modeling_qwen2.Qwen2RMSNorm,
        modeling_qwen3.Qwen3RMSNorm,
        modeling_gemma3.Gemma3RMSNorm,
        modeling_mistral.MistralRMSNorm,
        modeling_qwen3_5.Qwen3_5RMSNorm,
        modeling_internvl.InternVLVisionRMSNorm,
    ],
)
def test_rmsnorm_quantsim_config(rmsnorm_cls):
    """
    When: Create quantsim with well-known RMSNorm classes with HTP v81 config file
    Then: RMSNorm weights should be quantized asymmetrically
    """
    rmsnorm = rmsnorm_cls(100)
    model = torch.nn.Sequential(rmsnorm)
    x = torch.randn(1, 100, 100)
    sim = aimet_torch.QuantizationSimModel(model, x, config_file="htp_v81")
    assert not sim.model[0].param_quantizers["weight"].symmetric


@pytest.mark.parametrize(
    "gated_rmsnorm_cls",
    [
        modeling_qwen3_5.Qwen3_5RMSNormGated,
    ],
)
def test_gated_rmsnorm_quantsim_config(gated_rmsnorm_cls):
    """
    When: Create quantsim with well-known RMSNorm classes with HTP v81 config file
    Then: RMSNorm weights should be quantized asymmetrically
    """

    class Wrapper(torch.nn.Module):
        def __init__(self, norm):
            super().__init__()
            self.norm = norm

        def forward(self, x, y):
            return self.norm(x, y)

    rmsnorm = gated_rmsnorm_cls(100)
    model = Wrapper(rmsnorm)
    x = torch.randn(1, 100, 100)
    y = torch.randn(1, 100)
    sim = aimet_torch.QuantizationSimModel(model, (x, y), config_file="htp_v81")
    assert not sim.model.norm.param_quantizers["weight"].symmetric


@pytest.mark.skipif(
    not hasattr(modeling_qwen3_moe, "Qwen3MoeTopKRouter"),
    reason="transformers<5.0.0 uses an nn.Linear MoE gate instead of Qwen3MoeTopKRouter",
)
@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_qwen3_moe_topk_router(norm_topk_prob, tmp_path):
    """
    When: Create quantsim with Qwen3MoeTopKRouter
    Then: Quantization matches the nn.Linear gate of transformers<5.0.0:
          1) The gate weight is quantized per-channel
          2) The pre-softmax router logits are output-quantized, so softmax and top-k
             select experts from quantized logits
          3) The routing weights and expert indices are not quantized
          4) On export, the only activation encoding in the router is on the gate MatMul output
    """
    config = Qwen3MoeConfig(
        hidden_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        norm_topk_prob=norm_topk_prob,
    )
    torch.manual_seed(0)
    router = modeling_qwen3_moe.Qwen3MoeTopKRouter(config)
    # transformers zero-initializes the router weight
    torch.nn.init.normal_(router.weight, std=0.5)
    x = torch.randn(16, config.hidden_size)

    sim = aimet_torch.QuantizationSimModel(
        router, x, config_file="htp_v81", default_param_bw=4, default_output_bw=8
    )
    sim.compute_encodings(lambda model, _: model(x), None)
    gate = sim.model

    # The gate projection is quantized per output channel (one scale per expert)
    weight_qtzr = gate.param_quantizers["weight"]
    assert weight_qtzr.is_initialized()
    assert weight_qtzr.get_scale().numel() == config.num_experts

    # One output quantizer, for the router logits
    assert len(gate.output_quantizers) == 1
    assert gate.output_quantizers[0].is_initialized()

    with torch.no_grad():
        router_logits, router_scores, router_indices = gate(x)

        # The router logits are the quantized gate projection output
        hidden_states = gate.input_quantizers[0](x) if gate.input_quantizers[0] else x
        expected_logits = gate.output_quantizers[0](
            torch.nn.functional.linear(hidden_states, weight_qtzr(gate.weight))
        )
    assert torch.equal(router_logits, expected_logits)

    # Expert selection and routing weights are computed from the quantized logits, unquantized
    probs = torch.nn.functional.softmax(expected_logits, dtype=torch.float, dim=-1)
    expected_scores, expected_indices = torch.topk(
        probs, config.num_experts_per_tok, dim=-1
    )
    if norm_topk_prob:
        expected_scores = expected_scores / expected_scores.sum(dim=-1, keepdim=True)
    assert router_indices.dtype == torch.int64
    assert torch.equal(router_indices, expected_indices)
    assert torch.equal(router_scores, expected_scores)

    # On export, the router's only activation encoding is on the gate MatMul output
    onnx_path = str(tmp_path / "model.onnx")
    sim.onnx.export(args=(x,), f=onnx_path, dynamo=False, export_int32_bias=False)
    graph = onnx.load(onnx_path).graph
    with open(str(tmp_path / "model.encodings")) as f:
        encodings = json.load(f)
    encoded = {enc["name"] for enc in encodings["activation_encodings"]}
    producers = {out: node.op_type for node in graph.node for out in node.output}
    encoded_ops = sorted(producers[name] for name in encoded if name in producers)
    assert encoded_ops.count("MatMul") == 1
    assert not {"Softmax", "TopK", "Div", "Cast"} & set(encoded_ops)

    # The exported weight encoding is also per-channel (one scale per expert)
    (weight_encoding,) = encodings["param_encodings"]
    assert weight_encoding["enc_type"] == "PER_CHANNEL"
    assert len(weight_encoding["scale"]) == config.num_experts
