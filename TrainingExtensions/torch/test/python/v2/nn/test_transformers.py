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
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.internvl import modeling_internvl
import aimet_torch


FUSED_EXPERTS_CLASSES = [
    (modeling_qwen3_moe.Qwen3MoeExperts, Qwen3MoeConfig),
    (modeling_qwen3_5_moe.Qwen3_5MoeExperts, Qwen3_5MoeTextConfig),
]


class _ExpertsWrapper(torch.nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return self.experts(hidden_states, top_k_index, top_k_weights)


def _make_fused_experts(experts_cls, config_cls):
    config = config_cls(
        hidden_size=32, moe_intermediate_size=8, num_experts=4, num_experts_per_tok=2
    )
    torch.manual_seed(0)
    experts = experts_cls(config)
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.1)
        experts.down_proj.normal_(0, 0.1)

    num_tokens = 16
    hidden_states = torch.randn(num_tokens, config.hidden_size)
    # Route every token to experts (t, t + 1) so that every expert is hit
    token_ids = torch.arange(num_tokens)
    top_k_index = torch.stack(
        [token_ids % config.num_experts, (token_ids + 1) % config.num_experts], dim=-1
    )
    top_k_weights = torch.rand(num_tokens, config.num_experts_per_tok).softmax(dim=-1)
    return experts, (hidden_states, top_k_index, top_k_weights)


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


def _topk_router_classes():
    """Top-k router classes available in the installed transformers."""
    classes = []
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe

        classes.append(modeling_qwen3_5_moe.Qwen3_5MoeTopKRouter)
    except (ImportError, AttributeError):
        pass
    return classes


@pytest.mark.parametrize("router_cls", _topk_router_classes())
def test_topk_router_quantsim(router_cls):
    """
    When: Create quantsim with a MoE top-k router
    Then: The projection weight and the logits are quantized; the top-k scores
          and integer indices are not.

    Without a registered definition this raises UnknownModuleError.
    """

    class Config:
        num_experts_per_tok = 2
        num_experts = 8
        hidden_size = 32
        norm_topk_prob = True

    class Wrapper(torch.nn.Module):
        def __init__(self, router):
            super().__init__()
            self.router = router

        def forward(self, x):
            logits, scores, indices = self.router(x)
            return logits, scores, indices

    torch.manual_seed(0)
    router = router_cls(Config())
    with torch.no_grad():
        router.weight.normal_(0, 0.05)

    model = Wrapper(router)
    x = torch.randn(4, Config.hidden_size)
    sim = aimet_torch.QuantizationSimModel(model, x)

    qrouter = sim.model.router
    assert qrouter.param_quantizers["weight"] is not None
    assert len(qrouter.output_quantizers) == 1

    with aimet_torch.nn.compute_encodings(sim.model):
        sim.model(x)

    assert qrouter.param_quantizers["weight"].is_initialized()

    logits, scores, indices = sim.model(x)
    assert indices.dtype == torch.int64
    assert torch.isfinite(logits).all()
    assert torch.isfinite(scores).all()
    _, ref_indices = torch.topk(
        torch.softmax(router.weight @ x[0], dim=-1), Config.num_experts_per_tok
    )
    assert ref_indices.numel() == Config.num_experts_per_tok


@pytest.mark.parametrize("experts_cls, config_cls", FUSED_EXPERTS_CLASSES)
def test_fused_experts_quantsim(experts_cls, config_cls):
    """
    When: Create quantsim with fused experts
    Then: 1) Each fused weight is quantized per expert and per output channel
          2) The encodings and output match quantizing each expert as its own nn.Linear,
             with every transformers experts kernel
    """
    from aimet_torch.quantization.affine import QuantizeDequantize

    experts, inputs = _make_fused_experts(experts_cls, config_cls)
    num_experts, two_inter, hidden = experts.gate_up_proj.shape
    inter = two_inter // 2

    sim = aimet_torch.QuantizationSimModel(
        _ExpertsWrapper(experts), inputs, default_param_bw=4
    )
    qexperts = sim.model.experts
    assert isinstance(qexperts, experts_cls)
    assert type(qexperts) is not experts_cls
    sim.compute_encodings(lambda model, _: model(*inputs), None)

    gate_up_qtzr = qexperts.param_quantizers["gate_up_proj"]
    down_qtzr = qexperts.param_quantizers["down_proj"]
    assert gate_up_qtzr.shape == (num_experts, two_inter, 1)
    assert down_qtzr.shape == (num_experts, hidden, 1)

    # Reference: each expert as gate/up/down nn.Linear with a per-channel weight quantizer
    def qdq_linear_weight(weight):
        qtzr = QuantizeDequantize((weight.shape[0], 1), 4, symmetric=True)
        with qtzr.compute_encodings():
            qtzr(weight)
        return qtzr(weight).dequantize(), qtzr.get_scale()

    with torch.no_grad():
        gate_up_qdq_ref, down_qdq_ref = [], []
        for e in range(num_experts):
            gate, gate_scale = qdq_linear_weight(experts.gate_up_proj[e, :inter])
            up, up_scale = qdq_linear_weight(experts.gate_up_proj[e, inter:])
            down, down_scale = qdq_linear_weight(experts.down_proj[e])
            assert torch.allclose(gate_up_qtzr.get_scale()[e, :inter], gate_scale)
            assert torch.allclose(gate_up_qtzr.get_scale()[e, inter:], up_scale)
            assert torch.allclose(down_qtzr.get_scale()[e], down_scale)
            gate_up_qdq_ref.append(torch.cat([gate, up]))
            down_qdq_ref.append(down)

        # Only hidden_states is input-quantized, not the expert indices or routing weights
        assert len(qexperts.input_quantizers) == 1
        assert qexperts.input_quantizers[0].is_initialized()
        assert qexperts.output_quantizers[0].is_initialized()

        # Activation quantizers are disabled to compare the effect of weight quantization only
        qexperts.input_quantizers[0] = None
        qexperts.output_quantizers[0] = None
        qexperts.act_fn.output_quantizers[0] = None

        # Reference: float experts holding the per-nn.Linear QDQ'd weights, run with eager kernel
        reference = experts_cls(experts.config)
        reference.gate_up_proj.copy_(torch.stack(gate_up_qdq_ref))
        reference.down_proj.copy_(torch.stack(down_qdq_ref))
        reference.config._experts_implementation = "eager"
        expected = reference(*inputs)

        # Weights are QDQ'd before dispatch, so every transformers experts kernel must match
        for impl in ("eager", "batched_mm", "grouped_mm"):
            qexperts.config._experts_implementation = impl
            assert torch.allclose(sim.model(*inputs), expected, atol=1e-6), impl


@pytest.mark.parametrize("experts_cls, config_cls", FUSED_EXPERTS_CLASSES)
def test_fused_experts_quantsim_config(experts_cls, config_cls):
    """
    When: Create quantsim with fused experts
    Then: Fused weights follow the per-channel setting of MatMul in the config file
    """
    from aimet_torch.onnx_utils import map_torch_types_to_onnx

    experts, inputs = _make_fused_experts(experts_cls, config_cls)
    model = _ExpertsWrapper(experts)
    per_channel_shape = (*experts.gate_up_proj.shape[:2], 1)

    for config_file in ("default", "htp_v81"):
        sim = aimet_torch.QuantizationSimModel(model, inputs, config_file=config_file)
        assert (
            sim.model.experts.param_quantizers["gate_up_proj"].shape
            == per_channel_shape
        )

    # Config with per-tensor MatMul
    sim = aimet_torch.QuantizationSimModel(model, inputs, config_file="htp_v66")
    assert sim.model.experts.param_quantizers["gate_up_proj"].shape == ()

    # Without the MatMul mapping, the per-channel setting would not reach the fused weights
    qcls = aimet_torch.nn.QuantizationMixin.cls_to_qcls[experts_cls]
    assert map_torch_types_to_onnx[qcls] == ["MatMul"]
    onnx_types = map_torch_types_to_onnx.pop(qcls)
    try:
        sim = aimet_torch.QuantizationSimModel(model, inputs)
        assert sim.model.experts.param_quantizers["gate_up_proj"].shape == ()
    finally:
        map_torch_types_to_onnx[qcls] = onnx_types


@pytest.mark.parametrize("experts_cls, config_cls", FUSED_EXPERTS_CLASSES)
def test_fused_experts_unsupported_layout(experts_cls, config_cls):
    """
    When: Create quantsim with fused experts whose layout differs from
          [num_experts, out_features, in_features] weights without biases
    Then: Raise instead of quantizing along wrong channel axes
    """
    experts, inputs = _make_fused_experts(experts_cls, config_cls)
    experts.is_transposed = True
    with pytest.raises(
        RuntimeError, match="unsupported experts layout.*is_transposed=True"
    ):
        aimet_torch.QuantizationSimModel(_ExpertsWrapper(experts), inputs)

    experts, inputs = _make_fused_experts(experts_cls, config_cls)
    experts.gate_up_proj_bias = torch.nn.Parameter(
        torch.zeros(experts.gate_up_proj.shape[:2])
    )
    with pytest.raises(
        RuntimeError, match="unsupported experts layout.*gate_up_proj_bias"
    ):
        aimet_torch.QuantizationSimModel(_ExpertsWrapper(experts), inputs)


@pytest.mark.parametrize("experts_cls, config_cls", FUSED_EXPERTS_CLASSES)
def test_fused_experts_backward(experts_cls, config_cls):
    """
    When: Run backward through quantized fused experts
    Then: Every expert receives a gradient through the fused weight quantizers
    """
    experts, inputs = _make_fused_experts(experts_cls, config_cls)
    sim = aimet_torch.QuantizationSimModel(_ExpertsWrapper(experts), inputs)
    sim.compute_encodings(lambda model, _: model(*inputs), None)

    sim.model(*inputs).sum().backward()
    qexperts = sim.model.experts
    for name in ("gate_up_proj", "down_proj"):
        grad = getattr(qexperts, name).grad
        assert grad is not None
        # grad [E, O, I]: each expert's slice is non-zero, since every expert gets routed tokens
        assert (grad.abs().sum(dim=(1, 2)) > 0).all()


@pytest.mark.parametrize("experts_cls, config_cls", FUSED_EXPERTS_CLASSES)
def test_fused_experts_folded_checkpoint(experts_cls, config_cls):
    """
    When: Fold the fused weight quantizers and save the state dict
    Then: 1) The state dict keeps the transformers parameter names
          2) Slicing the folded weights into per-expert nn.Linear weights reproduces the
             quantized output, and the sliced weights are on the per-channel quantization grid
    """
    from aimet_torch.quantization.affine import QuantizeDequantize

    experts, inputs = _make_fused_experts(experts_cls, config_cls)
    num_experts, two_inter, _ = experts.gate_up_proj.shape
    inter = two_inter // 2

    # 4-bit, so that weights off the quantization grid fail the tolerances below
    sim = aimet_torch.QuantizationSimModel(
        _ExpertsWrapper(experts), inputs, default_param_bw=4
    )
    sim.compute_encodings(lambda model, _: model(*inputs), None)

    # Disable experts input/output and intermediate activation quantizers, so that only weight
    # quantization differs from the unfused nn.Linear layers below
    qexperts = sim.model.experts
    qexperts.input_quantizers[0] = None
    qexperts.output_quantizers[0] = None
    qexperts.act_fn.output_quantizers[0] = None

    with torch.no_grad():
        expected = sim.model(*inputs)  # weights QDQ'd on the fly in the forward
        sim.fold_param_quantizers()  # weights replaced by their QDQ'd values
        assert torch.equal(sim.model(*inputs), expected)

    # The checkpoint keeps the transformers parameter names, which the unfusing code slices by
    state_dict = sim.model.state_dict()
    assert {"experts.gate_up_proj", "experts.down_proj"} <= set(state_dict)

    # Unfuse into per-expert nn.Linear layers. as_subclass drops the DequantizedTensor wrapper of
    # the folded weights
    gate_up = state_dict["experts.gate_up_proj"].as_subclass(torch.Tensor)
    down = state_dict["experts.down_proj"].as_subclass(torch.Tensor)
    hidden_states, top_k_index, top_k_weights = inputs
    out = torch.zeros_like(hidden_states)
    with torch.no_grad():
        for e in range(num_experts):
            linears = []
            # gate_up_proj [E, 2I, H] holds the gate rows first, then the up rows
            for weight in (gate_up[e, :inter], gate_up[e, inter:], down[e]):
                linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
                linear.weight.copy_(weight)
                linears.append(linear)

                # PCQ leaves the sliced weight unchanged
                qtzr = QuantizeDequantize((weight.shape[0], 1), 4, symmetric=True)
                with qtzr.compute_encodings():
                    qtzr(weight)
                assert torch.allclose(qtzr(weight).dequantize(), weight, atol=1e-7)

            # Run expert e on its routed tokens, as the transformers eager kernel does
            gate_proj, up_proj, down_proj = linears
            token_idx, top_k_pos = torch.where(top_k_index == e)
            x = hidden_states[token_idx]
            y = down_proj(experts.act_fn(gate_proj(x)) * up_proj(x))
            out.index_add_(0, token_idx, y * top_k_weights[token_idx, top_k_pos, None])

    assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("experts_cls, config_cls", FUSED_EXPERTS_CLASSES)
def test_fused_experts_original_model(experts_cls, config_cls, tmp_path):
    """
    When: Get the original model of quantsim with fused experts
    Then: 1) No quantized module is left, including the act_fn submodule of the fused experts
          2) Weights can be exported to safetensors
    """
    experts, inputs = _make_fused_experts(experts_cls, config_cls)
    sim = aimet_torch.QuantizationSimModel(_ExpertsWrapper(experts), inputs)
    sim.compute_encodings(lambda model, _: model(*inputs), None)

    original_model = sim.get_original_model(sim.model)
    assert not any(
        isinstance(m, aimet_torch.nn.BaseQuantizationMixin)
        for m in original_model.modules()
    )

    sim.export_weights_to_safetensors(str(tmp_path), "model")
    assert (tmp_path / "model.safetensors").exists()


@pytest.mark.parametrize("experts_cls, config_cls", FUSED_EXPERTS_CLASSES)
@pytest.mark.parametrize("lpbq", [False, True])
def test_fused_experts_blockwise(experts_cls, config_cls, lpbq):
    """
    When: Set blockwise (BQ) or grouped blockwise (LPBQ) quantization for fused experts
    Then: 1) Each fused weight is quantized in blocks along the input channel axis
          2) The encodings match quantizing each expert as its own nn.Linear with the same
             block size
    """
    from aimet_torch.quantization.affine import (
        QuantizeDequantize,
        GroupedBlockQuantizeDequantize,
    )
    from aimet_torch.quantsim.config_utils import (
        set_blockwise_quantization_for_weights,
        set_grouped_blockwise_quantization_for_weights,
    )

    experts, inputs = _make_fused_experts(experts_cls, config_cls)
    num_experts, two_inter, hidden = experts.gate_up_proj.shape
    inter = two_inter // 2
    block_size = 4

    sim = aimet_torch.QuantizationSimModel(
        _ExpertsWrapper(experts), inputs, default_param_bw=4
    )
    if lpbq:
        set_grouped_blockwise_quantization_for_weights(
            sim, [experts_cls], 4, True, 8, block_size, -1
        )
    else:
        set_blockwise_quantization_for_weights(sim, [experts_cls], 4, True, block_size)
    sim.compute_encodings(lambda model, _: model(*inputs), None)

    qexperts = sim.model.experts
    gate_up_qtzr = qexperts.param_quantizers["gate_up_proj"]
    down_qtzr = qexperts.param_quantizers["down_proj"]
    assert gate_up_qtzr.shape == (num_experts, two_inter, hidden // block_size)
    assert down_qtzr.shape == (num_experts, hidden, inter // block_size)
    for qtzr in (gate_up_qtzr, down_qtzr):
        assert type(qtzr) is (
            GroupedBlockQuantizeDequantize if lpbq else QuantizeDequantize
        )
        assert qtzr.block_size == [1, 1, block_size]
        if lpbq:
            assert qtzr.block_grouping == [1, 1, -1]

    # Reference: each expert as gate/up/down nn.Linear with a blockwise weight quantizer
    def linear_weight_scale(weight):
        shape = (weight.shape[0], weight.shape[1] // block_size)
        if lpbq:
            qtzr = GroupedBlockQuantizeDequantize(
                shape, 4, True, 8, None, (1, block_size), (1, -1)
            )
        else:
            qtzr = QuantizeDequantize(shape, 4, True, block_size=(1, block_size))
        with qtzr.compute_encodings():
            qtzr(weight)
        return qtzr.get_scale()

    with torch.no_grad():
        for e in range(num_experts):
            gate_scale = linear_weight_scale(experts.gate_up_proj[e, :inter])
            up_scale = linear_weight_scale(experts.gate_up_proj[e, inter:])
            down_scale = linear_weight_scale(experts.down_proj[e])
            assert torch.equal(gate_up_qtzr.get_scale()[e, :inter], gate_scale)
            assert torch.equal(gate_up_qtzr.get_scale()[e, inter:], up_scale)
            assert torch.equal(down_qtzr.get_scale()[e], down_scale)


@pytest.mark.parametrize("experts_cls, config_cls", FUSED_EXPERTS_CLASSES)
def test_fused_experts_blockwise_invalid_block_size(experts_cls, config_cls):
    """
    When: Set blockwise quantization for fused experts with a block size that
          one of the fused weights doesn't divide evenly into
    Then: Raise, naming the fused weight that doesn't align
    """
    from aimet_torch.quantsim.config_utils import set_blockwise_quantization_for_weights

    experts, inputs = _make_fused_experts(experts_cls, config_cls)
    sim = aimet_torch.QuantizationSimModel(_ExpertsWrapper(experts), inputs)

    # gate_up_proj in_features (hidden=32) divides by 16, down_proj in_features (inter=8) doesn't
    with pytest.raises(RuntimeError, match="experts has down_proj of shape"):
        set_blockwise_quantization_for_weights(sim, [experts_cls], 4, True, 16)


@pytest.mark.parametrize("experts_cls, config_cls", FUSED_EXPERTS_CLASSES)
@pytest.mark.parametrize(
    "export_api", ["sim.export", "sim.onnx.export", "aimet_torch.onnx.export"]
)
def test_fused_experts_export(experts_cls, config_cls, export_api, tmp_path):
    """
    When: Export quantsim with fused experts
    Then: Raise with a message asking to unfuse the experts, before writing any file
    """
    experts, inputs = _make_fused_experts(experts_cls, config_cls)
    sim = aimet_torch.QuantizationSimModel(_ExpertsWrapper(experts), inputs)
    sim.compute_encodings(lambda model, _: model(*inputs), None)

    with pytest.raises(RuntimeError, match=r"fused MoE experts.*\['experts'\].*unfuse"):
        if export_api == "sim.export":
            sim.export(str(tmp_path), "model", inputs)
        elif export_api == "sim.onnx.export":
            sim.onnx.export(inputs, str(tmp_path / "model.onnx"))
        else:
            aimet_torch.onnx.export(sim.model, inputs, str(tmp_path / "model.onnx"))

    assert not any(tmp_path.iterdir())


def test_qwen3_moe_model_fused_experts():
    """
    When: Create quantsim with a Qwen3 MoE model
    Then: Every fused experts module is quantized per expert and per output channel
    """
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        use_cache=False,
    )

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids).logits

    torch.manual_seed(0)
    model = Wrapper(modeling_qwen3_moe.Qwen3MoeForCausalLM(config).eval())
    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    sim = aimet_torch.QuantizationSimModel(model, input_ids)
    sim.compute_encodings(lambda model, _: model(input_ids), None)

    experts = [
        m
        for m in sim.model.modules()
        if isinstance(m, modeling_qwen3_moe.Qwen3MoeExperts)
    ]
    assert len(experts) == config.num_hidden_layers
    for qexperts in experts:
        assert type(qexperts) is not modeling_qwen3_moe.Qwen3MoeExperts
        for name in ("gate_up_proj", "down_proj"):
            qtzr = qexperts.param_quantizers[name]
            assert qtzr.is_initialized()
            assert qtzr.shape == (
                config.num_experts,
                getattr(qexperts, name).shape[1],
                1,
            )
