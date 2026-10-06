# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantized Qwen3.5 MoE modules"""

import torch
import torch.nn.functional as F
from aimet_torch.nn.true_quant import QuantizationMixin, _dispatch

try:
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
except ImportError as exc:
    raise ImportError(
        "aimet_torch.nn.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe cannot be imported. Please make sure "
        "that you have a transformers version shipping the qwen3_5_moe model installed in your environment."
    ) from exc

from aimet_torch.utils import modules_to_treat_as_leaf
from aimet_torch.onnx_utils import map_torch_types_to_onnx

# Map Qwen3_5MoeRMSNorm and Qwen3_5MoeRMSNormGated to ONNX RMSNormalization so
# that quantsim config for RMSNormalization will be applied to both variants
map_torch_types_to_onnx[modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm] = ["RMSNormalization"]
map_torch_types_to_onnx[modeling_qwen3_5_moe.Qwen3_5MoeRMSNormGated] = [
    "RMSNormalization"
]

# Don't simulate quantization on rotary embedding layers
QuantizationMixin.ignore(modeling_qwen3_5_moe.Qwen3_5MoeTextRotaryEmbedding)
QuantizationMixin.ignore(modeling_qwen3_5_moe.Qwen3_5MoeVisionRotaryEmbedding)

# Qwen3_5MoeTopKRouter fuses the gate projection, softmax, and top-k selection into one
# module. Map it to ONNX Gemm/MatMul so that quantsim config for the gate projection
# (e.g. per-channel weight quantization) is applied to it.
map_torch_types_to_onnx[modeling_qwen3_5_moe.Qwen3_5MoeTopKRouter] = [
    "Gemm",
    "MatMul",
]

# Treat the router as a leaf in the connected graph. Without this, the connected graph traces
# into the router and represents it as separate reshape/linear/softmax/topk ops that own no
# module, so op-level config such as per-channel weight quantization for Gemm/MatMul never
# reaches the router's weight quantizer. Registering it as a leaf keeps the gate's weight
# per-channel.
modules_to_treat_as_leaf.append(modeling_qwen3_5_moe.Qwen3_5MoeTopKRouter)


@QuantizationMixin.implements(modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm)
class QuantizedQwen3_5MoeRMSNorm(
    QuantizationMixin, modeling_qwen3_5_moe.Qwen3_5MoeRMSNorm
):
    """Quantized Qwen3.5 MoE RMSNorm"""

    def __quant_init__(self):
        super().__quant_init__()

        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])
        self.param_quantizers = torch.nn.ModuleDict({"weight": None})

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states)

        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret


@QuantizationMixin.implements(modeling_qwen3_5_moe.Qwen3_5MoeRMSNormGated)
class QuantizedQwen3_5MoeRMSNormGated(
    QuantizationMixin, modeling_qwen3_5_moe.Qwen3_5MoeRMSNormGated
):
    """Quantized Qwen3.5 MoE Gated RMSNorm"""

    def __quant_init__(self):
        super().__quant_init__()

        self.input_quantizers = torch.nn.ModuleList([None, None])
        self.output_quantizers = torch.nn.ModuleList([None])
        self.param_quantizers = torch.nn.ModuleDict({"weight": None})

    def forward(self, hidden_states: torch.Tensor, gate=None) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)
        if gate is not None and self.input_quantizers[1]:
            gate = self.input_quantizers[1](gate)

        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states, gate=gate)

        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret


@QuantizationMixin.implements(modeling_qwen3_5_moe.Qwen3_5MoeTopKRouter)
class QuantizedQwen3_5MoeTopKRouter(
    QuantizationMixin, modeling_qwen3_5_moe.Qwen3_5MoeTopKRouter
):
    """Quantized Qwen3.5 MoE top-k router.

    As with Qwen3-MoE, the stock router's bare ``nn.Parameter`` projection makes
    quantsim construction fail without this definition. Only the logits are
    quantized: softmax and top-k then run on quantized logits, so expert
    selection and routing weights are derived from them, while the routing
    weights and expert indices themselves are left unquantized.
    """

    def __quant_init__(self):
        super().__quant_init__()

        # output_quantizers[0] is applied to the F.linear output inside the router (see
        # forward), not to any of the tensors the router returns.
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])
        self.param_quantizers = torch.nn.ModuleDict({"weight": None})

    def forward(self, hidden_states: torch.Tensor):  # pylint: disable=arguments-differ
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        # _dispatch patches every F.linear call made while the context is active, not just
        # the gate projection. Count calls so a future transformers release that adds a
        # second F.linear to the router's forward fails loudly instead of silently sharing
        # output_quantizers[0] between unrelated linear ops.
        call_count = 0

        def quantized_linear(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError(
                    "Qwen3_5MoeTopKRouter.forward called F.linear more than once; "
                    "quantized_linear assumes a single gate projection per forward call."
                )
            router_logits = F.linear(*args, **kwargs)
            if self.output_quantizers[0]:
                router_logits = self.output_quantizers[0](router_logits)
            return router_logits

        # Run forward with quantized inputs and parameters, quantizing the router logits
        # before they reach softmax and top-k
        with (
            self._patch_quantized_parameters(),
            _dispatch(F.linear, quantized_linear),
        ):
            return super().forward(hidden_states)


# The quantized subclass must be registered too, since the connected graph is built on the
# quantsim model, whose gate is a QuantizedQwen3_5MoeTopKRouter rather than the base class.
modules_to_treat_as_leaf.append(QuantizedQwen3_5MoeTopKRouter)
