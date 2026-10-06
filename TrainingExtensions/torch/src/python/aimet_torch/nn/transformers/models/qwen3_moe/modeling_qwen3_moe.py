# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantized Qwen3 MoE modules"""

import torch
import torch.nn.functional as F
from aimet_torch.nn.true_quant import QuantizationMixin, _dispatch

try:
    from transformers.models.qwen3_moe import modeling_qwen3_moe
except ImportError as exc:
    raise ImportError(
        "aimet_torch.nn.transformers.models.qwen3_moe.modeling_qwen3_moe cannot be imported. Please make sure "
        "that you have transformers >= 4.51.0 installed in your environment."
    ) from exc

from aimet_torch.utils import modules_to_treat_as_leaf
from aimet_torch.onnx_utils import map_torch_types_to_onnx

# Map Qwen3MoeRMSNorm to ONNX RMSNormalization so that
# quantsim config for RMSNormalization will be applied to Qwen3MoeRMSNorm
map_torch_types_to_onnx[modeling_qwen3_moe.Qwen3MoeRMSNorm] = ["RMSNormalization"]

# Don't simulate quantization on Qwen3RotaryEmbedding layers
QuantizationMixin.ignore(modeling_qwen3_moe.Qwen3MoeRotaryEmbedding)


@QuantizationMixin.implements(modeling_qwen3_moe.Qwen3MoeRMSNorm)
class QuantizedQwen3MoeRMSNorm(QuantizationMixin, modeling_qwen3_moe.Qwen3MoeRMSNorm):
    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])
        self.param_quantizers = torch.nn.ModuleDict({"weight": None})

    def forward(self, hidden_states):  # pylint: disable=arguments-differ
        # Quantize input tensors
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        # Run forward with quantized inputs and parameters
        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret


if hasattr(modeling_qwen3_moe, "Qwen3MoeTopKRouter"):
    # transformers>=5.0.0 replaced the ``nn.Linear`` MoE gate with Qwen3MoeTopKRouter,
    # which fuses the gate projection, softmax, and top-k selection into one module.
    # Map it to ONNX Gemm/MatMul so that quantsim config for the gate projection
    # (e.g. per-channel weight quantization) is applied as it was to the nn.Linear gate.
    map_torch_types_to_onnx[modeling_qwen3_moe.Qwen3MoeTopKRouter] = [
        "Gemm",
        "MatMul",
    ]

    # Treat the router as a leaf in the connected graph. Without this, the connected graph traces
    # into the router and represents it as separate reshape/linear/softmax/topk ops that own no
    # module, so op-level config such as per-channel weight quantization for Gemm/MatMul never
    # reaches the router's weight quantizer. Registering it as a leaf keeps the gate's weight
    # per-channel, matching the nn.Linear gate used by transformers<5.0.0.
    modules_to_treat_as_leaf.append(modeling_qwen3_moe.Qwen3MoeTopKRouter)

    @QuantizationMixin.implements(modeling_qwen3_moe.Qwen3MoeTopKRouter)
    class QuantizedQwen3MoeTopKRouter(
        QuantizationMixin, modeling_qwen3_moe.Qwen3MoeTopKRouter
    ):
        """Quantized Qwen3 MoE top-k router"""

        def __quant_init__(self):
            super().__quant_init__()

            # transformers<5.0.0 applied the gate as an nn.Linear and inlined softmax, top-k and
            # normalization into Qwen3MoeSparseMoeBlock.forward, so the only activation quantized
            # was the gate's output: the pre-softmax router logits. Softmax and top-k then ran on
            # quantized logits, and the routing weights and expert indices were left unquantized.
            #
            # To match that, output_quantizers[0] is applied to the F.linear output inside the
            # router (see forward), not to any of the tensors the router returns.
            self.input_quantizers = torch.nn.ModuleList([None])
            self.output_quantizers = torch.nn.ModuleList([None])
            self.param_quantizers = torch.nn.ModuleDict({"weight": None})

        def forward(self, hidden_states: torch.Tensor):  # pylint: disable=arguments-differ
            # Quantize input tensors
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
                        "Qwen3MoeTopKRouter.forward called F.linear more than once; "
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
    # quantsim model, whose gate is a QuantizedQwen3MoeTopKRouter rather than the base class.
    modules_to_treat_as_leaf.append(QuantizedQwen3MoeTopKRouter)
