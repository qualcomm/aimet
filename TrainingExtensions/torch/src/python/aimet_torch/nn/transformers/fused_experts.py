# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quantization of transformers fused MoE experts"""

import torch

from aimet_torch.nn.true_quant import QuantizationMixin


class _FusedExpertsQuantizationMixin(QuantizationMixin):
    """
    QuantizationMixin for transformers>=5.0.0 fused MoE experts, which store the weights of all
    experts of a layer as 3D tensors [num_experts, ...]:

    1) gate_up_proj: [num_experts, 2 * intermediate, hidden]
    2) down_proj: [num_experts, hidden, intermediate]

    With per-channel quantization, each weight is quantized per expert and per output channel,
    which gives the same encodings as quantizing each expert as its own nn.Linear.

    NOTE: Without per-channel quantization, one encoding is shared by all experts, and by the gate
          and up projections of gate_up_proj, whereas each expert's nn.Linear had its own.

    NOTE: Activations are quantized at the hidden_states input and at the module output.
          The act_fn submodule is quantized as its own module, so its output is quantized too,
          with one encoding shared by all experts. The other activations between the expert
          projections are not quantized.
    """

    def __quant_init__(self):
        super().__quant_init__()

        # Declare the number of input/output quantizers.
        # forward(hidden_states, top_k_index, top_k_weights): only hidden_states is quantized.
        # top_k_index holds integer expert indices, and the routing weights are left unquantized
        # as they were in transformers<5.0.0, where they never entered a quantized module.
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

        # The axes below assume [E, O, I] weights without biases, the layout of
        # Qwen3MoeExperts and Qwen3_5MoeExperts. Raise upfront and extend as we see more
        # such experts in the future.
        param_names = {name for name, _ in self.named_parameters(recurse=False)}
        if self.is_transposed or param_names != {"gate_up_proj", "down_proj"}:
            raise RuntimeError(
                f"{type(self).__qualname__} has an unsupported experts layout "
                f"(is_transposed={self.is_transposed}, parameters={sorted(param_names)}). "
                "_FusedExpertsQuantizationMixin supports gate_up_proj and down_proj of shape "
                "[num_experts, out_features, in_features] only for now."
            )

        # Quantize along the expert axis and the output channel axis
        self._param_channel_axes = {"gate_up_proj": (0, 1), "down_proj": (0, 1)}

    # pylint: disable-next=arguments-differ
    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Quantize input tensors
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)

        # Run forward with quantized parameters.
        # NOTE: This QDQs the weights of all experts, including those no token is routed to,
        #       so it dominates the forward time when only a few experts are hit (e.g. decoding)
        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states, top_k_index, top_k_weights)

        # Quantize output tensors
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)

        return ret
