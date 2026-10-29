# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic Torch LLM class"""

from __future__ import annotations

import torch

from aimet_torch.common.defs import QuantScheme
from aimet_torch.onnx_utils import map_torch_types_to_onnx
from aimet_torch.v2.nn.true_quant import (
    QuantizationMixin,
    QuantizedConv2d,
    QuantizedLinear,
)
from aimet_torch.v2.utils import remove_activation_quantizers
from aimet_torch import QuantizationSimModel

from GenAILab.bench.precision import (
    PrecisionConfig,
    float16,
    float32,
    int16,
)
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.qai_hub_lm.models.base import LLM

from GenAILab.qai_hub_lm.backends.torch.quantsim_utils import (
    _apply_block_granularity_to_decoder_stack,
    _remove_decoder_block_weight_quantizers,
    _set_lm_head_precision,
)


@YAMLConfigParser.register_default_llm
class LLM_Torch(LLM):
    """Generic LLM for AIMET-Torch quantization."""

    @classmethod
    def instantiate_quantsim(
        cls,
        model,
        context_length: int,
        sequence_length: int | list[int],
        precision: PrecisionConfig | None = None,
        *args,
        **kwargs,
    ) -> SimCollection:
        if precision is None:
            precision = PrecisionConfig()

        max_sequence_length = (
            max(sequence_length)
            if isinstance(sequence_length, list)
            else sequence_length
        )

        block_prec = precision.blocks["default"]
        # default_param_bw must be an int; when block weights are FP we strip
        # the weight quantizers below, so the value here is a placeholder.
        default_param_bw = 8 if block_prec.is_float else block_prec.qtype.bits
        default_output_bw = (
            16
            if precision.activations in (float16, float32)
            else precision.activations.bits
        )

        # Wrap model to enable JIT trace
        layer_cache_descs = build_layer_cache_descriptors(model.config)
        traceable_model = ONNXExportableModuleWithCache(
            model,
            cache_type=cls.get_cache_type(),
            input_names=cls.get_backbone_input_names(layer_cache_descs),
        )
        quantsim = QuantizationSimModel(
            model=traceable_model,
            quant_scheme=QuantScheme.post_training_tf,
            dummy_input=cls.get_sample_backbone_inputs(
                traceable_model, context_length, max_sequence_length
            ),
            default_output_bw=default_output_bw,
            default_param_bw=default_param_bw,
            in_place=True,
            config_file=QUANTSIM_CONFIG,
        )

        if precision.activations in (float16, float32):
            remove_activation_quantizers(quantsim.model)

        # Configure RMS Norm weights to 16-bits
        for module in quantsim.model.modules():
            if (
                cls._is_quantized_rms_norm(module)
                and "weight" in module.param_quantizers
            ):
                module.param_quantizers["weight"].bitwidth = 16

        # Set LM Head precision if specified
        _set_lm_head_precision(quantsim, precision.lm_head)
        # If block weights are FP, drop their weight quantizers entirely.
        # Otherwise, apply block-level granularity (LPBQ/BQ) if configured.
        if block_prec.is_float:
            _remove_decoder_block_weight_quantizers(quantsim)
        else:
            _apply_block_granularity_to_decoder_stack(quantsim, precision)

        # Plain (non-VLM) LLMs do not wire embed_tokens into SimCollection,
        # so precision.embedding cannot actually be applied. Reject any
        # override to avoid silently ignoring user intent. VLM subclasses
        # honor it via their own instantiate_quantsim overrides.
        if precision.embedding != int16:
            raise NotImplementedError(
                "Embedding quantization other than int16 is not currently "
                "supported for plain LLMs."
            )

        return SimCollection(
            backbone=quantsim, config=model.config, extras=cls.get_extras(model)
        )

    @staticmethod
    def _is_quantized_rms_norm(module: torch.nn.Module) -> bool:
        """Check if the given module is a quantized RMSNormalization layer."""
        return isinstance(
            module, QuantizationMixin
        ) and "RMSNormalization" in map_torch_types_to_onnx.get(type(module), [])
