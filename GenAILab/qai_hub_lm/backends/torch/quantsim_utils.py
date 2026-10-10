# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Torch quantsim utils"""

import torch

from aimet_torch.v2.nn.true_quant import QuantizedConv2d, QuantizedLinear
from aimet_torch.v2.quantsim.config_utils import (
    set_blockwise_quantization_for_weights,
    set_grouped_blockwise_quantization_for_weights,
)
from aimet_torch import QuantizationSimModel

from GenAILab.bench.precision import (
    Granularity,
    PrecisionConfig,
    WeightPrecision,
)


def _resolve_decoder_backbone(model):
    """Resolve the decoder language backbone from a raw HF model.

    Mirrors the resolution used by ``SpinQuant.apply_spinquant``: VLMs expose
    the decoder stack at ``model.model.language_model`` while plain LLMs expose
    it at ``model.model``.
    """
    return (
        model.model.language_model
        if hasattr(model.model, "language_model")
        else model.model
    )


def apply_spinquant_pre_sim(model, spinquant_config: dict | None) -> None:
    """Apply SpinQuant rotations to the raw float model before quantsim creation.

    SpinQuant fuses RMS norms and rotates float weights (R1/R2). It must run on
    the float ``nn.Module`` *before* the ``QuantizationSimModel`` is built so the
    sim wraps the rotated graph and calibrates against the rotated weights. The
    single :meth:`SpinQuant.apply_spinquant` call rotates both the decoder stack
    and (for supported VLMs) the visual encoder + merger layers, so backbone and
    visual stay consistent.

    No-op when ``spinquant_config`` is ``None`` (SpinQuant not requested).

    :param model: raw HuggingFace model, mutated in-place. For VLMs the
        ``embed_tokens`` module referenced by ``SimCollection.embedding`` is
        rotated as part of the decoder-stack pass.
    :param spinquant_config: dict of flags from the YAML SpinQuant recipe step
        (``enable_r1`` / ``enable_r2``); ``None`` to skip.
    """
    if spinquant_config is None:
        return

    # Imported lazily so the (experimental) SpinQuant dependency is only
    # required when a config actually requests it.
    from aimet_torch.spinquant.spinquant_optimizer import (
        SpinQuant as SpinQuantOptimizer,
    )

    if spinquant_config.get("enable_r3", False):
        raise NotImplementedError(
            "SpinQuant R3 online rotation is not supported for the torch framework."
        )

    # Untie embed_tokens / lm_head if they share storage — apply_spinquant
    # requires them untied.
    decoder_model = _resolve_decoder_backbone(model)
    lm_head = model.lm_head
    if decoder_model.embed_tokens.weight is lm_head.weight:
        old_weight = lm_head.weight
        lm_head.weight = torch.nn.Parameter(
            old_weight.data.clone().detach().to(old_weight.device),
            requires_grad=old_weight.requires_grad,
        )

    SpinQuantOptimizer.apply_spinquant(
        model,
        enable_r1=spinquant_config.get("enable_r1", True),
        enable_r2=spinquant_config.get("enable_r2", False),
    )


def _remove_decoder_block_weight_quantizers(
    quantsim: QuantizationSimModel, lm_head=None
):
    """Permanently disable weight quantizers on decoder-stack Linear/Conv layers.

    Used when ``precision.blocks.qtype`` is a floating-point type — the
    transformer-block weights stay in FP while activations (and lm_head)
    keep their own precision settings.
    """
    if lm_head is None:
        lm_head = quantsim.model.model.lm_head

    for module in quantsim.model.modules():
        if (
            isinstance(module, (QuantizedConv2d, QuantizedLinear))
            and module is not lm_head
            and "weight" in module.param_quantizers
        ):
            module.param_quantizers["weight"] = None


def _apply_block_granularity_to_decoder_stack(
    quantsim: QuantizationSimModel, precision: PrecisionConfig, lm_head=None
):
    """Apply block-level granularity (LPBQ/BQ) to weight quantizers if configured."""
    if lm_head is None:
        lm_head = quantsim.model.model.lm_head

    block_prec = precision.blocks["default"]
    if block_prec.is_float:
        # FP weights — nothing to configure here.
        return
    arg = lambda module: (
        isinstance(module, (QuantizedConv2d, QuantizedLinear))
        and module.param_quantizers["weight"]
        and module.param_quantizers["weight"].bitwidth == block_prec.qtype.bits
        and module is not lm_head
    )
    if block_prec.granularity == Granularity.LPBQ:
        set_grouped_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=block_prec.qtype.bits,
            symmetric=True,
            decompressed_bw=8,
            block_size=block_prec.block_size,
            block_grouping=-1,
        )
    elif block_prec.granularity == Granularity.BQ:
        set_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=block_prec.qtype.bits,
            symmetric=True,
            block_size=block_prec.block_size,
        )


def _set_lm_head_precision(
    quantsim: QuantizationSimModel, precision: WeightPrecision, lm_head=None
):
    if lm_head is None:
        lm_head = quantsim.model.model.lm_head
    if precision.is_float:
        # FP lm_head — drop the weight quantizer entirely.
        if "weight" in lm_head.param_quantizers:
            lm_head.param_quantizers["weight"] = None
        return
    arg = lambda module: (module is lm_head)
    if precision.granularity == Granularity.LPBQ:
        set_grouped_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=precision.qtype.bits,
            symmetric=True,
            decompressed_bw=8,
            block_size=precision.block_size,
            block_grouping=-1,
        )
    elif precision.granularity == Granularity.BQ:
        set_blockwise_quantization_for_weights(
            sim=quantsim,
            arg=arg,
            bitwidth=precision.qtype.bits,
            symmetric=True,
            block_size=precision.block_size,
        )
    else:
        lm_head.param_quantizers["weight"].bitwidth = precision.qtype.bits
