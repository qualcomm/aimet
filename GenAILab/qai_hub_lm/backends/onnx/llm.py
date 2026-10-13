# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic ONNX LLM class"""

from __future__ import annotations

import tempfile
import onnx
import torch
from transformers import AutoConfig

from aimet_onnx import quantsim
from aimet_onnx.quantsim import (
    QuantizationSimModel,
)
from aimet_onnx.utils import duplicate_shared_initializers
from aimet_onnx.llm_topology import LlmTopology
from aimet_onnx.llm_configurator import configure_llm

from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.bench.model_cache import DiskBackedModelCache, ModelCacheEntry
from GenAILab.bench.precision import (
    PrecisionConfig,
    float16,
    int16,
)
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.base import LLM, SimCollection
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache

from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
    get_onnx_model,
    load_model_components_from_disk,
    get_model_checkpoint_path,
    is_huggingface_ckpt,
)
from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import (
    _disable_float_quantizers,
    weight_precision_to_qspec,
    get_ort_providers,
    AttributePatch,
)


@YAMLConfigParser.register_default_llm
class LLM_ONNX(LLM):
    """Generic LLM for AIMET-ONNX quantization."""

    @classmethod
    def instantiate_float_model(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int | list[int],
        small_model: bool = False,
        dtype: torch.dtype = torch.float32,
        model_cache: DiskBackedModelCache | None = None,
        image_size: tuple[int, int] | None = None,
        *args,
        **kwargs,
    ) -> ModelCacheEntry:
        """Export (or load) the raw float ONNX model(s) for this model.

        Separated from :meth:`instantiate_quantsim` so the caller can transform
        the float graph (e.g. apply SpinQuant) before the sim is built. Returns
        a :class:`ModelCacheEntry` whose ``backbone`` (and, for VLMs, ``visual``
        / ``embedding``) hold the raw ONNX model(s) ready to be quantized.
        """
        cache_sl = (
            "dynamic"
            if isinstance(sequence_length, list) and len(sequence_length) > 1
            else (
                max(sequence_length)
                if isinstance(sequence_length, list)
                else sequence_length
            )
        )

        is_hf = is_huggingface_ckpt(model_id)

        # Use model cache for HuggingFace checkpoints when a cache is provided
        if is_hf:
            if model_cache is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    params = {
                        "model_id": model_id,
                        "class": cls.__name__,
                        "sequence_length": cache_sl,
                        "context_length": context_length,
                        "small_model": small_model,
                        "dtype": str(dtype),
                    }
                    key = DiskBackedModelCache.build_key(params)
                    entry = model_cache.get_or_export(
                        key,
                        lambda: cls._export_to_cache_entry(
                            model_id,
                            context_length,
                            sequence_length,
                            small_model,
                            tmpdir,
                            dtype=dtype,
                        ),
                        metadata=params,
                    )
            else:
                entry = cls._export_to_cache_entry(
                    model_id,
                    context_length,
                    sequence_length,
                    small_model,
                    get_model_checkpoint_path(model_id),
                    dtype=dtype,
                )
        else:
            onnx_model, *_ = load_model_components_from_disk(
                model_id,
                context_length=context_length,
                sequence_length=cache_sl,
            )
            config = AutoConfig.from_pretrained(model_id)
            entry = ModelCacheEntry(backbone=onnx_model, config=config)

        # Tied embeddings share one lm_head.weight initializer between the
        # embedding Gather and the lm_head MatMul; unshare it here so every
        # path (fresh export, cache hit, disk load) hands downstream pre-sim
        # techniques and ConnectedGraph a graph they accept.
        duplicate_shared_initializers(entry.backbone.graph)
        return entry

    @classmethod
    def instantiate_quantsim(
        cls,
        entry: ModelCacheEntry,
        precision: PrecisionConfig | None = None,
        *args,
        topology: LlmTopology,
        **kwargs,
    ) -> SimCollection:
        if precision is None:
            precision = PrecisionConfig()

        onnx_model = entry.backbone
        config = entry.config

        block_prec = precision.blocks["default"]
        # Float defaults apply to every parameter unless a more specific
        # precision (for example, lm_head) overrides them below.
        default_param_qtype = float16 if block_prec.is_float else block_prec.qtype
        default_activation_qtype = precision.activations

        with (
            AttributePatch(quantsim, "op_types_to_tie_qtzrs", ["Concat"]),
            AttributePatch(
                quantsim,
                "op_outputs_to_ignore",
                quantsim.op_outputs_to_ignore + ["Slice", "Constant"],
            ),
        ):
            quant_sim = QuantizationSimModel(
                model=onnx_model,
                quant_scheme="min_max",
                param_type=default_param_qtype,
                activation_type=default_activation_qtype,
                config_file=QUANTSIM_CONFIG,
                providers=get_ort_providers(
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                ),
            )

        kv_cache_type = precision.resolve_kv_cache_qtype()
        block_weight_spec = weight_precision_to_qspec(precision.blocks["default"])
        lm_head_spec = weight_precision_to_qspec(precision.lm_head)

        configure_llm(
            quant_sim,
            topology,
            kv_cache_type=kv_cache_type,
            backbone_weight_type=block_weight_spec,
            lm_head_weight_type=lm_head_spec,
        )
        _disable_float_quantizers(quant_sim)

        # Plain (non-VLM) LLMs do not wire embed_tokens into SimCollection,
        # so the runner's quantize_embedding_weights() call is always
        # skipped. Reject any override to avoid silently ignoring user
        # intent. VLM subclasses honor it via their own instantiate_quantsim
        # overrides.
        if precision.embedding != int16:
            raise NotImplementedError(
                "Embedding quantization other than int16 is not currently "
                "supported for plain LLMs."
            )

        return SimCollection(quant_sim, config=config)

    @classmethod
    def _export_to_cache_entry(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int | list[int],
        small_model: bool,
        directory: str,
        dtype: torch.dtype = torch.float32,
    ) -> ModelCacheEntry:
        """Export the torch model to ONNX in a temp dir and return a :class:`ModelCacheEntry`."""
        max_seq_len = (
            max(sequence_length)
            if isinstance(sequence_length, list)
            else sequence_length
        )

        instantiated_model = cls.instantiate_model(model_id, small_model)
        if isinstance(instantiated_model, tuple):
            if context_length != 4096 or max_seq_len != 2048:
                raise ValueError(
                    "Context length and sequence length must be 4096 and 2048 for AIHM adapted models."
                )
            assert isinstance(instantiated_model[0], onnx.ModelProto)
            return ModelCacheEntry(
                backbone=instantiated_model[0],
                config=instantiated_model[1],
            )

        assert isinstance(instantiated_model, torch.nn.Module)
        instantiated_model = instantiated_model.to(dtype=dtype)
        layer_cache_descs = build_layer_cache_descriptors(instantiated_model.config)
        exportable_model = ONNXExportableModuleWithCache(
            instantiated_model,
            cache_type=cls.get_cache_type(),
            input_names=cls.get_backbone_input_names(layer_cache_descs),
        )

        eos_ids = set()
        for src in (
            getattr(instantiated_model.config, "eos_token_id", None),
            getattr(instantiated_model.generation_config, "eos_token_id", None),
        ):
            if src is None:
                continue
            if isinstance(src, (list, tuple)):
                eos_ids.update(src)
            else:
                eos_ids.add(src)
        instantiated_model.config.eos_token_id = list(eos_ids)

        onnx_model, *_ = get_onnx_model(
            checkpoint=directory,
            fp_backbone_model=exportable_model,
            context_length=context_length,
            sequence_length=sequence_length,
            sample_input=cls.get_sample_backbone_inputs(
                exportable_model, context_length, max_seq_len
            ),
            input_names=cls.get_backbone_input_names(layer_cache_descs),
            output_names=cls.get_backbone_output_names(layer_cache_descs),
            dynamo=cls.use_dynamo_export(),
            dynamic_axes=cls.get_backbone_dynamic_axes(layer_cache_descs),
        )

        return ModelCacheEntry(
            backbone=onnx_model,
            config=instantiated_model.config,
        )
