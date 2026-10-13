# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""VLM ONNX quantization — base class and model registrations."""

from __future__ import annotations

import os
import tempfile
import warnings

import onnx
import torch
from transformers import AutoConfig

from aimet_onnx import quantsim
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.utils import duplicate_shared_initializers
from aimet_onnx.llm_topology import LlmTopology
from aimet_onnx.llm_configurator import configure_llm

from GenAILab.qai_hub_lm.backends import QUANTSIM_CONFIG
from GenAILab.bench.model_cache import DiskBackedModelCache, ModelCacheEntry
from GenAILab.bench.precision import PrecisionConfig, float16
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.components import model_components, spec
from GenAILab.qai_hub_lm.models.utils.exportable import ONNXExportableModuleWithCache
from GenAILab.qai_hub_lm.models.utils.layer_cache import (
    build_layer_cache_descriptors,
    _resolve_text_config,
)

from GenAILab.qai_hub_lm.backends.onnx.export_utils import (
    ONNX_OPSET_VERSION,
    _dynamo_export,
    check_opset_equal_to,
    consolidate_external_data,
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


class VLM_ONNX:
    """Mixin providing common ONNX VLM float-export + quantsim instantiation.

    Subclasses need only declare the class (inheriting from this mixin and
    their model-specific VLM base) and register via @YAMLConfigParser.register_model.
    All model-structure knowledge comes from methods on the VLM model class:
    get_language_model, get_lm_head, get_embedding, build_vision_wrapper, get_extras.

    The float-export step (:meth:`instantiate_float_model`) is separated from
    sim construction (:meth:`instantiate_quantsim`) so the caller can transform
    the float ONNX graph(s) (e.g. apply SpinQuant) before the sims are built.
    """

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
        audio_frames: int | None = None,
        *args,
        **kwargs,
    ) -> ModelCacheEntry:
        """Export (or load) the raw float backbone/encoder ONNX models + embedding.

        Separated from :meth:`instantiate_quantsim` so the caller can transform
        the float graph(s) (e.g. apply SpinQuant) before the sims are built.
        """
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

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

        if is_hf:
            if model_cache is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    params = {
                        "model_id": model_id,
                        "class": cls.__name__,
                        "sequence_length": cache_sl,
                        "context_length": context_length,
                        "small_model": small_model,
                        "image_size": image_size,
                        "dtype": str(dtype),
                    }
                    # Added only when set, so an audio-free model's key -- and
                    # therefore every existing on-disk cache entry -- is
                    # unchanged by audio support.
                    if audio_frames is not None:
                        params["audio_frames"] = audio_frames
                    key = DiskBackedModelCache.build_key(params)
                    entry = model_cache.get_or_export(
                        key,
                        lambda: cls._export_to_cache_entry(
                            model_id,
                            context_length,
                            sequence_length,
                            small_model,
                            tmpdir,
                            image_size=image_size,
                            audio_frames=audio_frames,
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
                    image_size=image_size,
                    audio_frames=audio_frames,
                    dtype=dtype,
                )
        else:
            config = AutoConfig.from_pretrained(model_id)
            backbone_onnx_model, _, embedding, extras = load_model_components_from_disk(
                model_id,
                context_length=context_length,
                sequence_length=cache_sl,
            )
            component_models = cls._load_components_from_disk(model_id)
            if not component_models or embedding is None:
                raise ValueError(
                    "Required model components could not be loaded from disk."
                )
            entry = ModelCacheEntry(
                backbone=backbone_onnx_model,
                embedding=embedding,
                config=config,
                extras=extras or None,
                **component_models,
            )

        # Tied embeddings share one lm_head.weight initializer between the
        # embedding Gather and the lm_head MatMul; unshare it here so every
        # path (fresh export, cache hit, disk load) hands downstream pre-sim
        # techniques and ConnectedGraph a graph they accept.
        duplicate_shared_initializers(entry.backbone.graph)
        for _component in entry.present_components():
            duplicate_shared_initializers(entry.component(_component).graph)
        return entry

    @classmethod
    def _load_components_from_disk(cls, checkpoint: str) -> dict[str, onnx.ModelProto]:
        """Collect every declared component's graph from a local checkpoint dir,
        using the same ``{checkpoint}/{component}/model.onnx`` layout the export writes.
        """
        component_models: dict[str, onnx.ModelProto] = {}
        for component in model_components(cls):
            path = os.path.join(checkpoint, component.name, "model.onnx")
            if os.path.exists(path):
                component_models[component.name] = onnx.load(path)
        return component_models

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
        # Only the components this model actually declares get defaults, so a
        # text+audio model never acquires a stray visual precision block.
        declared = model_components(cls)
        for component in declared:
            precision.ensure_component_defaults(component.name)

        backbone_onnx_model = entry.backbone
        embedding = entry.embedding
        config = entry.config
        extras = entry.extras or {}

        block_prec = precision.blocks["default"]
        default_param_qtype = float16 if block_prec.is_float else block_prec.qtype
        default_activation_qtype = precision.activations
        providers = get_ort_providers(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        with (
            AttributePatch(quantsim, "op_types_to_tie_qtzrs", ["Concat"]),
            # Temporary until subgraph quantization is enabled by default in aimet_onnx.
            AttributePatch(quantsim, "_quantize_subgraphs", True),
            AttributePatch(
                quantsim,
                "op_outputs_to_ignore",
                quantsim.op_outputs_to_ignore + ["Slice", "Constant"],
            ),
        ):
            backbone_quantsim = QuantizationSimModel(
                model=backbone_onnx_model,
                quant_scheme="min_max",
                param_type=default_param_qtype,
                activation_type=default_activation_qtype,
                config_file=QUANTSIM_CONFIG,
                providers=providers,
            )
            # One sim per declared modality encoder. A declared component with
            # no graph in the entry (a checkpoint exported before it existed)
            # simply gets no sim rather than failing here.
            component_sims: dict[str, QuantizationSimModel] = {}
            for component in declared:
                component_onnx_model = entry.component(component.name)
                if component_onnx_model is None:
                    continue
                component_sims[component.name] = QuantizationSimModel(
                    model=component_onnx_model,
                    quant_scheme="min_max",
                    param_type=precision.component_weight(component.name).qtype,
                    activation_type=precision.component_activations(component.name),
                    config_file=QUANTSIM_CONFIG,
                    providers=providers,
                )

        kv_cache_type = precision.resolve_kv_cache_qtype()
        block_weight_spec = weight_precision_to_qspec(block_prec)
        lm_head_spec = weight_precision_to_qspec(precision.lm_head)

        configure_llm(
            backbone_quantsim,
            topology,
            kv_cache_type=kv_cache_type,
            backbone_weight_type=block_weight_spec,
            lm_head_weight_type=lm_head_spec,
        )
        _disable_float_quantizers(backbone_quantsim)
        for component_sim in component_sims.values():
            _disable_float_quantizers(component_sim)

        # Note: embedding quantization is deferred to after recipe application
        # (in the test runner) to allow recipes like SpinQuant to rotate weights first.

        return SimCollection(
            backbone=backbone_quantsim,
            **component_sims,
            embedding=embedding,
            config=config,
            position_id_processor=cls.instantiate_position_processor(),
            extras=extras or None,
        )

    @classmethod
    def _export_to_cache_entry(
        cls,
        model_id: str,
        context_length: int,
        sequence_length: int | list[int],
        small_model: bool,
        directory: str,
        image_size: tuple[int, int] | None = None,
        audio_frames: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ModelCacheEntry:
        """Export the torch model to ONNX and return a :class:`ModelCacheEntry`."""
        max_seq_len = (
            max(sequence_length)
            if isinstance(sequence_length, list)
            else sequence_length
        )

        model = cls.instantiate_model(model_id, small_model).to(dtype=dtype)
        text_config = _resolve_text_config(model.config)
        layer_cache_descs = build_layer_cache_descriptors(text_config)

        language_model = cls.get_language_model(model)
        lm_head = cls.get_lm_head(model)

        backbone_kwargs = {
            "cache_type": cls.get_cache_type(),
            "input_names": cls.get_backbone_input_names(
                layer_cache_descs, config=model.config
            ),
        }
        if lm_head is not None:
            backbone_kwargs["lm_head"] = lm_head

        traceable_backbone = ONNXExportableModuleWithCache(
            language_model, **backbone_kwargs
        )

        backbone_onnx_model, backbone_reexported = get_onnx_model(
            checkpoint=directory,
            fp_backbone_model=traceable_backbone,
            context_length=context_length,
            sequence_length=sequence_length,
            sample_input=cls.get_sample_backbone_inputs(
                traceable_backbone,
                context_length,
                max_seq_len,
                layer_cache_descriptors=layer_cache_descs,
                image_size=image_size,
                config=model.config,
            ),
            input_names=cls.get_backbone_input_names(
                layer_cache_descs, config=model.config
            ),
            output_names=cls.get_backbone_output_names(layer_cache_descs),
            dynamo=cls.use_dynamo_export(),
            dynamic_axes=cls.get_backbone_dynamic_axes(
                layer_cache_descs, config=model.config
            ),
        )

        # Every encoder, vision included, exports through the same generic path.
        shape_values = {"image_size": image_size, "audio_frames": audio_frames}
        component_models = {
            component.name: cls._export_component_to_onnx(
                model,
                component.name,
                directory,
                force=backbone_reexported,
                dtype=model.dtype,
                **(
                    {component.shape_kwarg: shape_values[component.shape_kwarg]}
                    if component.shape_kwarg in shape_values
                    else {}
                ),
            )
            for component in model_components(cls)
        }

        embedding = cls.get_embedding(model)
        extras = cls.get_extras(model) or None

        return ModelCacheEntry(
            backbone=backbone_onnx_model,
            embedding=embedding,
            config=model.config,
            extras=extras,
            **component_models,
        )

    @classmethod
    def _export_component_to_onnx(
        cls,
        model,
        component: str,
        directory: str,
        *,
        force: bool = False,
        dtype: torch.dtype = torch.float32,
        **shape_kwargs,
    ) -> onnx.ModelProto:
        """Export one modality encoder to ``{directory}/{component}/model.onnx``.

        A graph already on disk at the expected opset is reused unless ``force``
        (the backbone was re-exported, so the config changed under it), which is
        what makes a warm ``onnx_checkpoints/`` dir cheap.
        """
        component_dir = os.path.join(directory, component)
        path = os.path.join(component_dir, "model.onnx")
        if (
            not force
            and os.path.exists(path)
            and check_opset_equal_to(path, ONNX_OPSET_VERSION)
        ):
            print(f"Loading cached ONNX {component} model...")
            return onnx.load(path)

        os.makedirs(component_dir, exist_ok=True)
        traceable = cls.build_component_wrapper(component, model)
        traceable.eval()
        sample_input = cls.get_sample_component_inputs(
            component, model.config, dtype=dtype, **shape_kwargs
        )
        input_names = cls.get_component_input_names(component)
        output_names = cls.get_component_output_names(component, config=model.config)
        dynamo = cls.use_dynamo_export_for(component)
        component_dynamic_axes = cls.get_component_dynamic_axes(component)
        if dynamo and component_dynamic_axes:
            # _dynamo_export takes no dynamic_axes; fail rather than silently
            # export fixed-shape.
            raise ValueError(
                f"{cls.__name__}'s {component} component exports via dynamo but "
                f"declares dynamic axes {sorted(component_dynamic_axes)}. The "
                "dynamo path cannot express them; either return {} from "
                f"{spec(component).dynamic_axes} or export this component with "
                "torchscript."
            )
        print(
            f"{component.capitalize()} exporting..."
            + (" (dynamo)" if dynamo else " (torchscript)")
        )
        with torch.no_grad():
            if dynamo:
                _dynamo_export(
                    traceable,
                    sample_input,
                    path,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=ONNX_OPSET_VERSION,
                )
            else:
                torch.onnx.export(
                    traceable,
                    sample_input,
                    path,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=ONNX_OPSET_VERSION,
                    dynamo=False,
                    dynamic_axes=component_dynamic_axes or None,
                )
        return consolidate_external_data(path)


# ---------------------------------------------------------------------------
# Model registrations
# ---------------------------------------------------------------------------

from GenAILab.qai_hub_lm.models.qwen2_vl import Qwen_25_VL


@YAMLConfigParser.register_model("qwen2_5_vl")
class Qwen_25_VL_ONNX(VLM_ONNX, Qwen_25_VL):
    pass


try:
    from GenAILab.qai_hub_lm.models.qwen3_vl import Qwen_3_VL

    @YAMLConfigParser.register_model("qwen3_vl")
    class Qwen_3_VL_ONNX(VLM_ONNX, Qwen_3_VL):
        pass

except ImportError:
    warnings.warn(
        "Qwen 3VL is not available. Please upgrade to a later version of transformers to use this model."
    )

try:
    from GenAILab.qai_hub_lm.models.gemma3 import Gemma3_VLM

    @YAMLConfigParser.register_model("gemma3")
    class Gemma3_ONNX(VLM_ONNX, Gemma3_VLM):
        pass

except ImportError:
    pass

try:
    from GenAILab.qai_hub_lm.models.gemma4 import Gemma4_VLM

    @YAMLConfigParser.register_model("gemma4")
    class Gemma4_ONNX(VLM_ONNX, Gemma4_VLM):
        pass

except ImportError:
    pass

try:
    from GenAILab.qai_hub_lm.models.internvl import InternVL_VLM

    @YAMLConfigParser.register_model("internvl_chat")
    class InternVL_ONNX(VLM_ONNX, InternVL_VLM):
        pass

except ImportError:
    pass

try:
    from GenAILab.qai_hub_lm.models.qwen3_asr import Qwen3ASR_LM

    @YAMLConfigParser.register_model("qwen3_asr")
    class Qwen3ASR_ONNX(VLM_ONNX, Qwen3ASR_LM):
        pass

except ImportError:
    warnings.warn(
        "Qwen3-ASR is not available. Please upgrade to transformers >= 5.16 "
        "(which provides transformers.models.qwen3_asr) to use this model."
    )
