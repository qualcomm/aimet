# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI test runner"""

import contextlib
import uuid
import pytest
import torch
import gc
import os
import yaml
from pathlib import Path

from aimet_torch.v2.nn import QuantizationMixin, compute_param_encodings
from aimet_torch.v2.utils import remove_all_quantizers

from GenAILab.qai_hub_lm.models.base import LLM, VLM
from GenAILab.qai_hub_lm.models.utils.layer_cache import build_layer_cache_descriptors
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.bench.profiler import (
    ComponentRecipeStats,
    RecipeStepStats,
    write_stats_to_disk,
)
from GenAILab.bench.determinism import set_seed
from GenAILab.bench.eval_context import EvaluationContext
from GenAILab.bench.fp_cache import DiskBackedFPCache
from GenAILab.bench.metrics import run_metrics
from GenAILab.bench.recipe_chain import (
    apply_pre_quantization_chain,
    apply_quantization_chain,
)
from GenAILab.bench import datasets, metrics  # noqa: F401 — triggers registration
from GenAILab.qai_hub_lm.backends import torch as models  # noqa: F401 — triggers registration
from GenAILab.bench.torch import quant_recipes  # noqa: F401 — triggers registration
from GenAILab.qai_hub_lm.backends.torch.generator_utils import generator_factory


def test_llm_quantization(
    test_config,
    fp_cache: DiskBackedFPCache,
    recipe_cache,
    export_dir,
    results_dir,
):
    if test_config is None:
        pytest.skip("No GenAI test parameters provided.")
    if "analysis" in test_config:
        raise ValueError(
            "The 'analysis' config section is only supported with the ONNX framework."
        )
    set_seed(42)

    config = YAMLConfigParser.parse_document(test_config, export_base_dir=export_dir)
    print(config)

    eval_ctx = EvaluationContext(fp_cache=fp_cache, model_config=config.model)

    model_cls = config.model.model_cls
    context_length = config.model.context_length
    sequence_length = config.model.sequence_length
    model_id = config.model.model_id
    model_type = config.model.model_type
    image_size = config.model.image_size
    audio_frames = config.model.audio_frames
    precomputed_encodings = config.model.encodings

    # Build model_kwargs for instantiate_float_model
    model_kwargs = config.model.extra_kwargs.copy()
    if config.model.dtype:
        model_kwargs["dtype"] = getattr(torch, config.model.dtype)

    precision = config.precision

    gc.collect()
    torch.cuda.empty_cache()

    model = model_cls.instantiate_float_model(
        model_id,
        **model_kwargs,
    )

    # Apply the pre-sim chain; the torch float model is the nn.Module.
    # Each pre-sim technique rotates the whole model once.
    # ``pre_sim_profilers`` maps technique name -> profiler for
    # re-attachment to the recorded recipe below.
    pre_sim_profilers = apply_pre_quantization_chain(
        config.recipe.pre_sim,
        model,
        profiler_kwargs=config.profiler.gpu_meter_kwargs,
        profiler_capture_intermediate_data=config.profiler.capture_intermediate_data,
    )

    # Pass model_id so a QAT-aware instantiate_quantsim (e.g. Gemma4_Torch) can
    # locate the packed checkpoint's scales. Other backends absorb it via **kwargs.
    sim_collection = model_cls.instantiate_quantsim(
        model,
        context_length,
        sequence_length,
        precision=precision,
        image_size=image_size,
        audio_frames=audio_frames,
        model_id=model_id,
        **model_kwargs,
    )
    tokenizer = model_cls.instantiate_tokenizer(model_id)
    generator = generator_factory(
        sim_collection,
        model_cls.get_generator_cls(),
        tokenizer,
        sequence_length,
        context_length,
        visual_output_names=model_cls.get_visual_output_names()
        if sim_collection.has("visual")
        else None,
        image_size=image_size,
        audio_frames=audio_frames,
        **model_kwargs,
    )

    if precomputed_encodings is not None:
        print(f"Loading precomputed encodings from {precomputed_encodings}.")
        sim_collection.backbone.load_encodings(
            precomputed_encodings,
            partial=True,
            strict=False,
            allow_overwrite=False,
        )
        for _component in sim_collection.present_components():
            sim_collection.component(_component).load_encodings(
                precomputed_encodings,
                partial=True,
                strict=False,
                allow_overwrite=False,
            )
        if sim_collection.embedding is not None:
            pass
            # todo: need to update this to intelligently load encodings for embedding table if it exists

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with generator.on_device(device):
        # Disable every modality encoder's quantizers during backbone recipes so
        # the encoders run in FP mode (their quantizers aren't calibrated yet).
        encoder_ctx = contextlib.ExitStack()
        for _component in sim_collection.present_components():
            encoder_ctx.enter_context(
                remove_all_quantizers(sim_collection.component(_component).model)
            )
        with encoder_ctx:
            backbone_steps = apply_quantization_chain(
                config.recipe.backbone,
                sim_collection.backbone,
                generator,
                tokenizer,
                context_length,
                image_size,
                config.profiler.gpu_meter_kwargs,
                config.profiler.capture_intermediate_data,
                framework="torch",
                model_id=model_id,
                precision=precision,
                model_kwargs=model_kwargs,
                component="backbone",
                recipe_cache=recipe_cache,
                adaptations=config.model.adaptations,
                pre_sim=config.recipe.pre_sim,
            )

        # One chain per modality encoder. Each runs with the backbone's
        # quantizers disabled and the generator rewired to yield that
        # component's encoder inputs from prefill().
        component_steps: dict[str, list] = {}
        for _component in sim_collection.present_components():
            _chain = config.recipe.component(_component)
            if _chain is None:
                continue
            backbone_ctx = remove_all_quantizers(sim_collection.backbone.model)
            with backbone_ctx, generator.component_quantization_mode(_component):
                component_steps[_component] = apply_quantization_chain(
                    _chain,
                    sim_collection.component(_component),
                    generator,
                    tokenizer,
                    context_length,
                    image_size,
                    config.profiler.gpu_meter_kwargs,
                    config.profiler.capture_intermediate_data,
                    framework="torch",
                    model_id=model_id,
                    precision=precision,
                    model_kwargs=model_kwargs,
                    component=_component,
                    recipe_cache=recipe_cache,
                    adaptations=config.model.adaptations,
                    pre_sim=config.recipe.pre_sim,
                )

        # Finalize embedding quantization after recipes have had a chance to
        # transform the weights. The embedding is already rotated when SpinQuant
        # ran on the float model before sim creation.
        # Skip if RemoveQuantization was applied to the backbone — the
        # embedding should stay in FP to match.
        backbone_removed_quant = any(
            s.recipe_name == "RemoveQuantization" for s in backbone_steps
        )
        if (
            sim_collection.embedding is not None
            and isinstance(sim_collection.embedding, QuantizationMixin)
            and not backbone_removed_quant
        ):
            compute_param_encodings(sim_collection.embedding)

        gc.collect()
        torch.cuda.empty_cache()

    # Every row needs an identity, not just those with an ONNX re-eval: a shared
    # empty string would merge unrelated runs. The exported config carries this
    # value so a following ONNX row joins on it.
    run_group = uuid.uuid4().hex[:16]
    export_dir = config.export
    # TODO: remove skip exports for models that require Dynamo export
    if export_dir and not model_cls.use_dynamo_export():
        tokenizer.save_pretrained(export_dir)
        sim_collection.config.save_pretrained(export_dir)

        os.mkdir(os.path.join(export_dir, "backbone"))
        use_dynamic = isinstance(sequence_length, list) and len(sequence_length) > 1
        max_sl = (
            max(sequence_length)
            if isinstance(sequence_length, list)
            else sequence_length
        )
        sl_tag = "dynamic" if use_dynamic else str(max_sl)
        layer_cache_descs = build_layer_cache_descriptors(
            sim_collection.backbone.model.model.config
        )
        sim_collection.backbone.onnx.export(
            f=os.path.join(
                export_dir, "backbone", f"model_sl{sl_tag}_cl{context_length}.onnx"
            ),
            args=model_cls.get_sample_backbone_inputs(
                model=sim_collection.backbone.model,
                context_length=context_length,
                sequence_length=max_sl,
                layer_cache_descriptors=generator.layer_cache_descriptors,
                image_size=image_size,
                config=sim_collection.config,
            ),
            input_names=model_cls.get_backbone_input_names(layer_cache_descs),
            output_names=model_cls.get_backbone_output_names(layer_cache_descs),
            opset_version=17,
            dynamo=model_cls.use_dynamo_export(),
            dynamic_axes=model_cls.get_backbone_dynamic_axes(layer_cache_descs)
            if use_dynamic
            else None,
            export_int32_bias=False,
        )

        # One export dir per modality encoder, named after the component.
        for _component in sim_collection.present_components():
            assert issubclass(model_cls, VLM)
            _shape_kwargs = (
                {"image_size": image_size}
                if _component == "visual"
                else {"audio_frames": audio_frames}
                if audio_frames is not None
                else {}
            )
            os.mkdir(os.path.join(export_dir, _component))
            sim_collection.component(_component).onnx.export(
                f=os.path.join(export_dir, _component, "model.onnx"),
                args=model_cls.get_sample_component_inputs(
                    _component, sim_collection.config, **_shape_kwargs
                ),
                input_names=model_cls.get_component_input_names(_component),
                output_names=model_cls.get_component_output_names(_component),
                opset_version=17,
                dynamo=model_cls.use_dynamo_export_for(_component),
                export_int32_bias=False,
            )

        if sim_collection.embedding is not None:
            if isinstance(sim_collection.embedding, QuantizationMixin):
                sim_collection.embedding.fold_param_quantizers()

            torch.save(
                sim_collection.embedding.state_dict()["weight"].as_subclass(
                    torch.Tensor
                ),
                os.path.join(export_dir, "embedding.pth"),
            )

        # Save any extra auxiliary modules (e.g. Gemma4's per-layer embedding
        # embed_tokens_per_layer) under extras/<name>.pth so the ONNX phase can
        # restore them into the generator's extras.
        if sim_collection.extras:
            extras_dir = os.path.join(export_dir, "extras")
            os.makedirs(extras_dir, exist_ok=True)
            for extra_name, extra_mod in sim_collection.extras.items():
                if extra_mod is None:
                    continue
                if isinstance(extra_mod, QuantizationMixin):
                    extra_mod.fold_param_quantizers()
                weight = getattr(extra_mod, "weight", None)
                if weight is None and hasattr(extra_mod, "state_dict"):
                    weight = extra_mod.state_dict().get("weight")
                if weight is not None:
                    torch.save(
                        weight.as_subclass(torch.Tensor),
                        os.path.join(extras_dir, f"{extra_name}.pth"),
                    )

        if config.eval_in_onnx:
            # Use the last Calibration step's dataset for ONNX re-calibration
            def _last_calibration_for_onnx(steps):
                for s in reversed(steps):
                    if s.recipe_name == quant_recipes.Calibration.__name__:
                        return {
                            "name": quant_recipes.Calibration.__name__,
                            "dataset": {"name": s.dataset_name, **s.dataset_kwargs},
                            **s.recipe_kwargs,
                        }
                return {"name": quant_recipes.RemoveQuantization.__name__}

            onnx_recipe = {
                "backbone": _last_calibration_for_onnx(backbone_steps),
            }
            for _component, _steps in component_steps.items():
                if _steps:
                    onnx_recipe[_component] = _last_calibration_for_onnx(_steps)

            data = {
                "model": {
                    "model_id": export_dir,
                    "encodings": export_dir,
                    "sequence_length": sequence_length,
                    "context_length": context_length,
                    **({"image_size": list(image_size)} if image_size else {}),
                    **({"audio_frames": audio_frames} if audio_frames else {}),
                    **model_kwargs,
                },
                "precision": precision.to_dict(),
                "run_group": run_group,
                "recipe": onnx_recipe,
                "metrics": [
                    {
                        "name": metric.name,
                        **metric.metric_kwargs,
                    }
                    for metric in config.metrics
                ],
            }

            with open(os.path.join(export_dir, "onnx_eval_config.yaml"), "w") as file:
                yaml.dump(data, file, default_flow_style=False)

    with generator.on_device(device):
        evaluation_results = run_metrics(
            config.metrics,
            generator,
            tokenizer,
            context_length,
            eval_ctx,
            image_size=image_size,
            audio_frames=audio_frames,
            gpu_meter_kwargs=config.profiler.gpu_meter_kwargs,
            capture_intermediate_data=config.profiler.capture_intermediate_data,
        )

    # Snapshot of the authored model section for the report, derived from the
    # parsed config (not the instantiation kwargs) so fields like ``adaptations``
    # are always recorded.
    report_modifiers = config.model.report_modifiers()

    # Re-attach pre-sim steps (e.g. SpinQuant) as synthetic leading steps so the
    # recorded recipe reflects the pre-sim rotations. A single pre-sim pass
    # rotates the whole model, so the same markers are prepended to both
    # backbone and visual component recipes.
    pre_markers = [
        RecipeStepStats(
            recipe_name=step.name,
            recipe_kwargs=step.recipe_kwargs,
            dataset_name="",
            dataset_kwargs={},
            profiler=pre_sim_profilers.get(step.name),
        )
        for step in config.recipe.pre_sim
    ]
    backbone_steps = [*pre_markers, *backbone_steps]
    component_steps = {
        name: [*pre_markers, *steps] for name, steps in component_steps.items() if steps
    }

    components = {
        "backbone": ComponentRecipeStats(steps=backbone_steps),
    }
    for _component, _steps in component_steps.items():
        components[_component] = ComponentRecipeStats(steps=_steps)

    results_folder = Path(results_dir)
    results_folder.mkdir(parents=True, exist_ok=True)
    precision_dict = precision.to_dict()
    write_stats_to_disk(
        output_folder=str(results_folder),
        filename="profiling_data",
        model_type=model_type,
        model_id=model_id,
        model_modifiers=report_modifiers,
        components=components,
        accuracy_results=evaluation_results,
        export_location=export_dir,
        precision=precision_dict,
        run_group=run_group,
    )

    if export_dir:
        write_stats_to_disk(
            output_folder=export_dir,
            filename="profiling_data",
            model_type=model_type,
            model_id=model_id,
            model_modifiers=report_modifiers,
            components=components,
            accuracy_results=evaluation_results,
            precision=precision_dict,
            run_group=run_group,
        )
