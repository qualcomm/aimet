# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared recipe chain application logic for Torch and ONNX test runners."""

from __future__ import annotations

import contextlib
import gc
from typing import TYPE_CHECKING

import torch

from transformers.processing_utils import ProcessorMixin

from GenAILab.bench.datasets import GeneratedDataset, Interleaved, TextDataset
from GenAILab.bench.profiler import GPUMeter, RecipeStepStats
from GenAILab.qai_hub_lm.transforms.exportable_moe import forced_expert_activation

if TYPE_CHECKING:
    from GenAILab.bench.yaml_config_parser import ResolvedStep


def apply_quantization_chain(
    recipe_list: list[ResolvedStep] | tuple[ResolvedStep, ...],
    sim_component,
    generator,
    tokenizer,
    context_length,
    image_size,
    profiler_kwargs,
    profiler_capture_intermediate_data,
    framework,
    model_id,
    precision,
    model_kwargs,
    component="backbone",
    recipe_cache=None,
    pre_sim=None,
    topology=None,
    adaptations=None,
):
    """Apply a chain of on-sim recipe steps, with automatic cache lookup and save.

    If recipe_cache is provided, performs a cache lookup first and skips
    already-cached steps. After each cacheable step, saves to the cache.
    Returns the full list of RecipeStepStats (cached + newly computed).

    ``pre_sim`` is the flat pre-sim step list (e.g. the SpinQuant step that
    rotated the float model before the sim was built). It is folded into the
    cache base hash so the chain distinguishes rotated from non-rotated graphs;
    pre-sim steps are applied by ``apply_pre_quantization_chain``, not here.

    ``topology`` is the decoder-stack analysis of the float model (ONNX only,
    ``None`` for torch). It is passed to every step; techniques that do not need
    it absorb it via ``**kwargs``.
    """
    skip_to = 0
    cached_step_stats = []
    chain_hashes = None

    if recipe_cache:
        skip_to, cached_step_stats, chain_hashes = recipe_cache.lookup(
            recipe_list,
            sim_component,
            model_id,
            precision,
            model_kwargs,
            framework,
            component,
            pre_sim,
            # Not in model_kwargs: adaptations live on the model class.
            adaptations,
        )

    step_stats = list(cached_step_stats)

    for i, step in enumerate(recipe_list[skip_to:], start=skip_to):
        recipe_cls = step.technique_cls
        recipe_kwargs = step.recipe_kwargs
        dataset_cls = step.dataset_cls
        dataset_kwargs = step.dataset_kwargs

        if dataset_cls is not None:
            # Text datasets need just the tokenizer; multimodal datasets
            # need the full processor. Interleaved is text-or-multimodal
            # depending on its sub-datasets, so it only requires a processor
            # when the model actually provides one.
            if not issubclass(dataset_cls, TextDataset):
                if image_size is not None:
                    dataset_kwargs.setdefault("image_size", image_size)
                if not issubclass(dataset_cls, Interleaved):
                    assert isinstance(tokenizer, ProcessorMixin), (
                        f"Multimodal dataset {dataset_cls.__name__} requires a "
                        f"ProcessorMixin, got {type(tokenizer).__name__}"
                    )
            # Interleaved handles per-sub tokenizer/processor selection
            # internally, so hand it the full tokenizer/processor object.
            dataset_tokenizer = (
                getattr(tokenizer, "tokenizer", tokenizer)
                if issubclass(dataset_cls, TextDataset)
                else tokenizer
            )
            # Move model to CPU if the dataset is generated, to avoid GPU memory issues during generation.
            if issubclass(dataset_cls, GeneratedDataset):
                dataset_kwargs.setdefault("model_id", model_id)
                gen_ctx = generator.on_device(torch.device("cpu"))
            else:
                gen_ctx = contextlib.nullcontext()
            with gen_ctx:
                train_dataset = dataset_cls.load_encoded_dataset(
                    dataset_tokenizer,
                    context_length,
                    **dataset_kwargs,
                )
        else:
            train_dataset = None

        with GPUMeter(
            **profiler_kwargs,
            capture_intermediate_data=profiler_capture_intermediate_data,
        ) as profiler:
            # MoE experts switch from their stock sparse path to an exportable
            # one for the duration. Must wrap the whole apply(): SeqMSE and
            # AdaScale hand off to aimet, and it is those internal per-layer
            # forwards that decide what each expert observes. No-op otherwise.
            with forced_expert_activation(sim_component):
                recipe_cls.apply(
                    sim_component,
                    generator,
                    train_dataset,
                    component=component,
                    topology=topology,
                    **recipe_kwargs,
                )
        step_stats.append(
            RecipeStepStats(
                recipe_name=recipe_cls.__name__,
                recipe_kwargs=recipe_kwargs,
                dataset_name=dataset_cls.__name__ if dataset_cls else "",
                dataset_kwargs=dataset_kwargs,
                profiler=profiler,
            )
        )

        # Save to cache after cacheable steps
        if recipe_cache and chain_hashes and recipe_cls.cacheable():
            recipe_cache.save(
                chain_hash=chain_hashes[i + 1],
                parent_hash=chain_hashes[i],
                step_stats=step_stats,
                sim_component=sim_component,
                framework=framework,
                component=component,
                model_id=model_id,
            )

        gc.collect()
        torch.cuda.empty_cache()
    return step_stats


def apply_pre_quantization_chain(
    pre_sim: tuple[ResolvedStep, ...],
    float_model,
    profiler_kwargs=None,
    profiler_capture_intermediate_data=False,
    topology=None,
):
    """Apply the pre-sim technique chain on the float model, before the sim is built.

    ``pre_sim`` is the FLAT pre-sim step tuple (e.g. SpinQuant). Each pre-sim
    technique acts on the whole float model, so it is applied exactly ONCE here
    (no component axis). ``float_model`` is the backend float bundle (nn.Module
    for torch; the entry with backbone/visual/embedding for onnx).

    ``topology`` is the decoder-stack analysis of ``float_model`` (ONNX only,
    ``None`` for torch), threaded the same way as in
    :func:`apply_quantization_chain`. It is passed to every step; techniques that
    do not need it absorb it via ``**kwargs``.

    Returns ``{technique_name: profiler_or_None}`` so the runner can re-attach
    pre-sim work to the recorded recipe for reporting.
    """
    profilers = {}
    for step in pre_sim:
        name = step.name
        technique_cls = step.technique_cls
        kwargs = step.recipe_kwargs
        if profiler_kwargs is not None:
            with GPUMeter(
                **profiler_kwargs,
                capture_intermediate_data=profiler_capture_intermediate_data,
            ) as profiler:
                technique_cls.apply(float_model, topology=topology, **kwargs)
            profilers[name] = profiler
        else:
            technique_cls.apply(float_model, topology=topology, **kwargs)
            profilers[name] = None
    return profilers
