# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Runs one text-to-image diffusion experiment document (torch, FP).

The command-line entry point is ``GenAILab/bench/torch/run_diffusion.py``.
"""

from __future__ import annotations

import gc
import inspect
import uuid
from pathlib import Path

import torch

from GenAILab.bench.determinism import set_seed
from GenAILab.bench.diffusion import metrics as diffusion_metrics  # noqa: F401 — triggers registration
from GenAILab.bench.diffusion.config import DiffusionConfig, parse_diffusion_document
from GenAILab.bench.eval_context import EvaluationContext
from GenAILab.bench.fp_cache import DiskBackedFPCache
from GenAILab.bench.profiler import (
    ComponentRecipeStats,
    GPUMeter,
    MetricResult,
    ScoredResult,
    write_stats_to_disk,
)
from GenAILab.qai_hub_lm.diffusion import stable_diffusion  # noqa: F401 — triggers registration
from GenAILab.qai_hub_lm.diffusion.base import COMPONENTS
from GenAILab.qai_hub_lm.diffusion.generator import DiffusionGenerator, GenerationConfig


class DiffusionEvaluationContext(EvaluationContext):
    """An :class:`EvaluationContext` keyed by a diffusion model config.

    The base class hashes the LLM ``ModelConfig`` fields; a diffusion config
    names what changes its images in ``cache_key_dict()`` instead.
    """

    @staticmethod
    def _model_config_to_dict(config) -> dict:
        return config.cache_key_dict()


def run_experiment(
    doc: dict,
    *,
    fp_cache: DiskBackedFPCache,
    results_dir,
):
    """Run one text-to-image experiment document and record its results.

    FP inference only: there are no recipes or exports, so unlike the LLM
    runners this takes no recipe cache or export directory.
    """
    set_seed(42)

    config = parse_diffusion_document(doc)
    print(config)
    model_config = config.model
    eval_ctx = DiffusionEvaluationContext(fp_cache=fp_cache, model_config=model_config)

    gc.collect()
    torch.cuda.empty_cache()

    model_cls = model_config.model_cls
    pipeline = model_cls.instantiate_float_model(
        model_config.model_id, dtype=getattr(torch, model_config.dtype)
    )
    resolution = model_cls.default_resolution(pipeline)
    components = model_cls.build_components(pipeline, scheduler=model_config.scheduler)
    del pipeline  # The components hold every module still needed.

    generator = DiffusionGenerator(
        components,
        GenerationConfig(
            height=model_config.height or resolution,
            width=model_config.width or resolution,
            num_inference_steps=model_config.num_inference_steps,
            guidance_scale=model_config.guidance_scale,
            negative_prompt=model_config.negative_prompt,
            seed=model_config.seed,
        ),
    )

    run_group = config.run_group or uuid.uuid4().hex[:16]
    results_folder = Path(results_dir)
    results_folder.mkdir(parents=True, exist_ok=True)
    image_dir = (
        results_folder / "images" / f"{Path(model_config.model_id).name}_{run_group}"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with generator.on_device(device):
        evaluation_results = run_metrics(
            config,
            generator,
            eval_ctx,
            extra_kwargs={"output_dir": image_dir},
        )

    report_modifiers = model_config.report_modifiers()
    report_modifiers.setdefault("height", generator.config.height)
    report_modifiers.setdefault("width", generator.config.width)
    report_modifiers.setdefault("scheduler", type(components.scheduler).__name__)

    write_stats_to_disk(
        output_folder=str(results_folder),
        filename="profiling_data",
        model_type=model_config.model_type,
        model_id=model_config.model_id,
        model_modifiers=report_modifiers,
        # FP inference: every component runs unquantized, with no recipe steps.
        components={name: ComponentRecipeStats(steps=[]) for name in COMPONENTS},
        accuracy_results=evaluation_results,
        run_group=run_group,
    )
    return evaluation_results


def run_metrics(
    config: DiffusionConfig,
    generator: DiffusionGenerator,
    eval_ctx: EvaluationContext,
    *,
    extra_kwargs: dict | None = None,
) -> list[MetricResult]:
    """Evaluate each metric in order, profiling each one.

    Unlike the LLM runner, each metric's profile is always recorded: for an FP
    diffusion run, generation time and peak memory are the resource numbers, as
    there are no recipe steps to attribute them to.

    ``extra_kwargs`` are passed only to metrics whose ``evaluate`` declares them,
    and never override a value the config set.
    """
    extra_kwargs = extra_kwargs or {}
    results = []
    with torch.no_grad():
        for metric in config.metrics:
            metric_cls = metric.metric_cls
            declared = inspect.signature(metric_cls.evaluate).parameters
            kwargs = {k: v for k, v in extra_kwargs.items() if k in declared}
            kwargs.update(metric.metric_kwargs)
            with GPUMeter(
                capture_intermediate_data=config.profiler.capture_intermediate_data,
                **config.profiler.gpu_meter_kwargs,
            ) as profiler:
                result = metric_cls.evaluate(generator, eval_ctx=eval_ctx, **kwargs)
            details = None
            if isinstance(result, ScoredResult):
                result, details = result.result, result.details
            print(f"{metric_cls.__name__} result: {result}")
            results.append(
                MetricResult(
                    metric_name=metric_cls.__name__,
                    result=result,
                    profiler=profiler,
                    scoring_version=metric_cls.SCORING_VERSION,
                    details=details,
                )
            )
    return results
