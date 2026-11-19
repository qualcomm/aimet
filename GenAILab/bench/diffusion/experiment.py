# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Runs one text-to-image diffusion experiment document.

FP inference runs with torch or ONNX Runtime. A document with a ``precision``
section quantizes those components with an aimet_onnx QuantSim (ONNX only) and
calibrates them before the metrics run; the other components stay in float.

The command-line entry points are ``GenAILab/bench/torch/run_diffusion.py`` and
``GenAILab/bench/onnx/run_diffusion.py``.
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
    RecipeStepStats,
    ScoredResult,
    write_stats_to_disk,
)
from GenAILab.qai_hub_lm.diffusion import stable_diffusion  # noqa: F401 — triggers registration
from GenAILab.bench.diffusion.onnx_cache import DiffusionONNXCache, export_key
from GenAILab.qai_hub_lm.diffusion import onnx as diffusion_onnx
from GenAILab.qai_hub_lm.diffusion.base import COMPONENTS, DiffusionComponents
from GenAILab.qai_hub_lm.diffusion.generator import DiffusionGenerator, GenerationConfig

# AIMET is needed only to quantize, so the FP paths run without it.
try:
    from GenAILab.bench.diffusion import quantize as diffusion_quantize
except ImportError as _exc:  # pragma: no cover - depends on the environment
    _QUANTIZE_IMPORT_ERROR = _exc
    diffusion_quantize = None
else:
    _QUANTIZE_IMPORT_ERROR = None


FRAMEWORKS = ("torch", "onnx")


class DiffusionEvaluationContext(EvaluationContext):
    """An :class:`EvaluationContext` keyed by a diffusion model config.

    The base class hashes the LLM ``ModelConfig`` fields; a diffusion config
    names what changes its images in ``cache_key_dict()`` instead. The framework
    is part of the key, so torch and ONNX results are never mixed.
    """

    def __init__(self, fp_cache, model_config, framework: str):
        self._framework = framework
        super().__init__(fp_cache=fp_cache, model_config=model_config)

    def _model_config_to_dict(self, config) -> dict:
        return {**config.cache_key_dict(), "framework": self._framework}


def run_experiment(
    doc: dict,
    *,
    fp_cache: DiskBackedFPCache,
    results_dir,
    framework: str = "torch",
    onnx_cache: DiffusionONNXCache | None = None,
):
    """Run one text-to-image experiment document and record its results.

    With ``framework="onnx"`` the components are exported (or taken from
    ``onnx_cache``) and run as ONNX Runtime sessions behind the same generator.
    Components named in the document's ``precision`` section are then replaced
    by QuantSims, each calibrated with the others in float (see
    :mod:`~GenAILab.bench.diffusion.quantize`).
    """
    if framework not in FRAMEWORKS:
        raise ValueError(f"Unknown framework {framework!r}; expected {FRAMEWORKS}.")
    if framework == "onnx" and onnx_cache is None:
        raise ValueError("The ONNX framework needs an onnx_cache.")
    set_seed(42)

    config = parse_diffusion_document(doc)
    print(config)
    model_config = config.model
    dtype = getattr(torch, model_config.dtype)
    # Fail before loading a multi-gigabyte pipeline.
    if config.is_quantized:
        if diffusion_quantize is None:
            raise ImportError(
                "Quantized diffusion experiments need aimet_onnx."
            ) from _QUANTIZE_IMPORT_ERROR
        if framework != "onnx":
            raise ValueError(
                "Quantized diffusion experiments run with aimet_onnx only; use "
                "GenAILab.bench.onnx.run_diffusion."
            )
        diffusion_quantize.check_dtype(dtype)
    if framework == "onnx":
        diffusion_onnx.check_dtype(dtype)
    eval_ctx = DiffusionEvaluationContext(fp_cache, model_config, framework)

    gc.collect()
    torch.cuda.empty_cache()

    model_cls = model_config.model_cls
    pipeline = model_cls.instantiate_float_model(model_config.model_id, dtype=dtype)
    resolution = model_cls.default_resolution(pipeline)
    components = model_cls.build_components(pipeline, scheduler=model_config.scheduler)
    del pipeline  # The components hold every module still needed.

    generation_config = GenerationConfig(
        height=model_config.height or resolution,
        width=model_config.width or resolution,
        num_inference_steps=model_config.num_inference_steps,
        guidance_scale=model_config.guidance_scale,
        negative_prompt=model_config.negative_prompt,
        seed=model_config.seed,
    )
    # Validates the resolution before an ONNX export bakes it in.
    generator = DiffusionGenerator(components, generation_config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    component_steps = {name: [] for name in COMPONENTS}
    if framework == "onnx":
        components, paths = _onnx_components(
            model_cls,
            components,
            model_config,
            generation_config,
            onnx_cache,
            device,
            revision=_checkpoint_revision(model_config.model_id),
        )
        generator = DiffusionGenerator(components, generation_config)
        if config.is_quantized:
            try:
                component_steps.update(_quantize(config, generator, paths, device))
            except BaseException:
                _release_sessions(generator.components)
                raise

    run_group = config.run_group or uuid.uuid4().hex[:16]
    results_folder = Path(results_dir)
    results_folder.mkdir(parents=True, exist_ok=True)
    image_dir = (
        results_folder / "images" / f"{Path(model_config.model_id).name}_{run_group}"
    )

    try:
        with generator.on_device(device):
            evaluation_results = run_metrics(
                config,
                generator,
                eval_ctx,
                extra_kwargs={"output_dir": image_dir},
            )
    finally:
        _release_sessions(generator.components)

    report_modifiers = model_config.report_modifiers()
    report_modifiers.setdefault("height", generator.config.height)
    report_modifiers.setdefault("width", generator.config.width)
    report_modifiers.setdefault("scheduler", type(components.scheduler).__name__)
    report_modifiers["framework"] = framework
    if framework == "onnx":
        report_modifiers["onnx_opset"] = diffusion_onnx.ONNX_OPSET

    write_stats_to_disk(
        output_folder=str(results_folder),
        filename="profiling_data",
        model_type=model_config.model_type,
        model_id=model_config.model_id,
        model_modifiers=report_modifiers,
        # A float component has no recipe steps.
        components={
            name: ComponentRecipeStats(steps=steps)
            for name, steps in component_steps.items()
        },
        accuracy_results=evaluation_results,
        precision=config.precision_dict() or None,
        run_group=run_group,
    )
    return evaluation_results


def _quantize(
    config: DiffusionConfig,
    generator: DiffusionGenerator,
    paths: dict[str, Path],
    device: torch.device,
) -> dict[str, list[RecipeStepStats]]:
    """Quantize and calibrate every component in ``config.precision``.

    Each is calibrated on prompts no metric scores, with the other components
    in float. Returns each quantized component's recipe steps, for the results.
    """
    calibration = config.calibration
    evaluated = {
        p["prompt"]
        for metric in config.metrics
        for p in metric.metric_cls.evaluation_prompts(**metric.metric_kwargs)
    }
    prompts = diffusion_metrics.held_out_parti_prompts(
        calibration.num_prompts, calibration.sample_seed, exclude=evaluated
    )
    profiles = diffusion_quantize.quantize_and_calibrate(
        config,
        generator,
        paths,
        prompts,
        device,
        meter=lambda: GPUMeter(
            capture_intermediate_data=config.profiler.capture_intermediate_data,
            **config.profiler.gpu_meter_kwargs,
        ),
    )
    return {
        name: [
            RecipeStepStats(
                recipe_name="Calibration",
                recipe_kwargs=calibration.to_dict(),
                dataset_name="PartiPrompts",
                dataset_kwargs={"held_out": True},
                profiler=profile,
            )
        ]
        for name, profile in profiles.items()
    }


def _release_sessions(components: DiffusionComponents) -> None:
    """Return the GPU memory of every ONNX session and QuantSim in ``components``.

    The shared ONNX Runtime arena outlives its sessions, so without this a
    failed experiment would leave the next one in the same process short of
    GPU memory.
    """
    for _, component in components.items():
        if hasattr(component, "release"):
            component.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _onnx_components(
    model_cls,
    components: DiffusionComponents,
    model_config,
    generation_config: GenerationConfig,
    onnx_cache: DiffusionONNXCache,
    device: torch.device,
    *,
    revision: str | None,
) -> DiffusionComponents:
    """``components`` with each torch module replaced by its ONNX Runtime session.

    Also returns the path of each component's exported graph.
    """
    key = export_key(
        model_config.model_id,
        revision,
        model_config.dtype,
        generation_config.height,
        generation_config.width,
    )

    def export(directory):
        # Exported on the device the sessions will run on: fp16 graphs traced
        # on the CPU would need fp16 CPU kernels torch does not always have.
        with components.on_device(device):
            diffusion_onnx.export_components(
                model_cls,
                components,
                generation_config.height,
                generation_config.width,
                directory,
            )

    paths = onnx_cache.get_or_export(key, export)
    onnx_components = diffusion_onnx.load_components(components, paths)
    # The torch modules are no longer needed; drop them before ORT allocates.
    for name in COMPONENTS:
        setattr(components, name, None)
    gc.collect()
    torch.cuda.empty_cache()
    return onnx_components, paths


def _checkpoint_revision(model_id: str) -> str | None:
    """The commit hash of a Hub checkpoint, so an upstream update re-exports.

    A local directory has no revision; its path is used instead.
    """
    if Path(model_id).is_dir():
        return str(Path(model_id).resolve())
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception:  # pylint: disable=broad-exception-caught
        # Offline: the cache is still keyed by everything else.
        return None


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
