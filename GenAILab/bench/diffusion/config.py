# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Config parsing for text-to-image diffusion experiments.

A diffusion document has a ``model`` section, ``metrics``, and optionally
``profiler`` and ``run_group``. Quantization (``precision``/``recipe``), export
and analysis are not supported for diffusion yet and are rejected by name, so a
config written for a later phase fails loudly instead of running FP silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from GenAILab.bench.yaml_config_parser import (
    ProfilerConfig,
    ResolvedMetric,
    YAMLConfigParser,
)
from GenAILab.qai_hub_lm.diffusion.base import (
    DiffusionModel,
    get_diffusion_model_class,
)

_UNSUPPORTED_SECTIONS = (
    "precision",
    "recipe",
    "dataset",
    "export",
    "eval_in_onnx",
    "analysis",
)
_DTYPES = ("float32", "float16", "bfloat16")

#: Sampling defaults. 10 steps keeps a regression run short; resolution
#: defaults to what the checkpoint was trained at.
DEFAULT_NUM_INFERENCE_STEPS = 10
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = 0


@dataclass(frozen=True)
class DiffusionModelConfig:
    """Parsed ``model`` section of a diffusion document."""

    model_cls: type[DiffusionModel]
    model_id: str
    #: The diffusers pipeline ``_class_name``; reported as the model type.
    model_type: str
    dtype: str = "float32"
    #: ``None`` means the checkpoint's native resolution.
    height: int | None = None
    width: int | None = None
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    negative_prompt: str = ""
    #: ``None`` means the checkpoint's default scheduler.
    scheduler: str | None = None
    #: Base seed. Prompt ``i`` of a metric is generated with seed ``seed + i``.
    seed: int = DEFAULT_SEED

    def report_modifiers(self) -> dict[str, Any]:
        """Model-section fields to record under the report's ``model_modifiers``."""
        modifiers = {
            "dtype": self.dtype,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
        }
        for key in ("height", "width", "scheduler"):
            if getattr(self, key) is not None:
                modifiers[key] = getattr(self, key)
        if self.negative_prompt:
            modifiers["negative_prompt"] = self.negative_prompt
        return modifiers

    def cache_key_dict(self) -> dict[str, Any]:
        """Everything that changes the FP images; the evaluation-cache key."""
        return {
            "class": self.model_cls,
            "model_id": self.model_id,
            "model_type": self.model_type,
            **self.report_modifiers(),
        }


@dataclass(frozen=True)
class DiffusionConfig:
    """The fully parsed diffusion document."""

    model: DiffusionModelConfig
    metrics: tuple[ResolvedMetric, ...]
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    run_group: str | None = None


def parse_diffusion_document(doc: dict) -> DiffusionConfig:
    """Validate and parse one diffusion config document. Consumes ``doc``."""
    for section in _UNSUPPORTED_SECTIONS:
        if section in doc:
            raise ValueError(
                f"'{section}' is not supported for text-to-image diffusion "
                f"experiments yet; only FP inference is."
            )

    model = _parse_model(doc.pop("model", None))

    metrics_raw = doc.pop("metrics", None)
    if not metrics_raw:
        raise RuntimeError("Metrics not specified.")
    if not isinstance(metrics_raw, list):
        metrics_raw = [metrics_raw]
    metrics = []
    for metric in metrics_raw:
        if not isinstance(metric, dict) or "name" not in metric:
            raise RuntimeError("Metric name not specified.")
        resolved = YAMLConfigParser._resolve_metric(metric)
        if not getattr(resolved.metric_cls, "IMAGE_GENERATION", False):
            raise ValueError(
                f"Metric '{resolved.name}' does not evaluate image generation, so "
                f"it cannot be used with a text-to-image model."
            )
        metrics.append(resolved)

    profiler_dict = dict(doc.pop("profiler", None) or {})
    profiler = ProfilerConfig(
        capture_intermediate_data=profiler_dict.pop("capture_intermediate_data", False),
        gpu_meter_kwargs=profiler_dict,
    )
    run_group = doc.pop("run_group", None)

    if doc:
        raise ValueError(f"Unrecognized sections in config: {list(doc)}")
    return DiffusionConfig(
        model=model, metrics=tuple(metrics), profiler=profiler, run_group=run_group
    )


def _parse_model(model: dict | None) -> DiffusionModelConfig:
    if not isinstance(model, dict):
        raise RuntimeError("Model section not specified.")
    model = dict(model)
    model_id = model.pop("model_id", None)
    if not model_id:
        raise RuntimeError("Model 'model_id' not specified.")
    for llm_key in ("sequence_length", "context_length", "adaptations", "encodings"):
        if llm_key in model:
            raise ValueError(
                f"'model.{llm_key}' does not apply to text-to-image diffusion models."
            )

    model_type = _pipeline_class_name(model_id)
    try:
        model_cls = get_diffusion_model_class(model_type)
    except LookupError as exc:
        raise LookupError(
            f"Failed to configure model for model_id='{model_id}'."
        ) from exc

    dtype = model.pop("dtype", "float32")
    if dtype not in _DTYPES:
        raise ValueError(
            f"Unsupported model dtype {dtype!r}; expected one of {_DTYPES}."
        )

    kwargs = {}
    for key in ("height", "width", "num_inference_steps", "seed"):
        if key in model:
            kwargs[key] = _int_field(model.pop(key), key)
    for key in ("height", "width", "num_inference_steps"):
        if key in kwargs and kwargs[key] <= 0:
            raise ValueError(f"'model.{key}' must be positive, got {kwargs[key]}.")
    if "guidance_scale" in model:
        value = model.pop("guidance_scale")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'model.guidance_scale' must be a number, got {value!r}.")
        kwargs["guidance_scale"] = float(value)
    for key in ("negative_prompt", "scheduler"):
        if key in model:
            value = model.pop(key)
            if not isinstance(value, str):
                raise ValueError(f"'model.{key}' must be a string, got {value!r}.")
            kwargs[key] = value

    if model:
        raise ValueError(
            f"Unrecognized keys in diffusion 'model' section: {sorted(model)}"
        )
    return DiffusionModelConfig(
        model_cls=model_cls,
        model_id=model_id,
        model_type=model_type,
        dtype=dtype,
        **kwargs,
    )


def _int_field(value, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'model.{key}' must be an integer, got {value!r}.")
    return value


def _pipeline_class_name(model_id: str) -> str:
    """The ``_class_name`` from the checkpoint's ``model_index.json``."""
    from diffusers import DiffusionPipeline

    try:
        index = DiffusionPipeline.load_config(model_id)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read model_index.json for '{model_id}'. Ensure the "
            f"model_id is a diffusers pipeline. Error: {exc}"
        ) from exc
    return index["_class_name"]
