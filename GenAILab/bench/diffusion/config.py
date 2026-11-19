# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Config parsing for text-to-image diffusion experiments.

A diffusion document has a ``model`` section, ``metrics``, and optionally
``precision`` with a ``recipe``, ``profiler`` and ``run_group``. ``precision``
names the components to quantize; the others stay in floating point. Export and
analysis are not supported for diffusion yet and are rejected by name, so a
config written for a later phase fails loudly instead of running FP silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from GenAILab.bench.precision import Granularity, WeightPrecision, resolve_qtype
from GenAILab.bench.yaml_config_parser import (
    ProfilerConfig,
    ResolvedMetric,
    YAMLConfigParser,
)
from GenAILab.qai_hub_lm.diffusion.base import (
    COMPONENTS,
    DiffusionModel,
    get_diffusion_model_class,
)

_UNSUPPORTED_SECTIONS = (
    "dataset",
    "export",
    "eval_in_onnx",
    "analysis",
)
#: Components a ``precision`` section may quantize.
QUANTIZABLE_COMPONENTS = COMPONENTS
_WEIGHT_QTYPES = ("int4", "int8", "int16")
_ACTIVATION_QTYPES = ("int8", "int16")
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
class ComponentPrecision:
    """Quantization of one component: integer weights and activations."""

    weight: WeightPrecision
    activations: Any  # aimet qtype

    def to_dict(self) -> dict:
        return {"weight": self.weight.to_dict(), "activations": repr(self.activations)}


#: Calibration defaults: 32 prompts give 640 UNet calls at 10 steps with guidance.
DEFAULT_CALIBRATION_PROMPTS = 32


@dataclass(frozen=True)
class CalibrationConfig:
    """The ``Calibration`` recipe: generate from held-out PartiPrompts.

    The prompts are drawn from the PartiPrompts that no metric of the document
    scores, so calibration never sees the evaluation set. Every quantized
    component is calibrated on the same prompts, separately, with the other
    components in float.
    """

    num_prompts: int = DEFAULT_CALIBRATION_PROMPTS
    sample_seed: int = 0

    def to_dict(self) -> dict:
        return {"num_prompts": self.num_prompts, "sample_seed": self.sample_seed}


@dataclass(frozen=True)
class DiffusionConfig:
    """The fully parsed diffusion document."""

    model: DiffusionModelConfig
    metrics: tuple[ResolvedMetric, ...]
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    run_group: str | None = None
    #: Component name -> precision, for the quantized components only. Empty
    #: for an FP run.
    precision: dict[str, ComponentPrecision] = field(default_factory=dict)
    #: Set exactly when ``precision`` is.
    calibration: CalibrationConfig | None = None

    @property
    def is_quantized(self) -> bool:
        return bool(self.precision)

    def precision_dict(self) -> dict:
        """``precision`` as recorded in the results."""
        return {name: p.to_dict() for name, p in self.precision.items()}


def parse_diffusion_document(doc: dict) -> DiffusionConfig:
    """Validate and parse one diffusion config document. Consumes ``doc``."""
    for section in _UNSUPPORTED_SECTIONS:
        if section in doc:
            raise ValueError(
                f"'{section}' is not supported for text-to-image diffusion "
                f"experiments yet."
            )

    model = _parse_model(doc.pop("model", None))
    precision = _parse_precision(doc.pop("precision", None))
    calibration = _parse_recipe(doc.pop("recipe", None))
    if precision and calibration is None:
        raise ValueError(
            "A diffusion 'precision' section needs a 'recipe' to compute its "
            "encodings: add `recipe: {name: Calibration}`."
        )
    if calibration is not None and not precision:
        raise ValueError(
            "A diffusion 'recipe' needs a 'precision' section naming the "
            f"components to quantize ({list(QUANTIZABLE_COMPONENTS)})."
        )

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
        model=model,
        metrics=tuple(metrics),
        profiler=profiler,
        run_group=run_group,
        precision=precision,
        calibration=calibration,
    )


def _parse_precision(section) -> dict[str, ComponentPrecision]:
    """``precision: {<component>: {weight: ..., activations: ...}}``."""
    if section is None:
        return {}
    if not isinstance(section, dict) or not section:
        raise ValueError(
            "A diffusion 'precision' section maps component names to their "
            f"precision, e.g. `precision: {{unet: {{weight: int8, activations: int16}}}}`; "
            f"got {section!r}."
        )
    precision = {}
    for name, block in section.items():
        if name not in COMPONENTS:
            raise ValueError(
                f"Unknown diffusion component {name!r} in 'precision'. "
                f"Known: {list(COMPONENTS)}."
            )
        if name not in QUANTIZABLE_COMPONENTS:
            raise ValueError(
                f"Quantizing the {name} is not supported yet; only "
                f"{list(QUANTIZABLE_COMPONENTS)} can be quantized. Omit it from "
                f"'precision' to keep it in floating point."
            )
        precision[name] = _parse_component_precision(name, block)
    return precision


def _parse_component_precision(name: str, block) -> ComponentPrecision:
    if not isinstance(block, dict):
        raise ValueError(
            f"'precision.{name}' must map 'weight' and 'activations', got {block!r}."
        )
    block = dict(block)
    weight = block.pop("weight", None)
    activations = block.pop("activations", None)
    if block:
        raise ValueError(
            f"Unrecognized keys in 'precision.{name}': {sorted(block)}. "
            f"Expected 'weight' and 'activations'."
        )
    if weight is None or activations is None:
        raise ValueError(f"'precision.{name}' needs both 'weight' and 'activations'.")

    if not isinstance(weight, dict):
        weight = {"qtype": weight}
    weight = dict(weight)
    qtype_name = weight.get("qtype")
    if isinstance(qtype_name, int) and not isinstance(qtype_name, bool):
        qtype_name = f"int{qtype_name}"
    if qtype_name not in _WEIGHT_QTYPES:
        raise ValueError(
            f"'precision.{name}.weight' qtype must be one of {_WEIGHT_QTYPES}, "
            f"got {weight.get('qtype')!r}."
        )
    weight["qtype"] = qtype_name
    granularity = weight.get("granularity", Granularity.PCQ.value)
    if granularity != Granularity.PCQ.value:
        raise ValueError(
            f"'precision.{name}.weight' supports granularity PCQ only so far, "
            f"got {granularity!r}."
        )
    unknown = set(weight) - {"qtype", "granularity"}
    if unknown:
        raise ValueError(
            f"Unrecognized keys in 'precision.{name}.weight': {sorted(unknown)}."
        )

    if isinstance(activations, int) and not isinstance(activations, bool):
        activations = f"int{activations}"
    if activations not in _ACTIVATION_QTYPES:
        raise ValueError(
            f"'precision.{name}.activations' must be one of {_ACTIVATION_QTYPES}, "
            f"got {activations!r}."
        )
    return ComponentPrecision(
        weight=WeightPrecision.from_dict(weight),
        activations=resolve_qtype(activations),
    )


def _parse_recipe(section) -> CalibrationConfig | None:
    """``recipe: {name: Calibration, num_prompts: 32, sample_seed: 0}``.

    A one-step list is accepted too, as the LLM configs write it.
    """
    if section is None:
        return None
    if isinstance(section, list):
        if len(section) != 1:
            raise ValueError(
                "Diffusion recipes support a single 'Calibration' step so far, "
                f"got {len(section)} steps."
            )
        (section,) = section
    if not isinstance(section, dict) or section.get("name") != "Calibration":
        name = section.get("name") if isinstance(section, dict) else section
        raise ValueError(
            f"Diffusion recipes support 'Calibration' only so far, got {name!r}."
        )
    kwargs = {k: v for k, v in section.items() if k != "name"}
    unknown = set(kwargs) - {"num_prompts", "sample_seed"}
    if unknown:
        raise ValueError(
            f"Unrecognized keys in the diffusion Calibration recipe: {sorted(unknown)}. "
            f"Expected 'num_prompts' and 'sample_seed'."
        )
    for key, value in kwargs.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"'recipe.{key}' must be an integer, got {value!r}.")
    if kwargs.get("num_prompts", DEFAULT_CALIBRATION_PROMPTS) <= 0:
        raise ValueError(
            f"'recipe.num_prompts' must be positive, got {kwargs['num_prompts']}."
        )
    return CalibrationConfig(**kwargs)


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
