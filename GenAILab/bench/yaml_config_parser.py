# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Config parser for GenAI model testing"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from transformers import AutoConfig

from pydantic import ValidationError

from .export import get_test_artifacts_path
from GenAILab.bench.precision import PrecisionConfig
from GenAILab.qai_hub_lm.models.base import VLM
from GenAILab.qai_hub_lm.models.components import model_components
from GenAILab.qai_hub_lm.schema.components import (
    MODALITY_COMPONENTS,
    require_component,
)
from GenAILab.qai_hub_lm.schema import (
    FP_WEIGHT_ALLOWED_TECHNIQUES,
    PrecisionSchema,
    Recipe,
    contract_mismatch,
    dataset_name_of,
    split_recipe,
    technique_name_of,
)

# apply() parameters that are execution fixtures, not schema-defined recipe
# knobs. Excluded when checking that a lowering implements exactly the schema's
# kwargs for its technique.
_RECIPE_APPLY_FIXTURES = {
    # on-sim technique fixtures
    "quantsim",
    "generator",
    "dataloader",
    "component",
    # Decoder-stack analysis of the float model, derived once by the runner and
    # handed to every on-sim step. A fixture, not a knob: it describes the model,
    # so it is never something a YAML recipe sets.
    "topology",
    # pre-sim technique fixture (the float-model bundle)
    "float_model",
}


@dataclass(frozen=True)
class ResolvedStep:
    """A single recipe step with its resolved implementation class."""

    name: str  # technique name (e.g. "Calibration")
    technique_cls: type  # resolved implementation class
    recipe_kwargs: dict[str, Any] = field(default_factory=dict)
    dataset_cls: type | None = None
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedRecipe:
    """Pre-sim + per-component on-sim chains, unified under one object."""

    pre_sim: tuple[ResolvedStep, ...]
    backbone: tuple[ResolvedStep, ...]
    visual: tuple[ResolvedStep, ...] | None = None
    audio: tuple[ResolvedStep, ...] | None = None

    def component(self, name: str) -> tuple[ResolvedStep, ...] | None:
        """The on-sim chain for a modality component, or ``None`` if absent."""
        require_component(name)
        return getattr(self, name)


@dataclass(frozen=True)
class ResolvedMetric:
    """A single metric with its resolved class and kwargs."""

    name: str
    metric_cls: type
    metric_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedAnalysis:
    """A single analysis pass, resolved against the registry.

    ``analysis_pass`` is the constructed pass instance. ``kwargs`` is the raw
    YAML kwargs it was built from (minus ``name``/``metrics``), kept as given
    for the report and the DB column. ``metrics`` is already the pass's
    resolved metric subset. When top-level metrics exist, these are the same
    ``ResolvedMetric`` objects, so the pass evaluates each metric with the
    top-level kwargs; an omitted subset means all top-level metrics. Without
    top-level metrics, the named metrics are resolved with default kwargs.
    """

    name: str
    analysis_pass: Any
    kwargs: dict[str, Any] = field(default_factory=dict)
    metrics: tuple[ResolvedMetric, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    """Parsed model section."""

    model_cls: type
    model_id: str
    model_type: str
    context_length: int
    sequence_length: int | list[int]
    adaptations: list[str | dict]
    image_size: list[int] | None = None
    #: Padded mel-frame count for audio models -- the audio analogue of
    #: ``image_size``. Any per-encoder constraint on the value is the model
    #: class's to enforce (Qwen3-ASR requires a multiple of ``n_window * 2``;
    #: Gemma4's tower has no such constraint).
    audio_frames: int | None = None
    encodings: str | None = None
    dtype: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def report_modifiers(self, *, dtype: str | None = None) -> dict[str, Any]:
        """Model-section fields to record under the report's ``model_modifiers``.

        Built straight from the parsed config rather than the kwargs used to
        instantiate the model, so the recorded snapshot always reflects the
        authored model section — in particular ``adaptations``, which are baked
        into the model class and never passed to the constructor. ``model_id``,
        ``model_type`` and the model class are separate top-level report fields
        and are intentionally omitted here.

        Args:
            dtype: Resolved dtype name to record when the framework applies a
                default (e.g. ONNX resolves ``float32``). Falls back to the
                configured ``dtype`` and is omitted when neither is set.
        """
        modifiers: dict[str, Any] = dict(self.extra_kwargs)
        modifiers["adaptations"] = self.adaptations
        modifiers["context_length"] = self.context_length
        modifiers["sequence_length"] = self.sequence_length
        dtype_name = dtype or self.dtype
        if dtype_name is not None:
            modifiers["dtype"] = dtype_name
        if self.image_size is not None:
            modifiers["image_size"] = list(self.image_size)
        if self.audio_frames is not None:
            modifiers["audio_frames"] = self.audio_frames
        if self.encodings is not None:
            modifiers["encodings"] = self.encodings
        return modifiers


@dataclass(frozen=True)
class ProfilerConfig:
    """Parsed profiler section."""

    capture_intermediate_data: bool = False
    gpu_meter_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedConfig:
    """The fully typed output of parse_document(). Replaces task_params dict."""

    model: ModelConfig
    precision: PrecisionConfig
    recipe: ResolvedRecipe
    metrics: tuple[ResolvedMetric, ...]
    profiler: ProfilerConfig
    export: str | None = None  # None = no export; str = export dir path
    eval_in_onnx: bool = False
    run_group: str | None = None
    analysis: ResolvedAnalysis | None = None


@dataclass
class AdaptationInfo:
    """Metadata about an adaptation."""

    mixin_cls: type
    exclusive: bool = False  # If True, cannot combine with other adaptations
    required_for_export: bool = False  # If True, auto-enforced when exporting


class YAMLConfigParser:
    recipe_lookup: dict = {}
    model_lookup: dict = {}  # {model_type: model_cls} for VLMs and other special models
    dataset_lookup: dict = {}
    metrics_lookup: dict = {}
    adaptation_lookup: dict = {}  # {(model_type, adaptation_name): AdaptationInfo}
    analysis_lookup: dict = {}  # {pass_name: pass_cls}

    _default_llm_cls: type = None

    # ========================
    # Default LLM registration
    # ========================

    @classmethod
    def register_default_llm(cls, llm_cls: type):
        """Register the default LLM class.

        Called by framework-specific modules on import.
        Fails if a different framework has already registered.

        Usage (in torch/models/llm.py):
            @YAMLConfigParser.register_default_llm
            class LLM_Torch(LLM):
                ...

        Usage (in onnx/models/llm.py):
            @YAMLConfigParser.register_default_llm
            class LLM_ONNX(LLM):
                ...
        """
        if cls._default_llm_cls is not None and cls._default_llm_cls is not llm_cls:
            raise RuntimeError(
                f"LLM class ({cls._default_llm_cls.__name__}) is already registered. "
                f"Only one framework can be active at a time."
            )
        cls._default_llm_cls = llm_cls
        return llm_cls

    @classmethod
    def get_default_llm(cls) -> type:
        """Get the registered default LLM class."""
        if cls._default_llm_cls is None:
            raise RuntimeError(
                "No default LLM class registered. "
                "Ensure you import the framework module before parsing configs."
            )
        return cls._default_llm_cls

    # ========================
    # Adaptation registration
    # ========================

    @classmethod
    def register_adaptation(
        cls,
        adaptation_name: str,
        model_type: str = "*",
        *,
        exclusive: bool = False,
        required_for_export: bool = False,
    ):
        """Register an adaptation mixin.

        Args:
            adaptation_name: Name used in config (e.g., "SHA", "AIHM")
            model_type: HuggingFace model_type this applies to, or "*" for all
            exclusive: If True, cannot combine with other adaptations
            required_for_export: If True, this adaptation is enforced when the
                model will be exported. Skipped when an exclusive adaptation
                (which owns the full pipeline) is already selected.

        Usage:
            @YAMLConfigParser.register_adaptation("SHA", model_type="llama")
            class LlamaSHAMixin:
                ...

            @YAMLConfigParser.register_adaptation("AIHM", model_type="*", exclusive=True)
            class AIHMMixin:
                ...
        """

        def decorator(mixin_cls):
            info = AdaptationInfo(
                mixin_cls=mixin_cls,
                exclusive=exclusive,
                required_for_export=required_for_export,
            )
            cls.adaptation_lookup[(model_type, adaptation_name)] = info
            return mixin_cls

        return decorator

    # ========================
    # Model registration (for VLMs and special models)
    # ========================

    @classmethod
    def register_model(cls, model_type: str):
        """Register a model class by its model_type.

        Used for VLMs and other special models that need custom handling.
        Regular LLMs use the default LLM class instead.

        Args:
            model_type: HuggingFace model_type this class handles (e.g., "qwen2_vl")

        Usage:
            @YAMLConfigParser.register_model("qwen2_vl")
            class Qwen_25_VL_Torch(VLM):
                ...
        """

        def decorator(model_cls):
            if model_type in cls.model_lookup:
                existing_cls = cls.model_lookup[model_type]
                raise RuntimeError(
                    f"Model type '{model_type}' is already registered by {existing_cls.__name__}. "
                    f"Cannot register {model_cls.__name__} under the same model_type."
                )
            cls.model_lookup[model_type] = model_cls
            return model_cls

        return decorator

    # ========================
    # Other registrations
    # ========================

    @classmethod
    def register_recipe(cls, spec_cls):
        """Register a recipe lowering, bound to its schema technique ``spec_cls``.

        The ``spec_cls`` argument (e.g. ``SeqMSESpec``) is the hard reference
        from the lowering to the schema. This is the schema<->lowering binding
        *and* its enforcement: at import time it verifies the lowering's
        ``apply()`` implements EXACTLY the spec's kwargs -- no more, no less
        (fixtures and ``**kwargs`` excluded). Because both the torch and onnx
        lowerings register against the same spec, this equality pivots on the
        schema and transitively guarantees a recipe valid under one backend is
        valid under the other.

        Usage:
            @YAMLConfigParser.register_recipe(SeqMSESpec)
            class SeqMSE(QuantizationTechnique): ...
        """
        name = technique_name_of(spec_cls)

        def decorator(recipe_cls):
            missing, extra = contract_mismatch(
                spec_cls, recipe_cls.apply, ignore_params=_RECIPE_APPLY_FIXTURES
            )
            if missing or extra:
                problems = []
                if missing:
                    problems.append(
                        f"does not implement schema kwargs {sorted(missing)} "
                        f"(they would be silently swallowed by **kwargs)"
                    )
                if extra:
                    problems.append(
                        f"implements kwargs {sorted(extra)} not in the schema "
                        f"(add them to {spec_cls.__name__}, or the other backend's "
                        f"lowering would reject valid recipes)"
                    )
                raise TypeError(
                    f"{recipe_cls.__name__}.apply() is out of contract with "
                    f"{spec_cls.__name__} ('{name}'): " + "; ".join(problems)
                )

            cls.recipe_lookup[name] = recipe_cls
            return recipe_cls

        return decorator

    @classmethod
    def register_analysis(cls, pass_cls):
        """Register an analysis pass under its class name.

        The pass's ``__init__`` keyword arguments are its config kwargs; see
        ``bench/analysis.py``. Passes are registered only by the runner
        that can execute them (``bench/onnx/test_genai.py`` imports
        ``bench/onnx/analysis_passes.py``).
        """
        cls.analysis_lookup[pass_cls.__name__] = pass_cls
        return pass_cls

    @classmethod
    def register_dataset(cls, spec_cls):
        """Register a dataset lowering, bound to its schema ``spec_cls``.

        The ``spec_cls`` argument (e.g. ``WikitextSpec``) is a hard reference
        from the lowering to the schema: a dataset can only be registered under
        a spec the schema declares, so the spec union and the registry cannot
        silently drift.

        Unlike ``register_recipe``, this does NOT enforce kwarg agreement yet:
        the ``load_encoded_dataset`` signatures are heterogeneous (tokenizer vs
        processor, image_size, source_datasets, ...). The per-dataset spec now
        makes that check straightforward to add; it is intentionally left off.

        Usage:
            @YAMLConfigParser.register_dataset(WikitextSpec)
            class Wikitext(TextDataset): ...
        """
        name = dataset_name_of(spec_cls)

        def decorator(dataset_cls):
            cls.dataset_lookup[name] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def register_metric(cls, metric_cls):
        cls.metrics_lookup[metric_cls.__name__] = metric_cls
        return metric_cls

    # ========================
    # Lookups
    # ========================

    @classmethod
    def get_recipe(cls, recipe_name):
        return cls.recipe_lookup[recipe_name]

    @classmethod
    def get_dataset(cls, dataset_name):
        return cls.dataset_lookup[dataset_name]

    @classmethod
    def get_metric(cls, metrics_name):
        return cls.metrics_lookup[metrics_name]

    @classmethod
    def _resolve_metric(cls, metric: dict) -> ResolvedMetric:
        metric_name = metric["name"]
        try:
            metric_cls = cls.get_metric(metric_name)
        except LookupError as exc:
            raise LookupError(
                f"Specified metric name ({metric_name}) not found."
            ) from exc
        return ResolvedMetric(
            name=metric_name,
            metric_cls=metric_cls,
            metric_kwargs={k: v for k, v in metric.items() if k != "name"},
        )

    @classmethod
    def get_analysis(cls, name: str) -> type:
        if name in cls.analysis_lookup:
            return cls.analysis_lookup[name]
        raise LookupError(
            f"Analysis pass '{name}' not found. Registered analysis passes: "
            f"{', '.join(sorted(cls.analysis_lookup)) or 'none'}."
        )

    @classmethod
    def _resolve_analysis(
        cls, pass_dict: dict, top_level_metrics: list[ResolvedMetric]
    ) -> ResolvedAnalysis:
        """Resolve one analysis pass dict against the registry.

        The pass is constructed from its kwargs (everything except ``name`` and
        ``metrics``); its ``__init__`` is the config contract, so an unknown or
        missing kwarg raises ``TypeError`` and a bad value ``ValueError``.

        The ``metrics`` subset rules:
          - A pass with ``uses_metrics = False`` takes no ``metrics`` list and
            needs none anywhere in the config.
          - Otherwise ``metrics``, when given, is a non-empty list of names.
          - With top-level metrics, every name must be one of them, and the
            subset reuses those ``ResolvedMetric`` objects, so it inherits their
            kwargs instead of restating them. Omitted means all of them.
          - Without top-level metrics there is no baseline run on the plain
            sim, so the pass must name its metrics; they get default kwargs.
        """
        pass_name = pass_dict["name"]
        pass_metrics = pass_dict.get("metrics")
        pass_kwargs = {
            k: v for k, v in pass_dict.items() if k not in ("name", "metrics")
        }
        pass_cls = cls.get_analysis(pass_name)

        if not getattr(pass_cls, "uses_metrics", True):
            if pass_metrics is not None:
                raise RuntimeError(
                    f"analysis pass '{pass_name}' does not evaluate metrics, "
                    f"so it takes no 'metrics' list (got {pass_metrics!r})."
                )
            subset_metrics = ()
        elif pass_metrics is None:
            if not top_level_metrics:
                raise RuntimeError(
                    "Top-level 'metrics' section not specified, and analysis "
                    f"pass '{pass_name}' does not specify its own 'metrics' "
                    "list either -- at least one of the two is required."
                )
            subset_metrics = tuple(top_level_metrics)
        else:
            if (
                not isinstance(pass_metrics, list)
                or not pass_metrics
                or not all(isinstance(n, str) for n in pass_metrics)
            ):
                raise RuntimeError(
                    f"Invalid analysis section: pass '{pass_name}' 'metrics' must "
                    f"be a non-empty list of metric names, got {pass_metrics!r}."
                )
            if top_level_metrics:
                by_name = {m.name: m for m in top_level_metrics}
                unknown = set(pass_metrics) - by_name.keys()
                if unknown:
                    raise RuntimeError(
                        f"analysis pass '{pass_name}' references metric(s) "
                        f"{sorted(unknown)} not present in the top-level "
                        f"'metrics' section ({sorted(by_name)})."
                    )
                subset_metrics = tuple(by_name[n] for n in pass_metrics)
            else:
                subset_metrics = tuple(
                    cls._resolve_metric({"name": n}) for n in pass_metrics
                )

        try:
            analysis_pass = pass_cls(**pass_kwargs)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid analysis pass '{pass_name}': {exc}") from exc

        return ResolvedAnalysis(
            name=pass_name,
            analysis_pass=analysis_pass,
            kwargs=pass_kwargs,
            metrics=subset_metrics,
        )

    @classmethod
    def detect_model_type(cls, model_id: str) -> str:
        """Detect model type from HuggingFace model_id or local checkpoint."""
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            return config.model_type
        except Exception as e:
            raise ValueError(
                f"Could not detect model type from '{model_id}'. "
                f"Ensure the model_id is valid or config.json is present in checkpoint. "
                f"Error: {e}"
            )

    # ========================
    # Adaptation handling
    # ========================

    @classmethod
    def _get_adaptation_info(
        cls, model_type: str, adaptation_name: str
    ) -> AdaptationInfo:
        """Find the AdaptationInfo for this model type and adaptation."""
        # Try model-specific first, then universal
        if (model_type, adaptation_name) in cls.adaptation_lookup:
            return cls.adaptation_lookup[(model_type, adaptation_name)]
        if ("*", adaptation_name) in cls.adaptation_lookup:
            return cls.adaptation_lookup[("*", adaptation_name)]

        raise LookupError(
            f"No '{adaptation_name}' adaptation registered for model_type='{model_type}'."
        )

    @staticmethod
    def _normalize_adaptations(
        adaptations_raw: list,
    ) -> tuple[list[str], dict[str, dict]]:
        """Normalize a mixed list of adaptation entries.

        Each entry can be:
        - A string:  ``"SHA"``
        - A single-key dict:  ``{"AttentionMaskScale": {"layer_multipliers": {0: 0.8}}}``

        Returns:
            (adaptation_names, adaptation_kwargs) where *adaptation_kwargs*
            maps adaptation name → dict of keyword arguments.
        """
        names: list[str] = []
        kwargs: dict[str, dict] = {}
        for entry in adaptations_raw:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError(
                        "Dict adaptation entry must have exactly one key "
                        f"(the adaptation name), got {list(entry.keys())}"
                    )
                name, params = next(iter(entry.items()))
                names.append(name)
                kwargs[name] = params if isinstance(params, dict) else {}
            else:
                raise ValueError(
                    f"Each adaptation must be a string or single-key dict, "
                    f"got {type(entry)}"
                )
        return names, kwargs

    @classmethod
    def _validate_adaptations(cls, model_type: str, adaptations: list[str]):
        """Validate that the requested adaptations are compatible."""
        if not adaptations:
            return

        for name in adaptations:
            info = cls._get_adaptation_info(model_type, name)
            if info.exclusive and len(adaptations) > 1:
                raise ValueError(
                    f"Adaptation '{name}' is exclusive and cannot be combined with "
                    f"other adaptations: {[a for a in adaptations if a != name]}"
                )

    @classmethod
    def get_required_export_adaptations(cls, model_type: str) -> list[str]:
        """Return non-exclusive adaptation names required for export for a model_type."""
        return [
            name
            for (mt, name), info in cls.adaptation_lookup.items()
            if mt == model_type and info.required_for_export and not info.exclusive
        ]

    @classmethod
    def get_model_class(
        cls,
        model_type: str,
        adaptations: Optional[list[str]] = None,
        adaptation_kwargs: Optional[dict[str, dict]] = None,
    ) -> type:
        """Get the final model class with adaptations applied.

        First checks model_lookup for a registered model (e.g., VLMs),
        then falls back to the default LLM class.

        Args:
            model_type: HuggingFace model_type string.
            adaptations: List of adaptation names.
            adaptation_kwargs: Per-adaptation keyword arguments to set as
                class attributes on the dynamically-created class.  Keyed
                by adaptation name.
        """
        if adaptation_kwargs is None:
            adaptation_kwargs = {}

        # Check for registered model first (VLMs, special models)
        if model_type in cls.model_lookup:
            base_cls = cls.model_lookup[model_type]
        else:
            # Fall back to default LLM
            base_cls = cls.get_default_llm()

        if not adaptations:
            return base_cls

        cls._validate_adaptations(model_type, adaptations)

        # Apply each adaptation
        result_cls = base_cls
        for adaptation_name in adaptations:
            info = cls._get_adaptation_info(model_type, adaptation_name)
            new_name = f"{result_cls.__name__}_{adaptation_name}"
            cls_dict = adaptation_kwargs.get(adaptation_name, {})
            result_cls = type(new_name, (info.mixin_cls, result_cls), cls_dict)

        return result_cls

    # ========================
    # Config validation & parsing
    # ========================

    @classmethod
    def validate_config(cls, doc):
        if "model" not in doc:
            raise RuntimeError("Model section not specified.")
        # Top-level 'metrics' may be omitted only when an analysis pass names
        # its own metrics or uses none; _resolve_analysis checks that.
        if "metrics" not in doc and "analysis" not in doc:
            raise RuntimeError("Metrics not specified.")

        if not isinstance(doc["model"], dict):
            raise RuntimeError(
                "Multiple models cannot be specified in a single document."
            )
        if "recipe" in doc and not isinstance(doc["recipe"], (dict, list)):
            raise RuntimeError("Recipe must be a dict or list.")

        if "model_id" not in doc["model"]:
            raise RuntimeError("Model 'model_id' not specified.")
        if "sequence_length" not in doc["model"]:
            raise RuntimeError("Sequence length not specified.")
        sl = doc["model"]["sequence_length"]
        if not isinstance(sl, (int, list)):
            raise RuntimeError("sequence_length must be an int or list of ints.")
        if isinstance(sl, list) and not all(isinstance(x, int) for x in sl):
            raise RuntimeError("sequence_length must be an int or list of ints.")
        if "context_length" not in doc["model"]:
            raise RuntimeError("Context length not specified.")

        # Validate + normalize the recipe section via the shared schema. The
        # schema (qai_hub_lm/schema) owns all recipe SHAPE, VOCABULARY, and
        # SEMANTIC RULES that used to live as ~140 lines of hand-written checks
        # here: single/list/component normalization, auto-insert Calibration when
        # no terminal step is present, SpinQuant >=1-rotation, pre-sim-before-
        # post-sim ordering, and backbone=>visual SpinQuant consistency.
        #
        # ``Recipe.to_components()`` lowers the validated recipe back into the
        # ``{component: [step_dict, ...]}`` shape that ``parse_document`` resolves
        # (name -> recipe class) downstream, so the resolution code is unchanged.
        #
        # NOTE: auto-insert of Calibration is now silent (the schema records it);
        # the previous warnings.warn banner is intentionally dropped.
        if "recipe" in doc:
            try:
                recipe = Recipe.model_validate(doc["recipe"])
            except ValidationError as exc:
                raise RuntimeError(f"Invalid recipe section:\n{exc}") from exc
            doc["recipe"] = recipe.to_components()

            # Cross-section rule (NOT in the recipe schema, since it reads
            # precision): when block weights are floating point, weight-modifying
            # recipes are no-ops, so only the FP-weight-allowed techniques are
            # permitted. Reject anything else early so users don't silently get
            # wrong results. Reads the VALIDATED precision schema rather than
            # re-parsing the raw dict.
            try:
                precision_schema = PrecisionSchema.model_validate(
                    doc.get("precision") or {}
                )
            except ValidationError as exc:
                raise RuntimeError(f"Invalid precision section:\n{exc}") from exc
            if precision_schema.blocks["default"].is_float:
                allowed = set(FP_WEIGHT_ALLOWED_TECHNIQUES)  # set of name strings
                for comp_name, recipe_list in doc["recipe"].items():
                    for step in recipe_list:
                        if step["name"] not in allowed:
                            raise RuntimeError(
                                f"Recipe '{step['name']}' modifies weights and is "
                                f"incompatible with floating-point block weights "
                                f"(precision.blocks.qtype). Only "
                                f"{sorted(allowed)} are allowed."
                            )

        # Backward compatibility: migrate top-level dataset into backbone component
        if "dataset" in doc:
            first_backbone = doc["recipe"]["backbone"][0]
            if "dataset" not in first_backbone:
                first_backbone["dataset"] = doc.pop("dataset")
            else:
                doc.pop("dataset")  # Component has its own dataset, discard top-level

        if "metrics" in doc:
            metrics = (
                doc["metrics"] if isinstance(doc["metrics"], list) else [doc["metrics"]]
            )
            for metric in metrics:
                if "name" not in metric:
                    raise RuntimeError("Metric name not specified.")

        # Only the analysis section's shape is checked here: exactly one pass
        # dict with a name. A list is accepted so that multiple passes can be
        # added later without a config format change. Everything that needs the
        # registered pass class is in _resolve_analysis.
        if "analysis" in doc:
            passes = doc["analysis"]
            if isinstance(passes, dict):
                passes = [passes]
            if not isinstance(passes, list) or not all(
                isinstance(p, dict) for p in passes
            ):
                raise RuntimeError(
                    "Invalid analysis section: expected a pass dict or a list of "
                    f"pass dicts, got {doc['analysis']!r}."
                )
            if len(passes) != 1:
                raise RuntimeError(
                    f"Exactly one analysis pass is supported per document, got "
                    f"{len(passes)} ({[p.get('name') for p in passes]}). Multiple "
                    f"analysis passes are not yet supported."
                )
            if not isinstance(passes[0].get("name"), str):
                raise RuntimeError(
                    "Invalid analysis section: pass 'name' not specified."
                )
            doc["analysis"] = passes

    @classmethod
    def parse_document(
        cls, doc, export_base_dir="GenAILab/artifacts/exports"
    ) -> ParsedConfig:
        cls.validate_config(doc)

        # Export/eval_in_onnx/run_group
        export_val = doc.pop("export", False)
        if not isinstance(export_val, (bool, str)):
            raise ValueError("Export field must be a boolean or a string path.")

        eval_in_onnx = doc.pop("eval_in_onnx", False)
        if not isinstance(eval_in_onnx, bool):
            raise ValueError("eval_in_onnx field must be a boolean value.")

        run_group = doc.pop("run_group", None)

        if eval_in_onnx and not export_val:
            warnings.warn(
                "eval_in_onnx is enabled, but export is disabled. Overriding export to True."
            )

        if export_val or eval_in_onnx:
            export_dir = (
                get_test_artifacts_path(doc, base_dir=export_base_dir)
                if isinstance(export_val, bool)
                else export_val
            )
            Path(export_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(export_dir, "config.yaml"), "w") as file:
                yaml.dump(doc, file)
        else:
            export_dir = None

        # Model setup
        model_id = doc["model"]["model_id"]
        adaptations_raw = doc["model"].pop("adaptations", [])
        adaptation_names, adaptation_kwargs = cls._normalize_adaptations(
            adaptations_raw
        )
        model_type = cls.detect_model_type(model_id)

        try:
            model_cls = cls.get_model_class(
                model_type, adaptation_names, adaptation_kwargs
            )
            model_dict = doc.pop("model")
        except LookupError as exc:
            raise LookupError(
                f"Failed to configure model for model_id='{model_id}', "
                f"adaptations={adaptation_names}."
            ) from exc

        # Check that required export adaptations are present.
        # Skip when an exclusive adaptation (e.g. AIHM) is selected — it owns
        # the full pipeline including export.
        # A local directory as model_id means the model is already exported.
        is_local_checkpoint = os.path.isdir(model_id)
        will_export = export_dir or (
            "ONNX" in cls.get_default_llm().__name__ and not is_local_checkpoint
        )
        has_exclusive = any(
            cls._get_adaptation_info(model_type, a).exclusive for a in adaptation_names
        )
        if will_export and not has_exclusive:
            required = cls.get_required_export_adaptations(model_type)
            missing = [a for a in required if a not in adaptation_names]
            if missing:
                raise ValueError(
                    f"ONNX export for model_type '{model_type}' requires the "
                    f"following adaptation(s): {missing}.\n"
                    f"Add them under 'model.adaptations' in your YAML config:\n"
                    f"  model:\n"
                    f"    model_id: {model_id}\n"
                    f"    adaptations:\n" + "".join(f"      - {a}\n" for a in missing)
                )

        # Extract known model fields
        context_length = model_dict.pop("context_length")
        sequence_length = model_dict.pop("sequence_length")
        image_size = model_dict.pop("image_size", None)
        audio_frames = model_dict.pop("audio_frames", None)
        encodings = model_dict.pop("encodings", None)
        dtype = model_dict.pop("dtype", None)
        # model_id already extracted above; remove it from the dict
        model_dict.pop("model_id")
        # Remaining keys go into extra_kwargs
        extra_kwargs = model_dict

        model_config = ModelConfig(
            model_cls=model_cls,
            model_id=model_id,
            model_type=model_type,
            context_length=context_length,
            sequence_length=sequence_length,
            adaptations=adaptations_raw,
            image_size=image_size,
            audio_frames=audio_frames,
            encodings=encodings,
            dtype=dtype,
            extra_kwargs=extra_kwargs,
        )

        # Precision config: validate through the shared schema (shape, vocab,
        # extra="forbid" typo-catching, polymorphic blocks/visual normalization),
        # then resolve the validated schema into the aimet-coupled PrecisionConfig
        # (qtype objects). validate_config already validated this section; re-run
        # here since parse_document may be called on the raw doc.
        precision_schema = PrecisionSchema.model_validate(
            doc.pop("precision", None) or {}
        )
        precision = PrecisionConfig.from_schema(precision_schema)

        # Recipe parsing. ``validate_config`` already validated + normalized the
        # recipe via the shared schema and lowered it to component form; here we
        # split off the pre-sim steps (e.g. SpinQuant, which rotates the float
        # model before the sim is built) and resolve only the on-sim chains to
        # recipe classes.
        if "recipe" in doc:
            # doc["recipe"] is already the lowered component dict (from
            # validate_config). Reconstruct the validated Recipe to split it.
            recipe_obj = Recipe.model_validate(doc["recipe"])
            pre_sim_steps, on_sim_components = split_recipe(recipe_obj)

            # Helper to resolve a step dict to ResolvedStep
            def resolve_step(step_config: dict) -> ResolvedStep:
                recipe_name = step_config["name"]
                try:
                    recipe_cls = cls.get_recipe(recipe_name)
                except LookupError as exc:
                    raise LookupError(
                        f"Specified quantization recipe name ({recipe_name}) not found."
                    ) from exc

                recipe_kwargs = {
                    k: v for k, v in step_config.items() if k not in ("name", "dataset")
                }
                dataset_cls = None
                dataset_kwargs = {}
                if "dataset" in step_config:
                    dataset_config = step_config["dataset"]
                    dataset_name = dataset_config["name"]
                    try:
                        dataset_cls = cls.get_dataset(dataset_name)
                    except LookupError as exc:
                        raise LookupError(
                            f"Specified dataset name ({dataset_name}) not found."
                        ) from exc
                    dataset_kwargs = {
                        k: v for k, v in dataset_config.items() if k != "name"
                    }

                return ResolvedStep(
                    name=recipe_name,
                    technique_cls=recipe_cls,
                    recipe_kwargs=recipe_kwargs,
                    dataset_cls=dataset_cls,
                    dataset_kwargs=dataset_kwargs,
                )

            pre_sim_resolved = tuple(resolve_step(s) for s in pre_sim_steps)
            backbone_resolved = tuple(
                resolve_step(s) for s in on_sim_components["backbone"]
            )
            component_resolved = {
                comp: tuple(resolve_step(s) for s in on_sim_components[comp])
                for comp in MODALITY_COMPONENTS
                if comp in on_sim_components
            }
            del doc["recipe"]
        else:
            has_encodings = encodings is not None
            default_recipe = "Skip" if has_encodings else "RemoveQuantization"
            default_cls = cls.get_recipe(default_recipe)
            default_step = ResolvedStep(
                name=default_recipe,
                technique_cls=default_cls,
                recipe_kwargs={},
                dataset_cls=None,
                dataset_kwargs={},
            )
            pre_sim_resolved = ()
            backbone_resolved = (default_step,)
            # Each modality component the model actually declares gets the
            # default step; plain LLMs get none. Keyed off the model class's
            # COMPONENTS rather than `issubclass(model_cls, VLM)`, because audio
            # models also subclass VLM and must NOT acquire a visual recipe.
            component_resolved = {
                component.name: (default_step,)
                for component in model_components(model_cls)
            }

        resolved_recipe = ResolvedRecipe(
            pre_sim=pre_sim_resolved,
            backbone=backbone_resolved,
            **component_resolved,
        )

        # Metrics parsing. Empty only when an analysis pass supplies its own
        # metrics or uses none (_resolve_analysis enforces this), meaning
        # "no baseline run".
        metrics_list = doc.pop("metrics", [])
        if not isinstance(metrics_list, list):
            metrics_list = [metrics_list]
        resolved_metrics = [cls._resolve_metric(metric) for metric in metrics_list]

        resolved_analysis = None
        if "analysis" in doc:
            # Exactly one pass, enforced by validate_config.
            (pass_dict,) = doc.pop("analysis")
            resolved_analysis = cls._resolve_analysis(pass_dict, resolved_metrics)

        # Profiler parsing
        profiler_dict = doc.pop("profiler", {})
        capture_intermediate = profiler_dict.pop("capture_intermediate_data", False)
        profiler_config = ProfilerConfig(
            capture_intermediate_data=capture_intermediate,
            gpu_meter_kwargs=profiler_dict,
        )

        if len(doc) > 0:
            raise ValueError(f"Unrecognized sections in config: {doc.keys()}")

        return ParsedConfig(
            model=model_config,
            precision=precision,
            recipe=resolved_recipe,
            metrics=tuple(resolved_metrics),
            profiler=profiler_config,
            export=export_dir,
            eval_in_onnx=eval_in_onnx,
            run_group=run_group,
            analysis=resolved_analysis,
        )

    @classmethod
    def parse(cls, filename, export_base_dir="GenAILab/artifacts/exports"):
        print(filename)
        with open(filename, "r") as file:
            docs = yaml.safe_load_all(file)
            for doc in docs:
                yield cls.parse_document(doc, export_base_dir=export_base_dir)
