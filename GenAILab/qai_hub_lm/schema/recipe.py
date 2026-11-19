# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative schema for the ``recipe`` section of a quant config.

One spec class per technique, carrying its wire ``name`` (a ``Literal``), its
kwargs (fields), and its metadata (``phase``/``terminal``/``fp_weight_allowed``
ClassVars). The specs form a discriminated union (``TechniqueSpec``) which IS the
vocabulary -- no separate name enum; the phase/terminal/fp sets are derived from
it. Parallels the dataset specs in ``dataset.py``.

Kwarg fields have no defaults: the backend ``apply()`` signature owns defaults
(torch/onnx differ), and the lowering forwards only set fields via
``model_dump(exclude_unset=True)``. Pre-sim flags (e.g. SpinQuant's rotations) are
instead REQUIRED: they are the identity of which transforms ran, and a defaulted flag
is both invisible in the config and deletable by an ``exclude_defaults`` dump, so a
bare ``- name: SpinQuant`` is a validation error rather than an implicit R1.
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable, Container
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, Union

from pydantic import BaseModel, Field, model_serializer, model_validator

from .components import ALL_COMPONENTS
from .dataset import DatasetSpec, WikitextSpec


class Phase(str, Enum):
    """When in the quant lifecycle a technique runs."""

    pre_sim = "pre_sim"  # operates on the float model, before the sim is built
    on_sim = "on_sim"  # operates on the quantsim (the classic chain)


class _TechniqueSpecBase(BaseModel, extra="forbid"):
    # phase: pre_sim (float model) vs on_sim (quantsim). terminal: produces
    # encodings (a chain lacking one gets an auto-inserted Calibration).
    # fp_weight_allowed: valid when block weights are floating point.
    phase: ClassVar[Phase]
    terminal: ClassVar[bool] = False
    fp_weight_allowed: ClassVar[bool] = False

    dataset: DatasetSpec | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "phase" not in cls.__dict__ and not hasattr(cls, "phase"):
            raise TypeError(f"{cls.__name__} must set the `phase` ClassVar.")


class SpinQuantSpec(_TechniqueSpecBase):
    name: Literal["SpinQuant"]
    phase = Phase.pre_sim
    fp_weight_allowed = True

    enable_r1: bool = False
    enable_r2: bool = False
    enable_r3: bool = False

    @model_validator(mode="after")
    def _at_least_one_rotation(self) -> "SpinQuantSpec":
        if not (self.enable_r1 or self.enable_r2 or self.enable_r3):
            raise ValueError(
                "SpinQuant requires at least one of enable_r1/enable_r2/enable_r3."
            )
        return self


class RemoveQuantizationSpec(_TechniqueSpecBase):
    name: Literal["RemoveQuantization"]
    phase = Phase.on_sim
    terminal = True
    fp_weight_allowed = True


class SkipSpec(_TechniqueSpecBase):
    name: Literal["Skip"]
    phase = Phase.on_sim
    terminal = True
    fp_weight_allowed = True


class ClipSpec(_TechniqueSpecBase):
    name: Literal["Clip"]
    phase = Phase.on_sim
    value: float | None = None


class CalibrationSpec(_TechniqueSpecBase):
    name: Literal["Calibration"]
    phase = Phase.on_sim
    terminal = True
    fp_weight_allowed = True
    num_iterations: int | None = None


class SeqMSESpec(_TechniqueSpecBase):
    name: Literal["SeqMSE"]
    phase = Phase.on_sim
    num_iterations: int | None = None


class AdaScaleSpec(_TechniqueSpecBase):
    name: Literal["AdaScale"]
    phase = Phase.on_sim
    num_batches: int | None = None
    num_iterations: int | None = None
    early_stopping: bool | None = None


# The discriminated union IS the technique vocabulary. Adding a technique = add a
# spec here (and its lowering). No separate name enum to keep in sync.
TechniqueSpec = Annotated[
    Union[
        SpinQuantSpec,
        RemoveQuantizationSpec,
        SkipSpec,
        ClipSpec,
        CalibrationSpec,
        SeqMSESpec,
        AdaScaleSpec,
    ],
    Field(discriminator="name"),
]


# ---------------------------------------------------------------------------
# Derived vocabulary + metadata (the spec union is the single source of truth).
# ---------------------------------------------------------------------------
def _technique_specs() -> tuple[type[_TechniqueSpecBase], ...]:
    return typing.get_args(typing.get_args(TechniqueSpec)[0])


def technique_name_of(spec_cls: type[_TechniqueSpecBase]) -> str:
    """Wire name (the ``name`` Literal value) of a technique spec class."""
    (value,) = typing.get_args(spec_cls.model_fields["name"].annotation)
    return value


def technique_names() -> set[str]:
    """Valid technique names, derived from the spec union."""
    return {technique_name_of(s) for s in _technique_specs()}


def spec_for_technique(name: str) -> type[_TechniqueSpecBase]:
    """Spec class for a technique name (reverse of the discriminator)."""
    for s in _technique_specs():
        if technique_name_of(s) == name:
            return s
    raise KeyError(f"No technique spec for name {name!r}")


# Name-string sets derived from the spec ClassVars.
PRE_SIM_TECHNIQUES = {
    technique_name_of(s) for s in _technique_specs() if s.phase == Phase.pre_sim
}
ON_SIM_TECHNIQUES = {
    technique_name_of(s) for s in _technique_specs() if s.phase == Phase.on_sim
}
TERMINAL_TECHNIQUES = {technique_name_of(s) for s in _technique_specs() if s.terminal}
FP_WEIGHT_ALLOWED_TECHNIQUES = {
    technique_name_of(s) for s in _technique_specs() if s.fp_weight_allowed
}


def spec_kwargs(spec_cls: type[_TechniqueSpecBase]) -> set[str]:
    """Schema kwargs for a spec (fields minus name/dataset) -- what apply() must match."""
    return set(spec_cls.model_fields) - {"name", "dataset"}


def contract_mismatch(
    spec_cls: type[_TechniqueSpecBase],
    apply_fn: Callable,
    ignore_params: Container[str] = frozenset(),
) -> tuple[set[str], set[str]]:
    """Return ``(missing, extra)`` comparing apply()'s explicit params to spec kwargs.

    ``missing`` = spec kwargs apply() doesn't declare (would be swallowed by
    ``**kwargs``); ``extra`` = params apply() declares that the spec can't express.
    Both empty == exact contract. ``**kwargs`` is ignored; ``ignore_params`` are
    fixtures (quantsim/generator/dataloader). Pure/aimet-free.
    """
    expected = spec_kwargs(spec_cls)
    params = inspect.signature(apply_fn).parameters
    implemented = {
        name
        for name, p in params.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        and name not in ignore_params
    }
    return expected - implemented, implemented - expected


def _default_calibration() -> CalibrationSpec:
    return CalibrationSpec(
        name="Calibration",
        dataset=WikitextSpec(name="Wikitext", split="train"),
    )


class Recipe(BaseModel, extra="forbid"):
    """Per-component ordered chains of technique specs.

    ``backbone`` is required; each modality component (``visual``, ``audio``) is
    optional and present only for models that have it. Order is load-bearing;
    within a chain all pre_sim steps must precede on_sim steps, and the pre_sim
    prefix must be identical across components. ``_normalize`` accepts three
    YAML shapes: a single step dict, a top-level list, or
    ``{backbone: [...], visual: [...], audio: [...]}``.

    Adding a component means adding one field here plus one entry in
    ``schema/components.py`` -- the validators and lowering helpers below all
    enumerate ``ALL_COMPONENTS`` rather than hardcoding names.
    """

    backbone: list[TechniqueSpec]
    visual: list[TechniqueSpec] | None = None
    audio: list[TechniqueSpec] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data):
        if isinstance(data, list):
            return {"backbone": data}
        if isinstance(data, dict):
            # already component form?
            if any(comp in data for comp in ALL_COMPONENTS):
                out = dict(data)
                for comp in ALL_COMPONENTS:
                    if comp in out and isinstance(out[comp], dict):
                        out[comp] = [out[comp]]
                return out
            # single step dict (has a technique name and no component keys)
            if "name" in data:
                return {"backbone": [data]}
        return data

    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any) -> Any:
        # Re-narrow to the bare-list form `_normalize` widens: the `backbone:` key
        # only carries information when there is a second component to name.
        data = handler(self)
        if self.visual is None:
            return data["backbone"]
        return data

    @model_validator(mode="after")
    def _semantic_rules(self) -> "Recipe":
        # Rule 1: auto-insert Calibration when a chain has no terminal step
        # (recorded, not warned). The inserted step is on_sim and appended last,
        # so it cannot land before the pre_sim prefix.
        for steps in self._component_chains():
            if not any(s.terminal for s in steps):
                steps.append(_default_calibration())

        # Rule 2: pre_sim steps form a contiguous prefix (all before any on_sim).
        for comp, steps in self._named_chains():
            seen_on_sim = False
            for s in steps:
                if s.phase == Phase.on_sim:
                    seen_on_sim = True
                elif seen_on_sim:  # a pre_sim step after an on_sim step
                    raise ValueError(
                        f"In the '{comp}' chain, pre-sim step '{s.name}' appears "
                        f"after an on-sim step. Pre-sim steps (e.g. SpinQuant) must "
                        f"come first."
                    )

        # Rule 3: the pre_sim prefix must be BYTE-IDENTICAL across all components.
        # Pre-sim techniques (e.g. SpinQuant rotation) act on the whole float
        # model, so per-component prefixes that differ would be meaningless; and
        # the internal representation collapses them to one (see split_recipe).
        # Strict: every component carries the identical prefix (visual can't omit
        # what backbone declares).
        prefixes = {
            comp: self._pre_sim_prefix(steps) for comp, steps in self._named_chains()
        }
        first_comp, first_prefix = next(iter(prefixes.items()))
        for comp, prefix in prefixes.items():
            if prefix != first_prefix:
                raise ValueError(
                    f"Pre-sim prefix of component '{comp}' differs from '{first_comp}'. "
                    f"Pre-sim steps act on the whole float model and must be identical "
                    f"(same techniques, order, and kwargs) across all components."
                )
        return self

    @staticmethod
    def _pre_sim_prefix(steps: list) -> list[dict]:
        """The leading run of pre_sim specs as comparable dicts (for identity check)."""
        prefix = []
        for s in steps:
            if s.phase != Phase.pre_sim:
                break
            prefix.append(s.model_dump())
        return prefix

    # ---- lowering helpers ---------------------------------------------------
    def phased_steps(self, component: str = "backbone") -> tuple[list, list]:
        """Return ``(pre_sim_steps, on_sim_steps)`` for a component, in order."""
        steps = getattr(self, component) or []
        pre = [s for s in steps if s.phase == Phase.pre_sim]
        on_sim = [s for s in steps if s.phase == Phase.on_sim]
        return pre, on_sim

    def to_components(self) -> dict[str, list[dict]]:
        """Lower to ``{component: [step_dict, ...]}`` (name + set kwargs + dataset).

        Uses exclude_unset so backend apply() defaults win for omitted kwargs.
        Round-trips: ``Recipe.model_validate(r.to_components()) == r``.
        """
        out: dict[str, list[dict]] = {}
        for comp, steps in self._named_chains():
            lowered: list[dict] = []
            for s in steps:
                d: dict = {"name": s.name}
                d.update(s.model_dump(exclude_unset=True, exclude={"name", "dataset"}))
                if s.dataset is not None:
                    ds: dict = {"name": s.dataset.name}
                    ds.update(
                        s.dataset.model_dump(exclude_unset=True, exclude={"name"})
                    )
                    d["dataset"] = ds
                lowered.append(d)
            out[comp] = lowered
        return out

    def _component_chains(self):
        for _, steps in self._named_chains():
            yield steps

    def _named_chains(self):
        for comp in ALL_COMPONENTS:
            steps = getattr(self, comp, None)
            if steps is not None:
                yield comp, steps


# Pre-sim / on-sim split + cache identity (operate on lowered step dicts).
def split_recipe(
    recipe: "Recipe",
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Split a validated recipe into ``(pre_sim, on_sim)``.

    ``pre_sim`` is a FLAT list of step dicts (the whole-model pre-sim prefix,
    guaranteed identical across components by Rule 3, so collapsed to one).
    ``on_sim`` is ``{component: [step_dict, ...]}`` (the per-component tails).
    """
    all_components = recipe.to_components()
    pre_sim: list[dict] = [
        s for s in all_components["backbone"] if s["name"] in PRE_SIM_TECHNIQUES
    ]
    on_sim: dict[str, list[dict]] = {
        comp: [s for s in steps if s["name"] not in PRE_SIM_TECHNIQUES]
        for comp, steps in all_components.items()
    }
    return pre_sim, on_sim


def pre_sim_identity(pre_sim: list[dict]) -> dict | None:
    """Pre-sim identity to fold into the cache base hash.

    ``None`` when the pre-sim list is empty -- callers MUST then omit the pre-sim
    hash key so a non-rotated run's cache key stays byte-identical to before.
    """
    identity = {
        step["name"]: {k: v for k, v in step.items() if k not in ("name", "dataset")}
        for step in pre_sim
    }
    return identity or None


def pre_sim_flags(pre_sim: list[dict], technique: str) -> dict | None:
    """Flat kwargs of one pre-sim technique from the pre-sim list, for the backend apply API."""
    for step in pre_sim:
        if step["name"] == technique:
            return {k: v for k, v in step.items() if k not in ("name", "dataset")}
    return None


def has_pre_sim(pre_sim: list[dict]) -> bool:
    """True if the pre-sim list is non-empty."""
    return bool(pre_sim)
