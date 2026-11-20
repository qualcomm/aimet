# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative schema for the ``dataset`` block of a recipe step.

One spec class per dataset, each declaring the fields that dataset accepts
(mirrors its ``load_encoded_dataset`` kwargs). The specs form a discriminated
union (``DatasetSpec``) which IS the vocabulary -- no name enum. Fields carry no
defaults (omitted -> dataset's own default via exclude_unset); runtime mechanics
(tokenizer/context_length) are injected by the consumer, not recorded here.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, PositiveInt


class _DatasetSpecBase(BaseModel, extra="forbid"):
    pass


class WikitextSpec(_DatasetSpecBase):
    name: Literal["Wikitext"]
    split: str | None = None


class TinyMMLUSpec(_DatasetSpecBase):
    name: Literal["TinyMMLU"]
    split: str | None = None


class MMLUSpec(_DatasetSpecBase):
    name: Literal["MMLU"]
    split: str | None = None
    num_fewshot: int | None = None
    fewshot_split: str | None = None


class MMLUProSpec(_DatasetSpecBase):
    name: Literal["MMLUPro"]
    split: str | None = None
    num_fewshot: int | None = None
    fewshot_split: str | None = None


class MMMLUSpec(_DatasetSpecBase):
    name: Literal["MMMLU"]
    split: str | None = None
    num_fewshot: int | None = None


class MMMUSpec(_DatasetSpecBase):
    name: Literal["MMMU"]
    split: str | None = None
    image_size: tuple[int, int] | None = None


class C4Spec(_DatasetSpecBase):
    name: Literal["C4"]
    split: str | None = None
    num_samples: int | None = None


class AOKVQASpec(_DatasetSpecBase):
    name: Literal["AOKVQA"]
    split: str | None = None
    image_size: tuple[int, int] | None = None


class ERQASpec(_DatasetSpecBase):
    name: Literal["ERQA"]
    split: str | None = None


class Where2PlaceSpec(_DatasetSpecBase):
    name: Literal["Where2Place"]
    split: str | None = None
    image_size: tuple[int, int] | None = None


class HypersimSpec(_DatasetSpecBase):
    name: Literal["Hypersim"]
    split: str | None = None
    image_size: tuple[int, int] | None = None


class LibriSpeechSpec(_DatasetSpecBase):
    """LibriSpeech ASR audio dataset (calibration + WER/CER evaluation).

    ``split`` uses the pinned ``all`` config's names (``validation.clean``,
    ``test.clean``). ``num_samples`` takes from the head of the split.
    ``language`` forces the transcription language. ``include_reference`` attaches
    the reference transcript -- evaluation needs it, calibration wants pure tensors.

    ``n_window`` is deliberately not a knob: the dataset reads it from the
    processor, so a YAML value could only disagree with the model.
    """

    name: Literal["LibriSpeech"]
    split: str | None = None
    num_samples: PositiveInt | None = None
    language: str | None = None
    include_reference: bool | None = None


class FleursSpec(_DatasetSpecBase):
    """FLEURS multilingual ASR audio dataset (one language per spec).

    ``language`` is required: a FLEURS language name (``English``, ``Korean``,
    ``Mandarin Chinese``) -- not a config code or a model's own spelling.
    ``split`` is ``train`` / ``validation`` / ``test``. ``num_samples`` takes
    from the head of the split. ``include_reference`` as for LibriSpeech.
    """

    name: Literal["FLEURS"]
    split: str | None = None
    num_samples: PositiveInt | None = None
    language: str
    include_reference: bool | None = None


class GeneratedDatasetSpec(_DatasetSpecBase):
    name: Literal["GeneratedDataset"]
    model_id: str | None = None
    num_inputs: int | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None


class InterleavedSpec(_DatasetSpecBase):
    name: Literal["Interleaved"]
    # Interleaved composes other datasets: each entry is itself a DatasetSpec.
    source_datasets: list["DatasetSpec"] = Field(default_factory=list)


# The union IS the vocabulary; adding a dataset = adding a spec here.
DatasetSpec = Annotated[
    Union[
        WikitextSpec,
        TinyMMLUSpec,
        MMLUSpec,
        MMLUProSpec,
        MMMLUSpec,
        MMMUSpec,
        C4Spec,
        AOKVQASpec,
        ERQASpec,
        Where2PlaceSpec,
        HypersimSpec,
        LibriSpeechSpec,
        FleursSpec,
        GeneratedDatasetSpec,
        InterleavedSpec,
    ],
    Field(discriminator="name"),
]

InterleavedSpec.model_rebuild()  # resolve the DatasetSpec forward ref

# The datasets an ASR error-rate metric (WER / CER) can score on.
ASRDatasetSpec = Annotated[
    Union[LibriSpeechSpec, FleursSpec],
    Field(discriminator="name"),
]


# Derived vocabulary helpers (the spec union is the single source of truth).
def _dataset_specs() -> tuple[type[_DatasetSpecBase], ...]:
    import typing

    return typing.get_args(typing.get_args(DatasetSpec)[0])


def dataset_name_of(spec_cls: type[_DatasetSpecBase]) -> str:
    """Wire name (the ``name`` Literal value) of a dataset spec class."""
    import typing

    (value,) = typing.get_args(spec_cls.model_fields["name"].annotation)
    return value


def dataset_names() -> set[str]:
    """Valid dataset names, derived from the spec union."""
    return {dataset_name_of(s) for s in _dataset_specs()}


def spec_for_dataset(name: str) -> type[_DatasetSpecBase]:
    """Spec class for a dataset name (reverse of the discriminator)."""
    for s in _dataset_specs():
        if dataset_name_of(s) == name:
            return s
    raise KeyError(f"No dataset spec for name {name!r}")
