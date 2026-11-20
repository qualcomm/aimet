# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Schema tests for the audio/ASR dataset spec (schema island: no torch)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from GenAILab.qai_hub_lm.schema.dataset import (
    ASRDatasetSpec,
    DatasetSpec,
    FleursSpec,
    LibriSpeechSpec,
    dataset_name_of,
    dataset_names,
    spec_for_dataset,
)

_ADAPTER = TypeAdapter(DatasetSpec)


class TestLibriSpeechSpecVocabulary:
    def test_name_is_in_the_dataset_vocabulary(self):
        assert "LibriSpeech" in dataset_names()

    def test_spec_lookup_round_trips(self):
        assert spec_for_dataset("LibriSpeech") is LibriSpeechSpec
        assert dataset_name_of(LibriSpeechSpec) == "LibriSpeech"


class TestLibriSpeechSpecRoundTrip:
    def test_discriminated_union_selects_the_librispeech_spec(self):
        spec = _ADAPTER.validate_python(
            {"name": "LibriSpeech", "split": "test.clean", "num_samples": 8}
        )
        assert isinstance(spec, LibriSpeechSpec)
        assert spec.split == "test.clean"
        assert spec.num_samples == 8

    def test_all_fields_round_trip(self):
        payload = {
            "name": "LibriSpeech",
            "split": "validation.clean",
            "num_samples": 32,
            "language": "English",
            "include_reference": True,
        }
        spec = _ADAPTER.validate_python(payload)
        assert spec.model_dump(exclude_unset=True) == payload

    def test_n_window_is_not_a_yaml_knob(self):
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python({"name": "LibriSpeech", "n_window": 50})

    def test_omitted_fields_are_not_recorded(self):
        # Fields carry no defaults of their own: an omitted field must stay
        # omitted so the dataset's own default applies.
        spec = _ADAPTER.validate_python({"name": "LibriSpeech"})
        assert spec.model_dump(exclude_unset=True) == {"name": "LibriSpeech"}

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python({"name": "LibriSpeech", "sample_rate": 16000})

    def test_bad_field_type_is_rejected(self):
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python({"name": "LibriSpeech", "num_samples": "many"})

    def test_usable_as_an_interleaved_source(self):
        spec = _ADAPTER.validate_python(
            {
                "name": "Interleaved",
                "source_datasets": [
                    {"name": "LibriSpeech", "split": "validation.clean"},
                    {"name": "Wikitext", "split": "train"},
                ],
            }
        )
        assert isinstance(spec.source_datasets[0], LibriSpeechSpec)


class TestFleursSpec:
    def test_name_is_in_the_dataset_vocabulary(self):
        assert "FLEURS" in dataset_names()
        assert spec_for_dataset("FLEURS") is FleursSpec
        assert dataset_name_of(FleursSpec) == "FLEURS"

    def test_all_fields_round_trip(self):
        payload = {
            "name": "FLEURS",
            "split": "test",
            "num_samples": 16,
            "language": "Korean",
            "include_reference": True,
        }
        spec = _ADAPTER.validate_python(payload)
        assert isinstance(spec, FleursSpec)
        assert spec.model_dump(exclude_unset=True) == payload

    def test_omitted_fields_are_not_recorded(self):
        payload = {"name": "FLEURS", "language": "Korean"}
        spec = _ADAPTER.validate_python(payload)
        assert spec.model_dump(exclude_unset=True) == payload

    def test_language_is_required(self):
        with pytest.raises(ValidationError, match="language"):
            _ADAPTER.validate_python({"name": "FLEURS"})

    def test_unknown_field_is_rejected(self):
        # One language per spec, for calibration and evaluation alike.
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python(
                {"name": "FLEURS", "language": "English", "languages": ["English"]}
            )

    def test_usable_as_an_interleaved_source(self):
        spec = _ADAPTER.validate_python(
            {
                "name": "Interleaved",
                "source_datasets": [
                    {"name": "FLEURS", "language": "Japanese"},
                    {"name": "LibriSpeech"},
                ],
            }
        )
        assert isinstance(spec.source_datasets[0], FleursSpec)


class TestASRDatasetSpec:
    """The selections a WER / CER entry may score on."""

    _ASR = TypeAdapter(ASRDatasetSpec)

    @pytest.mark.parametrize(
        "spec_cls, data",
        [
            (LibriSpeechSpec, {"name": "LibriSpeech", "split": "test.other"}),
            (FleursSpec, {"name": "FLEURS", "language": "Korean"}),
        ],
    )
    def test_accepts_both_asr_datasets(self, spec_cls, data):
        assert isinstance(self._ASR.validate_python(data), spec_cls)

    def test_rejects_a_non_asr_dataset(self):
        with pytest.raises(ValidationError, match="Wikitext"):
            self._ASR.validate_python({"name": "Wikitext"})

    @pytest.mark.parametrize(
        "base", [{"name": "LibriSpeech"}, {"name": "FLEURS", "language": "Korean"}]
    )
    @pytest.mark.parametrize("bad", [0, -1])
    def test_num_samples_must_be_positive(self, base, bad):
        with pytest.raises(ValidationError, match="num_samples"):
            self._ASR.validate_python({**base, "num_samples": bad})
