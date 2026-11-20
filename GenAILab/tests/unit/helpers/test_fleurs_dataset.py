# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the FLEURS ASR dataset and its language-name lookup.

The raw HuggingFace dataset and the ASR processor are both mocked: nothing here
downloads audio or loads a model.
"""

import pytest

from GenAILab.bench.datasets import (
    FLEURS,
    FLEURS_LANGUAGE_CONFIGS,
    Interleaved,
    LazyASRDataset,
    MultimodalDataset,
    check_asr_language,
    fleurs_config_for,
    fleurs_language_code,
)
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.tests.unit.helpers.test_librispeech_dataset import (
    _FakeASRProcessor,
    _raw_dataset,
)


def _fleurs_rows(num_items=2):
    rows = _raw_dataset(num_items)
    for i, row in enumerate(rows):
        row["raw_transcription"] = f"Hello, World {i}."
        row["transcription"] = f"hello world {i}"
        del row["text"]
    return rows


class _FakeHF:
    def __init__(self, items, captured=None):
        self.items = items
        self.captured = captured if captured is not None else {}

    def cast_column(self, name, feature):
        self.captured["cast"] = (name, feature)
        return self

    def select(self, indices):
        return _FakeHF([self.items[i] for i in indices], self.captured)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class TestLanguageConfigs:
    def test_vendored_table_is_one_to_one_fleurs_names_only(self):
        # No model-card aliases: a model's spelling is not FLEURS vocabulary.
        assert len(FLEURS_LANGUAGE_CONFIGS) == 102
        assert len(set(FLEURS_LANGUAGE_CONFIGS.values())) == 102
        assert "Chinese" not in FLEURS_LANGUAGE_CONFIGS
        assert "Cantonese" not in FLEURS_LANGUAGE_CONFIGS

    def test_language_codes_are_unique(self):
        codes = [fleurs_language_code(c) for c in FLEURS_LANGUAGE_CONFIGS.values()]
        assert len(set(codes)) == 102

    @pytest.mark.parametrize(
        "config, code",
        [
            ("ko_kr", "ko"),
            ("en_us", "en"),
            ("fil_ph", "fil"),
            ("yue_hant_hk", "yue"),
            ("cmn_hans_cn", "zh"),  # ISO 639-3 Mandarin -> ISO 639-1
        ],
    )
    def test_language_code(self, config, code):
        assert fleurs_language_code(config) == code

    def test_every_qwen3_asr_language_is_reachable_by_code(self):
        from GenAILab.qai_hub_lm.models.qwen3_asr import Qwen3ASRGenerator

        codes = {fleurs_language_code(c) for c in FLEURS_LANGUAGE_CONFIGS.values()}
        assert set(Qwen3ASRGenerator.SUPPORTED_LANGUAGES) <= codes

    def test_model_card_name_suggests_the_fleurs_name(self):
        with pytest.raises(ValueError, match="did you mean 'Mandarin Chinese'"):
            fleurs_config_for("Chinese")

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("English", ("English", "en_us")),
            ("english", ("English", "en_us")),
            ("KOREAN", ("Korean", "ko_kr")),
            ("Mandarin Chinese", ("Mandarin Chinese", "cmn_hans_cn")),
            ("cantonese chinese", ("Cantonese Chinese", "yue_hant_hk")),
            ("Japanese", ("Japanese", "ja_jp")),
        ],
    )
    def test_lookup_is_case_insensitive(self, name, expected):
        assert fleurs_config_for(name) == expected

    def test_unknown_language_suggests_a_close_match(self):
        with pytest.raises(ValueError, match="did you mean 'Portuguese'"):
            fleurs_config_for("Portugese")

    def test_unknown_language_without_a_close_match(self):
        with pytest.raises(ValueError, match="Unknown language 'Klingon'$"):
            fleurs_config_for("Klingon")


class TestFleursWrapper:
    def test_is_a_multimodal_dataset_and_is_registered(self):
        assert issubclass(FLEURS, MultimodalDataset)
        assert YAMLConfigParser.get_dataset("FLEURS") is FLEURS

    def test_source_is_pinned(self):
        assert FLEURS.REPO_ID == "google/fleurs"
        assert len(FLEURS.REVISION) == 40
        assert all(c in "0123456789abcdef" for c in FLEURS.REVISION)

    def test_load_dataset_scopes_to_one_split(self, monkeypatch):
        captured = {}

        def fake_load_dataset(repo, data_files=None, split=None, revision=None):
            captured.update(
                repo=repo, data_files=data_files, split=split, revision=revision
            )
            return _FakeHF(_fleurs_rows(5), captured)

        monkeypatch.setattr("GenAILab.bench.datasets.load_dataset", fake_load_dataset)
        raw = FLEURS.load_dataset("test", language="korean")
        assert captured["repo"] == "google/fleurs"
        assert captured["data_files"] == {"test": "parquet-data/ko_kr/test-*.parquet"}
        assert captured["split"] == "test"
        assert captured["revision"] == FLEURS.REVISION
        assert captured["cast"][0] == "audio"
        assert captured["cast"][1].decode is False
        assert len(raw) == 5

    def test_unknown_language_fails_before_download(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("must not download")

        monkeypatch.setattr("GenAILab.bench.datasets.load_dataset", boom)
        with pytest.raises(ValueError, match="did you mean"):
            FLEURS.load_dataset("test", language="Japanse")

    def test_num_samples_takes_the_head_and_is_clamped(self, monkeypatch):
        monkeypatch.setattr(
            "GenAILab.bench.datasets.load_dataset",
            lambda *a, **k: _FakeHF(_fleurs_rows(5)),
        )
        assert len(FLEURS.load_dataset("test", num_samples=3)) == 3
        assert len(FLEURS.load_dataset("test", num_samples=99)) == 5

    def test_load_encoded_dataset_defaults_to_calibration_safe_items(self, monkeypatch):
        monkeypatch.setattr(
            FLEURS, "_load_config", staticmethod(lambda *a, **k: _fleurs_rows(2))
        )
        dataset = FLEURS.load_encoded_dataset(_FakeASRProcessor(), 512)
        assert isinstance(dataset, LazyASRDataset)
        assert "reference" not in dataset[0]

    def test_reference_is_the_raw_transcription(self, monkeypatch):
        monkeypatch.setattr(
            FLEURS, "_load_config", staticmethod(lambda *a, **k: _fleurs_rows(2))
        )
        dataset = FLEURS.load_encoded_dataset(
            _FakeASRProcessor(), 512, include_reference=True
        )
        assert dataset[0]["reference"] == "Hello, World 0."
        assert dataset[1]["reference"] == "Hello, World 1."

    @pytest.mark.parametrize(
        "language, config, code",
        [
            ("japanese", "ja_jp", "ja"),
            ("Mandarin Chinese", "cmn_hans_cn", "zh"),
        ],
    )
    def test_the_model_is_given_the_language_code(
        self, monkeypatch, language, config, code
    ):
        # The processor maps the code to its own prompt name ("zh" -> "Chinese").
        loaded = []

        def fake_load_config(split, num_samples, config):
            loaded.append(config)
            return _fleurs_rows(1)

        monkeypatch.setattr(FLEURS, "_load_config", staticmethod(fake_load_config))
        processor = _FakeASRProcessor()
        dataset = FLEURS.load_encoded_dataset(processor, 512, language=language)
        dataset[0]
        assert loaded == [config]
        assert processor.calls[0]["language"] == code


class _QwenLikeModel:
    SUPPORTED_LANGUAGES = {"en": "English", "zh": "Chinese", "ko": "Korean"}


class TestCheckAsrLanguage:
    @pytest.mark.parametrize("language", ["zh", "ZH", "Chinese", "cHiNeSe"])
    def test_a_code_or_name_in_any_case_is_accepted(self, language):
        check_asr_language(_QwenLikeModel(), language)

    def test_an_unknown_language_is_rejected(self):
        with pytest.raises(ValueError, match="Swahili is not supported by _QwenLike"):
            check_asr_language(_QwenLikeModel(), "Swahili")


class TestLanguageCheckBeforeDownload:
    @pytest.fixture(autouse=True)
    def no_download(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("must not download")

        monkeypatch.setattr(FLEURS, "_load_config", staticmethod(boom))

    def test_an_unsupported_language_fails_before_download(self):
        with pytest.raises(ValueError, match="sw is not supported by _QwenLikeModel"):
            FLEURS.load_encoded_dataset(
                _FakeASRProcessor(), 512, language="Swahili", model=_QwenLikeModel()
            )

    def test_the_error_lists_the_supported_names(self):
        with pytest.raises(ValueError, match="Supported: Chinese, English, Korean"):
            FLEURS.load_encoded_dataset(
                _FakeASRProcessor(), 512, language="Swahili", model=_QwenLikeModel()
            )

    @pytest.mark.parametrize("model", [None, object()])
    def test_no_check_without_a_declared_vocabulary(self, monkeypatch, model):
        monkeypatch.setattr(
            FLEURS, "_load_config", staticmethod(lambda *a, **k: _fleurs_rows(1))
        )
        dataset = FLEURS.load_encoded_dataset(
            _FakeASRProcessor(), 512, language="Swahili", model=model
        )
        assert dataset.language == "sw"

    def test_a_supported_language_loads(self, monkeypatch):
        monkeypatch.setattr(
            FLEURS, "_load_config", staticmethod(lambda *a, **k: _fleurs_rows(1))
        )
        dataset = FLEURS.load_encoded_dataset(
            _FakeASRProcessor(),
            512,
            language="Mandarin Chinese",
            model=_QwenLikeModel(),
        )
        assert dataset.language == "zh"


class TestInterleavedFleursCalibration:
    """A multilingual calibration mix: one FLEURS source per language."""

    @pytest.fixture(autouse=True)
    def _fake_rows(self, monkeypatch):
        def fake_load_config(split, num_samples, config):
            return _fleurs_rows(num_samples)

        monkeypatch.setattr(FLEURS, "_load_config", staticmethod(fake_load_config))

    def _load(self, **kwargs):
        return Interleaved.load_encoded_dataset(
            _FakeASRProcessor(),
            512,
            source_datasets=[
                {"name": "FLEURS", "language": "English", "num_samples": 3},
                {"name": "FLEURS", "language": "Korean", "num_samples": 1},
            ],
            **kwargs,
        )

    def test_per_source_num_samples_sets_the_mix(self):
        dataset = self._load()
        english, korean = dataset.datasets
        assert (english.language, len(english)) == ("en", 3)
        assert (korean.language, len(korean)) == ("ko", 1)
        assert len(dataset) == 4

    def test_static_audio_kwargs_reach_every_source(self):
        # Without audio_frames + model the sub-datasets skip the static-shape
        # padding the ONNX audio encoder needs.
        model = object()
        dataset = self._load(audio_frames=2400, model=model)
        for sub in dataset.datasets:
            assert sub.audio_frames == 2400
            assert sub.model is model

    def test_unaccepted_parent_kwargs_are_not_forwarded(self):
        # FLEURS takes no image_size; forwarding it would raise TypeError.
        dataset = self._load(image_size=(224, 224))
        assert len(dataset) == 4
