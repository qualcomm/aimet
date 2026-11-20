# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ASR error-rate metrics (WER / CER).

Error rates are checked against hand-computed values; the generative path runs
against a fake model/processor, so nothing here loads weights or audio.
"""

import inspect
import types

import pytest
import torch

from GenAILab.bench.datasets import LibriSpeech
from GenAILab.bench.metrics import (
    CER,
    _ASRErrorRateBase,
    WER,
    char_tokens,
    corpus_error_rate,
    levenshtein_distance,
    normalize_transcript,
    word_tokens,
)
from GenAILab.bench.yaml_config_parser import YAMLConfigParser


class TestNormalizeTranscript:
    """The normalization is part of the scoring contract (see SCORING_VERSION)."""

    def test_lowercases_and_strips_punctuation(self):
        assert normalize_transcript("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_transcript("  a\t\tb\n c ") == "a b c"

    def test_keeps_apostrophes_and_folds_variants(self):
        assert normalize_transcript("DON’T STOP") == "don't stop"
        assert normalize_transcript("don't stop") == "don't stop"

    def test_hyphens_and_slashes_become_word_boundaries(self):
        assert normalize_transcript("well-known") == "well known"
        assert normalize_transcript("and/or") == "and or"
        assert normalize_transcript("a—b") == "a b"

    def test_symbols_are_dropped(self):
        assert normalize_transcript("50% $5") == "50 5"

    def test_letters_outside_ascii_survive(self):
        assert normalize_transcript("Café") == "café"

    def test_empty_and_whitespace_only(self):
        assert normalize_transcript("") == ""
        assert normalize_transcript("   ") == ""

    def test_is_idempotent(self):
        once = normalize_transcript("It's WELL-known, isn't it?")
        assert normalize_transcript(once) == once

    def test_librispeech_upper_case_reference_matches_lower_case_hypothesis(self):
        assert normalize_transcript("HE WENT HOME") == normalize_transcript(
            "he went home."
        )


class TestTokenizers:
    def test_word_tokens(self):
        assert word_tokens("Hi,  there!") == ["hi", "there"]

    def test_word_tokens_of_empty_text(self):
        assert word_tokens("  ") == []

    def test_char_tokens_include_the_separating_space(self):
        assert char_tokens("A b") == ["a", " ", "b"]

    def test_char_tokens_of_empty_text(self):
        assert char_tokens("") == []


class TestLevenshteinDistance:
    def test_identical_sequences(self):
        assert levenshtein_distance(["a", "b"], ["a", "b"]) == 0

    def test_both_empty(self):
        assert levenshtein_distance([], []) == 0

    def test_single_substitution(self):
        assert levenshtein_distance(["a", "b", "c"], ["a", "x", "c"]) == 1

    def test_empty_hypothesis_costs_one_deletion_per_reference_token(self):
        assert levenshtein_distance(["a", "b", "c"], []) == 3

    def test_empty_reference_costs_one_insertion_per_hypothesis_token(self):
        assert levenshtein_distance([], ["a", "b"]) == 2

    def test_length_mismatch_mixes_edits(self):
        # "the quick brown fox" -> "the quick red": 1 substitution + 1 deletion.
        reference = ["the", "quick", "brown", "fox"]
        hypothesis = ["the", "quick", "red"]
        assert levenshtein_distance(reference, hypothesis) == 2

    def test_classic_kitten_sitting(self):
        assert levenshtein_distance(list("kitten"), list("sitting")) == 3

    def test_is_symmetric(self):
        a, b = list("kitten"), list("sitting")
        assert levenshtein_distance(a, b) == levenshtein_distance(b, a)


class TestCorpusErrorRate:
    def test_perfect_transcription_scores_zero(self):
        assert corpus_error_rate(["The cat sat."], ["the cat sat"]) == 0.0

    def test_hand_computed_wer(self):
        # 2 edits over a 4-word reference.
        assert corpus_error_rate(["the quick brown fox"], ["the quick red"]) == 50.0

    def test_empty_hypothesis_is_all_deletions(self):
        assert corpus_error_rate(["one two three"], [""]) == 100.0

    def test_insertions_can_exceed_one_hundred_percent(self):
        # 2 insertions over a 1-word reference.
        assert corpus_error_rate(["a"], ["a b c"]) == 200.0

    def test_aggregation_is_corpus_level_not_the_mean_of_utterances(self):
        references = ["a", "b c d e f g h i j"]
        hypotheses = ["z", "b c d e f g h i j"]
        # Corpus: 1 edit / 10 reference words. Mean of per-utterance rates
        # would be (100 + 0) / 2 = 50.
        assert corpus_error_rate(references, hypotheses) == 10.0

    def test_empty_reference_set_with_empty_hypotheses_is_zero(self):
        assert corpus_error_rate([""], [""]) == 0.0

    def test_empty_reference_set_with_output_is_one_hundred(self):
        assert corpus_error_rate([""], ["hello"]) == 100.0

    def test_character_unit(self):
        # "cat" -> "car": 1 edit over 3 characters.
        assert corpus_error_rate(["cat"], ["car"], unit="char") == pytest.approx(
            100.0 / 3
        )

    def test_character_unit_counts_spaces(self):
        # One missing space over 5 reference characters.
        assert corpus_error_rate(["a b c"], ["a bc"], unit="char") == pytest.approx(
            100.0 / 5
        )

    def test_mismatched_list_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="1:1"):
            corpus_error_rate(["a", "b"], ["a"])

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(KeyError):
            corpus_error_rate(["a"], ["a"], unit="phoneme")


# ---------------------------------------------------------------------------
# Generative path
# ---------------------------------------------------------------------------

_VOCAB = {0: "<pad>", 1: "hello", 2: "world", 3: "goodbye"}


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def decode(self, token_ids, skip_special_tokens=False):
        return " ".join(_VOCAB[int(i)] for i in token_ids)


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class _FakeModel:
    """Echoes a scripted continuation after the prompt tokens."""

    device = "cpu"

    def __init__(self, continuations):
        self.continuations = list(continuations)
        self.calls = []
        self.config = types.SimpleNamespace(eos_token_id=0)
        self.generation_config = types.SimpleNamespace(eos_token_id=0)

    def generate(self, generation_config=None, **inputs):
        self.calls.append(inputs)
        assert "input_features" in inputs, "audio features must reach generate()"
        assert "input_features_mask" in inputs
        prompt = inputs["input_ids"]
        index = (len(self.calls) - 1) % len(self.continuations)
        continuation = torch.tensor([self.continuations[index]], dtype=torch.long)
        return torch.cat([prompt, continuation], dim=-1)


def _fake_items():
    return [
        {
            "input_ids": torch.zeros((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "input_features": torch.zeros((1, 128, 100)),
            "input_features_mask": torch.ones((1, 100), dtype=torch.long),
            "reference": "HELLO WORLD",
        },
        {
            "input_ids": torch.zeros((1, 3), dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "input_features": torch.zeros((1, 128, 100)),
            "input_features_mask": torch.ones((1, 100), dtype=torch.long),
            "reference": "HELLO WORLD",
        },
    ]


@pytest.fixture
def scripted(monkeypatch):
    """One perfect utterance and one with a single substituted word."""
    monkeypatch.setattr(
        WER, "get_dataset", classmethod(lambda cls, *a, **k: _fake_items())
    )
    monkeypatch.setattr(
        CER, "get_dataset", classmethod(lambda cls, *a, **k: _fake_items())
    )
    return _FakeModel([[1, 2], [1, 3]])


class _FakeEvalContext:
    """Minimal EvaluationContext stand-in: caches by collection name."""

    def __init__(self):
        self.cache = {}

    def get_or_compute_quant(self, name, compute_fn):
        if name not in self.cache:
            self.cache[name] = compute_fn()
        return self.cache[name]


class TestRegistrationAndContract:
    def test_both_metrics_are_registered(self):
        assert YAMLConfigParser.get_metric("WER") is WER
        assert YAMLConfigParser.get_metric("CER") is CER

    def test_units(self):
        assert WER.UNIT == "word"
        assert CER.UNIT == "char"

    def test_scoring_version_is_two(self):
        assert WER.SCORING_VERSION == 2
        assert CER.SCORING_VERSION == 2

    def test_transcriptions_are_cached_under_one_shared_key(self):
        # WER and CER must reuse a single decode pass.
        assert WER.get_collection_name() == CER.get_collection_name()


class TestDecodeTranscript:
    def test_strips_everything_before_the_asr_text_marker(self):
        class _T:
            def decode(self, ids, skip_special_tokens=False):
                return "language English<asr_text>he went home"

        assert WER.decode_transcript(_T(), [1]) == "he went home"

    def test_strips_remaining_special_markers(self):
        class _T:
            def decode(self, ids, skip_special_tokens=False):
                return "he went home<|im_end|>"

        assert WER.decode_transcript(_T(), [1]) == "he went home"

    def test_passes_plain_text_through(self):
        assert WER.decode_transcript(_FakeTokenizer(), [1, 2]) == "hello world"


class TestGenerativeEvaluate:
    def test_wer_over_two_utterances(self, scripted):
        # 1 substituted word over 4 reference words.
        result = WER.evaluate(scripted, _FakeProcessor(), 512, eval_ctx=None)
        assert result.result == 25.0

    def test_cer_differs_from_wer_and_matches_the_helper(self, scripted):
        result = CER.evaluate(scripted, _FakeProcessor(), 512, eval_ctx=None).result
        expected = corpus_error_rate(
            ["HELLO WORLD", "HELLO WORLD"],
            ["hello world", "hello goodbye"],
            unit="char",
        )
        assert result == pytest.approx(expected)
        assert result != 25.0

    def test_reference_is_not_forwarded_to_the_model(self, scripted):
        WER.evaluate(scripted, _FakeProcessor(), 512, eval_ctx=None)
        assert all("reference" not in call for call in scripted.calls)

    def test_without_eval_context_it_warns(self, scripted):
        with pytest.warns(UserWarning, match="EvaluationContext"):
            WER.evaluate(scripted, _FakeProcessor(), 512, eval_ctx=None)

    def test_eval_context_shares_the_decode_pass_between_wer_and_cer(self, scripted):
        eval_ctx = _FakeEvalContext()
        processor = _FakeProcessor()
        WER.evaluate(scripted, processor, 512, eval_ctx=eval_ctx)
        num_after_wer = len(scripted.calls)
        CER.evaluate(scripted, processor, 512, eval_ctx=eval_ctx)
        assert num_after_wer == 2
        assert len(scripted.calls) == 2  # CER reused the cached transcriptions

    def test_transcribe_all_returns_aligned_references_and_hypotheses(self, scripted):
        data = WER.transcribe_all(scripted, _FakeProcessor(), 512)
        assert data["references"] == ["HELLO WORLD", "HELLO WORLD"]
        assert data["hypotheses"] == ["hello world", "hello goodbye"]


class TestGetDatasetWiring:
    def test_asks_librispeech_for_references_and_bounded_samples(self, monkeypatch):
        captured = {}

        def fake_load_encoded_dataset(processor, context_length, **kwargs):
            captured.update(context_length=context_length, **kwargs)
            return []

        monkeypatch.setattr(
            "GenAILab.bench.datasets.LibriSpeech.load_encoded_dataset",
            staticmethod(fake_load_encoded_dataset),
        )
        WER.get_dataset(_FakeProcessor(), 512)
        # Split and language are left to LibriSpeech's own defaults.
        assert "split" not in captured and "language" not in captured
        assert captured["num_samples"] == WER.DEFAULT_NUM_SAMPLES
        assert captured["include_reference"] is True

    def test_librispeech_defaults_are_test_clean_in_english(self):
        params = inspect.signature(LibriSpeech.load_encoded_dataset).parameters
        assert params["split"].default == "test.clean"
        assert params["language"].default == "English"

    def test_overrides_are_honoured(self, monkeypatch):
        captured = {}

        def fake_load_encoded_dataset(processor, context_length, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(
            "GenAILab.bench.datasets.LibriSpeech.load_encoded_dataset",
            staticmethod(fake_load_encoded_dataset),
        )
        spec = WER.dataset_spec(
            split="validation.clean", num_samples=4, language="German"
        )
        WER.get_dataset(_FakeProcessor(), 512, spec)
        assert captured["split"] == "validation.clean"
        assert captured["num_samples"] == 4
        assert captured["language"] == "German"


class TestCasefoldNormalization:
    """Transcripts are casefolded, not lowercased."""

    def test_casefold_folds_sharp_s(self):
        assert normalize_transcript("Straße") == "strasse"
        assert normalize_transcript("STRASSE") == "strasse"

    def test_casefold_folds_greek_final_sigma(self):
        assert normalize_transcript("λόγος") == normalize_transcript("ΛΌΓΟΣ")

    def test_casefold_recomposes_to_one_codepoint(self):
        # casefold("ΐ") is iota + two combining marks; the second NFKC folds it
        # back to U+0390 so CER counts one character.
        assert normalize_transcript("ΐ") == "\u0390"
        assert char_tokens("ΐ") == ["\u0390"]

    def test_casefold_is_idempotent(self):
        once = normalize_transcript("Größe ΐ İstanbul")
        assert normalize_transcript(once) == once

    @pytest.mark.parametrize(
        "text",
        [
            "HE WENT HOME, DIDN'T HE?",
            "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES",
            "well-known and/or 50% café",
        ],
    )
    def test_casefold_matches_lower_on_ascii_and_latin1(self, text):
        # LibriSpeech-style text normalizes as it did under lower() (v1).
        assert normalize_transcript(text) == normalize_transcript(text.lower())
        assert text.casefold() == text.lower()

    def test_corpus_error_rate_casefolds(self):
        refs, hyps = ["Straße"], ["strasse"]
        assert corpus_error_rate(refs, hyps, unit="word") == 0.0
        assert corpus_error_rate(refs, hyps, unit="char") == 0.0


# ---------------------------------------------------------------------------
# Dataset selection: WER / CER with ``dataset: FLEURS``
# ---------------------------------------------------------------------------


class _MultilingualModel(_FakeModel):
    # Qwen3ASRGenerator-shaped: {code: name}, read by the dataset's check.
    SUPPORTED_LANGUAGES = {
        "en": "English",
        "ko": "Korean",
        "ja": "Japanese",
        "zh": "Chinese",
        "de": "German",
    }


def _stub_transcribe(monkeypatch, data_by_language):
    """Replace the decode loop; record the spec each pass was decoded with."""
    calls = []

    def fake(cls, model, processor, context_length, spec=None, audio_frames=None):
        calls.append({"spec": spec, "audio_frames": audio_frames})
        return data_by_language[spec.language or "English"]

    monkeypatch.setattr(_ASRErrorRateBase, "transcribe_all", classmethod(fake))
    return calls


def _run(metric, model=None, eval_ctx=None, **kwargs):
    return metric.evaluate(
        model or _MultilingualModel([[1]]),
        _FakeProcessor(),
        512,
        eval_ctx=_FakeEvalContext() if eval_ctx is None else eval_ctx,
        **kwargs,
    )


class TestFleursIsNotASeparateMetric:
    def test_only_wer_and_cer_are_registered(self):
        with pytest.raises(LookupError):
            YAMLConfigParser.get_metric("FLEURS")


class TestDatasetSelectionValidation:
    """Every bad selection must raise before a single sample is decoded."""

    @pytest.fixture(autouse=True)
    def no_decode(self, monkeypatch):
        def boom(cls, *a, **k):
            raise AssertionError("decode must not run")

        monkeypatch.setattr(_ASRErrorRateBase, "transcribe_all", classmethod(boom))

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError, match="languagez"):
            _run(WER, dataset="FLEURS", languagez="English")

    def test_a_language_list_is_rejected(self):
        # One entry per language: a list would fan one entry out into many rows.
        with pytest.raises(ValueError, match="languages"):
            _run(WER, dataset="FLEURS", languages=["English"])

    def test_a_non_asr_dataset_is_rejected(self):
        with pytest.raises(ValueError, match="Wikitext"):
            _run(WER, dataset="Wikitext")

    def test_wrong_type_is_rejected(self):
        with pytest.raises(ValueError):
            _run(WER, dataset="FLEURS", language="English", num_samples="many")

    @pytest.mark.parametrize("bad", [0, -4])
    def test_non_positive_num_samples_is_rejected(self, bad):
        with pytest.raises(ValueError):
            _run(WER, dataset="FLEURS", language="English", num_samples=bad)

    def test_include_reference_is_not_a_metric_option(self):
        with pytest.raises(ValueError, match="include_reference"):
            _run(WER, dataset="FLEURS", language="English", include_reference=False)


class TestFleursScoring:
    DATA = {
        "English": {"references": ["Hello, World!"], "hypotheses": ["hello there"]},
        "Korean": {"references": ["안녕 하세요"], "hypotheses": ["안녕 하세요"]},
        "Japanese": {"references": ["こんにちは世界"], "hypotheses": ["こんにちは"]},
        "German": {"references": ["Straße"], "hypotheses": ["strasse"]},
    }

    @pytest.fixture
    def calls(self, monkeypatch):
        return _stub_transcribe(monkeypatch, self.DATA)

    def test_default_dataset_is_librispeech(self, calls):
        _run(WER)
        assert calls[0]["spec"].name == "LibriSpeech"

    def test_fleurs_defaults_are_the_datasets_own(self, calls):
        # Only the sample bound is the metric's; split is left to
        # FLEURS.load_encoded_dataset (test).
        _run(WER, dataset="FLEURS", language="English")
        spec = calls[0]["spec"]
        assert WER.dataset_kwargs(spec) == {
            "language": "English",
            "num_samples": WER.DEFAULT_NUM_SAMPLES,
        }

    def test_fleurs_needs_a_language(self, calls):
        with pytest.raises(ValueError, match="language"):
            _run(WER, dataset="FLEURS")
        assert not calls

    def test_the_metric_name_fixes_the_unit(self, calls):
        wer = _run(WER, dataset="FLEURS", language="Japanese")
        cer = _run(CER, dataset="FLEURS", language="Japanese")
        # One "word" in the reference, a different one in the hypothesis.
        assert wer.result == pytest.approx(100.0)
        # 7 reference chars, 5 hypothesis chars: 2 deletions.
        assert cer.result == pytest.approx(100 * 2 / 7)

    def test_english_wer(self, calls):
        # "hello world" vs "hello there": 1 substitution over 2 words.
        out = _run(WER, dataset="FLEURS", language="English")
        assert out.result == pytest.approx(50.0)

    def test_scoring_casefolds(self, calls):
        # Straße vs strasse is an exact match only under casefold.
        assert _run(WER, dataset="FLEURS", language="German").result == 0.0

    def test_details(self, calls):
        out = _run(
            WER, dataset="FLEURS", language="Korean", split="validation", num_samples=8
        )
        assert out.details == {
            "name": "FLEURS",
            "language": "Korean",
            "split": "validation",
            "unit": "word",
            "num_samples": 1,  # samples actually scored, not requested
        }

    def test_audio_frames_reach_the_decode_and_image_size_is_ignored(self, calls):
        _run(
            WER,
            dataset="FLEURS",
            language="English",
            audio_frames=3000,
            image_size=(1, 1),
        )
        assert calls[0]["audio_frames"] == 3000


class TestTranscriptionCaching:
    DATA = TestFleursScoring.DATA

    def test_wer_and_cer_share_one_decode_per_selection(self, monkeypatch):
        calls = _stub_transcribe(monkeypatch, self.DATA)
        ctx = _FakeEvalContext()
        _run(WER, eval_ctx=ctx, dataset="FLEURS", language="Korean")
        _run(CER, eval_ctx=ctx, dataset="FLEURS", language="Korean")
        assert len(calls) == 1

    def test_each_selection_is_its_own_cache_entry(self, monkeypatch):
        calls = _stub_transcribe(monkeypatch, self.DATA)
        ctx = _FakeEvalContext()
        for kw in (
            {},
            {"dataset": "FLEURS", "language": "English"},
            {"dataset": "FLEURS", "language": "Korean"},
            {"dataset": "FLEURS", "language": "English", "split": "validation"},
            {"dataset": "FLEURS", "language": "English", "num_samples": 8},
        ):
            _run(WER, eval_ctx=ctx, **kw)
        assert len(calls) == len(ctx.cache) == 5

    def test_an_explicit_default_hits_the_same_entry(self):
        default = WER.dataset_spec("FLEURS", language="Korean")
        explicit = WER.dataset_spec(
            "FLEURS", language="Korean", num_samples=WER.DEFAULT_NUM_SAMPLES
        )
        assert WER.get_collection_name(default) == WER.get_collection_name(explicit)

    def test_runs_without_a_context_but_warns(self, monkeypatch):
        _stub_transcribe(monkeypatch, self.DATA)
        with pytest.warns(UserWarning, match="No EvaluationContext"):
            WER.evaluate(_MultilingualModel([[1]]), _FakeProcessor(), 512)


class TestFleursEndToEndThroughTranscribeAll:
    """Real decode loop (fake model), so the loader kwargs and the reference pop
    are exercised, not just the scoring."""

    def test_loads_through_the_fleurs_dataset_and_scores(self, monkeypatch):
        seen = []

        def fake_load_encoded_dataset(processor, context_length, **kwargs):
            seen.append(kwargs)
            return _fake_items()

        monkeypatch.setattr(
            "GenAILab.bench.datasets.FLEURS.load_encoded_dataset",
            staticmethod(fake_load_encoded_dataset),
        )
        model = _MultilingualModel([[1, 2], [1, 3]])
        out = _run(WER, model=model, dataset="FLEURS", language="Korean", num_samples=2)
        assert seen == [
            {
                "language": "Korean",
                "num_samples": 2,
                "audio_frames": None,
                "model": model,
                "include_reference": True,
            }
        ]
        # Same scripted decode as the WER tests: one perfect, one substituted.
        assert out.result == pytest.approx(25.0)

    def test_an_unsupported_language_fails_before_download_or_decode(self, monkeypatch):
        # The language check lives with the dataset; the metric reaches it
        # through get_dataset(model=...), before FLEURS downloads anything.
        def boom(*a, **k):
            raise AssertionError("must not download")

        monkeypatch.setattr(
            "GenAILab.bench.datasets.FLEURS._load_config", staticmethod(boom)
        )
        with pytest.raises(ValueError, match="sw is not supported"):
            _run(WER, dataset="FLEURS", language="Swahili")
