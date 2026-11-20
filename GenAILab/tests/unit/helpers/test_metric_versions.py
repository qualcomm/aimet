# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Pins each metric's scoring contract to its declared SCORING_VERSION.

If this fails, you changed scoring behavior without bumping SCORING_VERSION --
update it (and the fingerprint below) to match.
"""

import numpy as np
import torch

from GenAILab.bench.datasets import LazyERQADataset, LazyMMMUDataset
from GenAILab.bench.metrics import (
    CER,
    ERQA,
    WER,
    EvaluationMetric,
    MMMU,
    Where2Place,
    corpus_error_rate,
    normalize_transcript,
)


class _FakeTokenizer:
    """Minimal tokenizer stand-in with a controllable vocab for letter tokens."""

    def __init__(self, single_token_letters=("A", "B", "C", "D")):
        self._single_token_letters = set(single_token_letters)

    def __call__(self, text, add_special_tokens=False):
        stripped = text.strip()
        if text.startswith(" ") and stripped in self._single_token_letters:
            return {"input_ids": [hash(("space", stripped)) % 100000]}
        if stripped in self._single_token_letters:
            return {"input_ids": [hash(("bare", stripped)) % 100000]}
        return {"input_ids": [1, 2]}  # multi-token: forces bare-letter fallback


class TestEvaluationMetricDefaultVersion:
    def test_base_class_default_is_one(self):
        assert EvaluationMetric.SCORING_VERSION == 1


class TestMMMUScoringContractV2:
    def test_scoring_version_is_two(self):
        assert MMMU.SCORING_VERSION == 2

    def test_prefers_space_prefixed_letter_token(self):
        tokenizer = _FakeTokenizer(single_token_letters=("A", "B", "C", "D"))
        space_id = tokenizer(" A")["input_ids"][0]
        bare_id = tokenizer("A")["input_ids"][0]
        assert space_id != bare_id
        assert MMMU._token_id(tokenizer, "A") == space_id

    def test_falls_back_to_bare_letter_when_space_prefixed_is_multi_token(self):
        class _FallbackTokenizer:
            def __call__(self, text, add_special_tokens=False):
                if not text.startswith(" ") and text.strip() == "A":
                    return {"input_ids": [42]}
                return {"input_ids": [1, 2]}

        assert MMMU._token_id(_FallbackTokenizer(), "A") == 42

    def test_all_four_letters_resolve_to_distinct_ids(self):
        tokenizer = _FakeTokenizer(single_token_letters=("A", "B", "C", "D"))
        ids = [MMMU._token_id(tokenizer, letter) for letter in "ABCD"]
        assert len(set(ids)) == 4


class TestASRErrorRateScoringContractV2:
    """Golden fingerprint for the WER/CER scoring contract.

    If one of these fails you changed ASR scoring semantics (normalization,
    tokenization units, or aggregation): bump SCORING_VERSION on
    ``_ASRErrorRateBase`` and update the expectations here.
    """

    def test_scoring_version_is_two(self):
        assert WER.SCORING_VERSION == 2
        assert CER.SCORING_VERSION == 2

    def test_normalization_fingerprint(self):
        # casefolded, punctuation dropped, apostrophe kept, hyphen split,
        # whitespace collapsed.
        assert (
            normalize_transcript("  It’s WELL-known, isn't it?  ")
            == "it's well known isn't it"
        )

    def test_casefold_fingerprint(self):
        # v2: casefold, not lower() -- sharp s and final sigma fold.
        assert normalize_transcript("Straße ΛΌΓΟΣ") == "strasse λόγοσ"

    def test_units_fingerprint(self):
        assert (WER.UNIT, CER.UNIT) == ("word", "char")

    def test_aggregation_is_corpus_level(self):
        references = ["a", "b c d e f g h i j"]
        hypotheses = ["z", "b c d e f g h i j"]
        assert corpus_error_rate(references, hypotheses, unit="word") == 10.0

    def test_transcriptions_are_shared_between_wer_and_cer(self):
        assert WER.get_collection_name() == CER.get_collection_name()


class _FakeProcessor:
    """Records the messages passed to apply_chat_template for assertion."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.chat_template = None
        self.last_messages = None
        self.last_continue_final_message = None

    def apply_chat_template(self, messages, tokenize, continue_final_message, **kwargs):
        self.last_messages = messages
        self.last_continue_final_message = continue_final_message
        return "<rendered prompt>"

    def __call__(self, text, images, return_tensors):
        return {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }


class TestMMMUPromptContractV2:
    def _get_sample(self):
        raw_dataset = [
            {
                "question": "What is 2 + 2?",
                "options": ["3", "4", "5", "6"],
                "answer": "B",
            }
        ]
        processor = _FakeProcessor()
        dataset = LazyMMMUDataset(raw_dataset, processor, context_length=128)
        dataset[0]
        return processor

    def test_answer_prompt_is_in_an_assistant_turn(self):
        processor = self._get_sample()
        messages = processor.last_messages
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Answer:"
        assert messages[0]["role"] == "user"

    def test_continues_the_final_message(self):
        processor = self._get_sample()
        assert processor.last_continue_final_message is True


class TestERQAScoringContractV1:
    def test_scoring_version_is_one(self):
        assert ERQA.SCORING_VERSION == 1

    def test_reuses_mmmu_letter_token_resolution(self):
        # ERQA deliberately has no _token_id of its own -- confirms it keeps
        # calling MMMU's rather than a silently-diverged copy.
        assert not hasattr(ERQA, "_token_id")
        tokenizer = _FakeTokenizer(single_token_letters=("A", "B", "C", "D"))
        assert MMMU._token_id(tokenizer, "A") == tokenizer(" A")["input_ids"][0]

    def _get_sample(self):
        raw_dataset = [
            {
                "question": "Choices: A. one. B. two. C. three. D. four. "
                "Please answer directly with only the letter.",
                "images": ["<image-1>", "<image-2>"],
                "answer": "B",
            }
        ]
        processor = _FakeProcessor()
        dataset = LazyERQADataset(raw_dataset, processor, context_length=128)
        dataset[0]
        return processor

    def test_answer_prompt_is_in_an_assistant_turn(self):
        processor = self._get_sample()
        messages = processor.last_messages
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Answer:"
        assert messages[0]["role"] == "user"

    def test_continues_the_final_message(self):
        processor = self._get_sample()
        assert processor.last_continue_final_message is True

    def test_each_image_gets_a_separating_text_label(self):
        # Regression test: with zero separating tokens between consecutive
        # images, the fused sequence's visual tokens merge into one
        # contiguous run, which breaks the ONNX generator's per-image
        # boundary-based chunking/padding (confirmed via a real "Got: 98
        # Expected: 256" deepstack shape mismatch). Each image must be
        # preceded by its own text block so images stay in separate runs.
        processor = self._get_sample()
        content = processor.last_messages[0]["content"]
        assert content[0] == {"type": "text", "text": "Image 1:"}
        assert content[1] == {"type": "image"}
        assert content[2] == {"type": "text", "text": "Image 2:"}
        assert content[3] == {"type": "image"}
        assert content[4]["type"] == "text"
        assert content[4]["text"].startswith("Choices:")


class TestWhere2PlaceScoringContractV1:
    """Golden fingerprint for the pointing-metric scoring contract.

    ``_parse_points``/``_any_point_in_mask`` have no existing analog elsewhere
    in the codebase (unlike MMMU/ERQA's choice-logit reuse), so this pins their
    exact behavior rather than just a version number.
    """

    def test_scoring_version_is_one(self):
        assert Where2Place.SCORING_VERSION == 1

    def test_parses_a_single_point(self):
        assert Where2Place._parse_points("The point is at (0.42, 0.71).") == [
            (0.42, 0.71)
        ]

    def test_parses_a_list_of_points_in_the_requested_format(self):
        text = "Here are some points: [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]"
        assert Where2Place._parse_points(text) == [
            (0.1, 0.2),
            (0.3, 0.4),
            (0.5, 0.6),
        ]

    def test_ignores_non_numeric_parenthesized_text(self):
        # The prompt's own instruction template, e.g. "(x1, y1), (x2, y2)",
        # must not be mistaken for an answer if it leaks into the scanned text.
        text = "format: [(x1, y1), (x2, y2)] then answer: (0.5, 0.5)"
        assert Where2Place._parse_points(text) == [(0.5, 0.5)]

    def test_clamps_out_of_range_coordinates_rather_than_discarding(self):
        assert Where2Place._parse_points("(1.5, -0.2)") == [(1.0, 0.0)]

    def test_no_point_present_returns_empty_list(self):
        assert Where2Place._parse_points("I cannot locate any free space.") == []

    def test_any_point_in_mask_true_when_one_of_several_hits(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        points = [(0.0, 0.0), (0.55, 0.55)]  # second point -> pixel (5, 5)
        assert Where2Place._any_point_in_mask(points, mask) is True

    def test_any_point_in_mask_false_when_none_hit(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        points = [(0.0, 0.0), (0.9, 0.9)]
        assert Where2Place._any_point_in_mask(points, mask) is False
