# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Grace metric and its grader.

Grace scores only compare across runs while the prompt set, the rubric and the
aggregation stay identical, so these tests pin the parts a well-meaning refactor
would otherwise drift: the prompt set's shape, the rating/rationale/summary
parsers, and the rule that an unrated item scores 0 and stays in the denominator.
"""

import contextlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from GenAILab.qai_hub_lm.scoring.grace.grace import (
    GRACE_TASK_NAME,
    GRACE_VERSION,
    EvalPrompt,
    default_categories_by_idx,
    default_categories_by_prompt,
    load_default_eval_prompts,
    load_eval_prompts,
    select_balanced,
)
from GenAILab.qai_hub_lm.scoring.grace.report import build_summary
from GenAILab.qai_hub_lm.scoring.grace.grader import (
    DEFAULT_PROMPT_TEMPLATE,
    GRADER_FAILURE_NOTE,
    MAX_POINTS,
    MAX_RATIONALE_CHARS,
    MAX_SUMMARY_ITEM_CHARS,
    GradeResult,
    parse_forced_rating,
    parse_rating,
    parse_rationale,
    parse_summary_items,
    summarize,
)


class TestPromptSet:
    def test_names(self):
        assert GRACE_VERSION == 2
        assert GRACE_TASK_NAME == "grace2"

    def test_shape_is_10x10(self):
        prompts = load_eval_prompts()
        assert len(prompts) == 100
        counts = Counter(p.category for p in prompts)
        assert len(counts) == 10
        assert set(counts.values()) == {10}, counts

    def test_idx_is_contiguous_and_ordered(self):
        # idx is the join key between device responses and grader summaries.
        assert [p.idx for p in load_eval_prompts()] == list(range(100))

    def test_prompts_are_unique_and_nonempty(self):
        prompts = load_eval_prompts()
        assert all(p.prompt.strip() for p in prompts)
        assert len({p.prompt for p in prompts}) == 100

    def test_loader_rejects_non_contiguous_idx(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"idx": 0, "category": "a", "prompt": "x"}\n'
            '{"idx": 7, "category": "a", "prompt": "y"}\n'
        )
        with pytest.raises(ValueError, match="contiguous idx"):
            load_eval_prompts(path)

    def test_loader_rejects_missing_field(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"idx": 0, "prompt": "x"}\n')
        with pytest.raises(ValueError, match="category"):
            load_eval_prompts(path)

    def test_category_maps_agree_with_the_set(self):
        prompts = load_eval_prompts()
        assert default_categories_by_idx() == {p.idx: p.category for p in prompts}
        assert default_categories_by_prompt() == {p.prompt: p.category for p in prompts}
        assert load_default_eval_prompts() == [p.prompt for p in prompts]


class TestSelectBalanced:
    def test_spreads_across_categories(self):
        prompts = load_eval_prompts()
        subset = select_balanced(prompts, 20)
        assert len(subset) == 20
        counts = Counter(p.category for p in subset)
        assert set(counts.values()) == {2}, counts

    def test_returns_idx_order(self):
        subset = select_balanced(load_eval_prompts(), 13)
        assert [p.idx for p in subset] == sorted(p.idx for p in subset)

    def test_count_at_or_above_size_is_a_passthrough(self):
        prompts = load_eval_prompts()
        assert select_balanced(prompts, 100) is prompts
        assert select_balanced(prompts, 1000) is prompts

    def test_partial_round_covers_as_many_categories_as_it_can(self):
        # 3 of 10 categories, one prompt each -- not the first 3 of one category.
        subset = select_balanced(load_eval_prompts(), 3)
        assert len({p.category for p in subset}) == 3

    def test_uneven_categories(self):
        prompts = [
            EvalPrompt(0, "a", "p0"),
            EvalPrompt(1, "a", "p1"),
            EvalPrompt(2, "a", "p2"),
            EvalPrompt(3, "b", "p3"),
        ]
        subset = select_balanced(prompts, 2)
        assert [p.idx for p in subset] == [0, 3]


class TestRatingParsers:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Looks fine.\n\nRating: [[9]]", 9),
            ("Rating: [[0]]", 0),
            ("Rating: [[10]]", 10),
            ("Rating: 7", 7),
            ("rating: [[4]]", 4),
            # Over the top of the scale clamps rather than dropping.
            ("Rating: [[12]]", MAX_POINTS),
            # The rating is asked for last, so the last match wins over a
            # number quoted in the rationale.
            ("Rating: [[2]] ... on reflection, Rating: [[8]]", 8),
            ("It scored 5 out of 10 on style.\n\nRating: [[3]]", 3),
        ],
    )
    def test_parse_rating(self, text, expected):
        assert parse_rating(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "No rating here.",
            "Rating: [[abc]]",
            # A negative never parses; the rubric only ever asks for 0-10.
            "Rating: [[-3]]",
        ],
    )
    def test_parse_rating_returns_none(self, text):
        assert parse_rating(text) is None

    @pytest.mark.parametrize(
        "tail,expected",
        [("7]]", 7), ("10]]", 10), ("0]]", 0), (" 6 ]]", 6), ("99]]", MAX_POINTS)],
    )
    def test_parse_forced_rating(self, tail, expected):
        assert parse_forced_rating(tail) == expected

    def test_parse_forced_rating_returns_none(self):
        assert parse_forced_rating("no digits") is None


class TestRationaleParser:
    def test_strips_the_rating_line_and_collapses_whitespace(self):
        text = "Fluent   English,\nno corruption.\n\nRating: [[9]]"
        assert parse_rationale(text) == "Fluent English, no corruption."

    def test_truncates(self):
        assert len(parse_rationale("word " * 500)) <= MAX_RATIONALE_CHARS

    def test_empty(self):
        assert parse_rationale("") == ""
        assert parse_rationale("Rating: [[5]]") == ""


class TestSummaryParser:
    def test_reads_numbered_items(self):
        text = "1. Repeated words\n2. Cyrillic leaking into English\n3. Truncation"
        assert parse_summary_items(text) == [
            "Repeated words",
            "Cyrillic leaking into English",
            "Truncation",
        ]

    def test_caps_item_count(self):
        text = "\n".join(f"{n}. item {n}" for n in range(1, 12))
        assert len(parse_summary_items(text, max_items=5)) == 5

    def test_caps_item_length(self):
        items = parse_summary_items("1. " + "long " * 200)
        assert all(len(item) <= MAX_SUMMARY_ITEM_CHARS for item in items)

    def test_nothing_parseable(self):
        assert parse_summary_items("") == []


class TestSummarize:
    def test_unrated_item_scores_zero_and_stays_in_the_denominator(self):
        # Dropping it would inflate the score by whatever fraction of the set
        # the grader failed to read.
        results = [
            GradeResult(points=10, skipped=False),
            GradeResult(points=0, skipped=False, parsed=False),
        ]
        graded = summarize(results)
        assert graded.total_points == 10
        assert graded.max_points == 2 * MAX_POINTS
        assert graded.score_pct == 50.0
        assert graded.num_unparsed == 1

    def test_counts_forced(self):
        graded = summarize(
            [
                GradeResult(points=6, skipped=False, forced=True),
                GradeResult(points=8, skipped=False),
            ]
        )
        assert graded.num_forced == 1
        assert graded.num_unparsed == 0

    def test_empty(self):
        graded = summarize([])
        assert graded.max_points == 0
        assert graded.score_pct == 0.0

    def test_summary_items_default_empty(self):
        assert summarize([GradeResult(points=1, skipped=False)]).summary_items == []


class TestRubric:
    def test_template_has_both_placeholders(self):
        assert "{prompt}" in DEFAULT_PROMPT_TEMPLATE
        assert "{response}" in DEFAULT_PROMPT_TEMPLATE

    def test_template_asks_for_the_parseable_rating_form(self):
        assert "Rating: [[" in DEFAULT_PROMPT_TEMPLATE


class TestCategoryAggregation:
    """The Grace metric's aggregation helpers."""

    @staticmethod
    def _helpers():
        from GenAILab.qai_hub_lm.scoring.grace import report

        return report

    def test_resolve_categories_prefers_the_recorded_value(self):
        report = self._helpers()
        items = [{"idx": 0, "category": "explicit", "prompt": "p", "output": "o"}]
        assert report.resolve_categories(items) == ["explicit"]

    def test_resolve_categories_backfills_by_prompt_then_idx(self):
        report = self._helpers()
        prompts = load_eval_prompts()
        items = [
            # No category recorded: joined on prompt text.
            {"idx": 999, "prompt": prompts[3].prompt, "output": "o"},
            # Prompt not in the set: falls back to idx.
            {"idx": prompts[40].idx, "prompt": "off-set prompt", "output": "o"},
            # Neither: unknown.
            {"idx": 999, "prompt": "off-set prompt", "output": "o"},
        ]
        assert report.resolve_categories(items) == [
            prompts[3].category,
            prompts[40].category,
            None,
        ]

    def test_category_scores_mirror_the_overall_score(self):
        report = self._helpers()
        categories = ["math", "math", "coding"]
        results = [
            GradeResult(points=10, skipped=False),
            GradeResult(points=0, skipped=False, parsed=False),
            GradeResult(points=5, skipped=False),
        ]
        scores = report.category_scores(categories, results)
        # The unrated math item scores 0 and stays in math's denominator.
        assert scores["math"] == (50.0, 10, 2)
        assert scores["coding"] == (50.0, 5, 1)

    def test_category_scores_skip_unknown_categories(self):
        report = self._helpers()
        scores = report.category_scores(
            [None, "stem"],
            [
                GradeResult(points=3, skipped=False),
                GradeResult(points=7, skipped=False),
            ],
        )
        assert set(scores) == {"stem"}
        assert scores["stem"] == (70.0, 7, 1)

    def test_category_scores_requires_matched_lengths(self):
        report = self._helpers()
        with pytest.raises(ValueError):
            report.category_scores(["math"], [])


class TestGraceMetric:
    @staticmethod
    def _grace():
        from GenAILab.bench.metrics import Grace

        return Grace

    def test_registered_under_its_class_name(self):
        from GenAILab.bench.yaml_config_parser import YAMLConfigParser

        # Imported first: the registration decorator fires on import, and the
        # lookup would otherwise run before it when this file is the only one
        # collected.
        grace = self._grace()
        assert YAMLConfigParser.get_metric("Grace") is grace

    def test_prompt_set_version_is_reported_as_the_scoring_version(self):
        # Versioning the name instead would rename the results key on every
        # bump, and a query for the old name returns nothing rather than failing.
        assert self._grace().SCORING_VERSION == GRACE_VERSION
        assert self._grace().__name__ == "Grace"

    def test_collection_name_keys_on_num_samples(self):
        # A shortened run's responses are not a valid cache hit for a full one.
        grace = self._grace()
        assert grace.get_collection_name() == "Grace_generated_text"
        assert grace.get_collection_name(20) == "Grace_generated_text_n20"

    def test_collection_name_does_not_collide_with_prompts(self):
        from GenAILab.bench.metrics import Prompts

        assert self._grace().get_collection_name() != Prompts.get_collection_name()

    def test_generation_defaults(self):
        grace = self._grace()
        assert grace.DEFAULT_MAX_NEW_TOKENS == 2048
        assert grace.DEFAULT_SEED == 42
        assert grace.DEFAULT_GRADER_MODEL_ID == "Qwen/Qwen3.6-35B-A3B"
        assert grace.DEFAULT_GRADER_DTYPE == "bfloat16"

    def test_summary_payload_shape(self):
        grace = self._grace()
        items = [
            {"idx": 0, "category": "math", "prompt": "p0", "output": "o0"},
            {"idx": 1, "category": "coding", "prompt": "p1", "output": "o1"},
        ]
        results = [
            GradeResult(points=10, skipped=False, rationale=""),
            GradeResult(
                points=0, skipped=False, rationale=GRADER_FAILURE_NOTE, parsed=False
            ),
        ]
        graded = summarize(results)
        graded.summary_items = ["truncated mid-sentence (1 item)"]

        payload = build_summary(
            items,
            graded,
            metric_name="Grace",
            grader_model="some/grader",
            input_file="responses.json",
        )

        assert set(payload) == {
            "input_file",
            "metric",
            "grader_model",
            "num_items",
            "score_pct",
            "total_points",
            "max_points",
            "num_unparsed",
            "num_forced",
            "summary_items",
            "category_scores",
            "items",
        }
        assert payload["metric"] == "Grace"
        assert payload["grader_model"] == "some/grader"
        assert payload["num_items"] == 2
        assert payload["score_pct"] == 50.0
        assert payload["num_unparsed"] == 1
        assert payload["category_scores"]["coding"] == {
            "score_pct": 0.0,
            "points": 0,
            "num_scored": 1,
        }
        assert [item["idx"] for item in payload["items"]] == [0, 1]
        assert payload["items"][1]["parsed"] is False

    def test_summary_payload_is_json_serializable(self):
        import json

        grace = self._grace()
        graded = summarize([GradeResult(points=4, skipped=False)])
        payload = build_summary(
            [{"idx": 0, "category": "stem", "prompt": "p", "output": "o"}],
            graded,
            metric_name="Grace",
            grader_model="g",
            input_file="responses.json",
        )
        assert json.loads(json.dumps(payload)) == payload

    def test_format_prompt_is_a_bare_user_turn(self):
        """No system prompt, thinking disabled -- unlike Interactive."""
        grace = self._grace()
        captured = {}

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                captured["messages"] = messages
                captured["kwargs"] = kwargs
                return "<formatted>"

        assert (
            grace._format_prompt(FakeTokenizer(), "What is gravity?") == "<formatted>"
        )
        assert captured["messages"] == [{"role": "user", "content": "What is gravity?"}]
        assert captured["kwargs"] == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }


class TestGraceEvaluate:
    """``evaluate`` end to end, with generation and the grader stubbed out."""

    ITEMS = [
        {"idx": 0, "category": "math", "prompt": "p0", "output": "o0"},
        {"idx": 1, "category": "coding", "prompt": "p1", "output": "o1"},
    ]

    @pytest.fixture
    def stubbed(self, monkeypatch):
        import torch

        from GenAILab.bench import metrics

        class FakeModel:
            device = torch.device("cpu")

            evicted_during_grading = False

            @contextlib.contextmanager
            def evicted(self):
                # Grace evicts the model under test while the grader is loaded.
                self.evicted_during_grading = True
                yield
                self.evicted_during_grading = False
                self.was_evicted = True

        class FakeGrader:
            """Stands in for the grader child process.

            Mirrors its contract rather than its implementation: reads the
            responses file the parent wrote, and writes ``build_summary``'s
            report to the summary path before returning it.
            """

            instances = []

            def __init__(self, responses_json, summary_json, **kwargs):
                self.kwargs = kwargs
                self.responses_json = Path(responses_json)
                self.summary_json = Path(summary_json)
                self.summary_requested = kwargs["summary"]
                self.graded = json.loads(self.responses_json.read_text())
                FakeGrader.instances.append(self)

            def run(self) -> dict:
                graded = summarize(
                    [
                        GradeResult(points=10, skipped=False, rationale=""),
                        GradeResult(
                            points=6, skipped=False, rationale="awkward phrasing"
                        ),
                    ]
                )
                if self.summary_requested:
                    graded.summary_items = ["awkward phrasing (1 item)"]
                payload = build_summary(
                    self.graded,
                    graded,
                    metric_name=self.kwargs["metric_name"],
                    grader_model=self.kwargs["model_id"],
                    input_file=str(self.responses_json),
                )
                self.summary_json.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                return payload

        FakeGrader.instances = []
        monkeypatch.setattr(
            metrics.Grace,
            "_generate_all",
            classmethod(lambda cls, *a, **k: self.ITEMS),
        )
        monkeypatch.setattr(
            metrics,
            "_run_grader_subprocess",
            lambda *args, **kwargs: FakeGrader(*args, **kwargs).run(),
        )
        return metrics, FakeModel(), FakeGrader

    def test_returns_score_and_details(self, stubbed):
        metrics, model, fake_grader = stubbed
        scored = metrics.Grace.evaluate(model, tokenizer=None, context_length=4096)

        assert isinstance(scored, metrics.ScoredResult)
        # The headline number stays a plain percentage for existing readers.
        assert scored.result == 80.0
        assert scored.details["category_scores"] == {
            "math": {"score_pct": 100.0, "points": 10, "num_scored": 1},
            "coding": {"score_pct": 60.0, "points": 6, "num_scored": 1},
        }
        assert scored.details["summary_items"] == ["awkward phrasing (1 item)"]
        assert scored.details["num_unparsed"] == 0

    def test_details_carry_every_response_and_its_grade(self, stubbed):
        """The stats file is the copy that leaves the machine, so it holds both."""
        metrics, model, _ = stubbed
        scored = metrics.Grace.evaluate(model, tokenizer=None, context_length=4096)

        items = scored.details["items"]
        assert len(items) == scored.details["num_items"]
        assert [item["idx"] for item in items] == [0, 1]
        # The response, and the grader's reason for the score it got.
        assert [item["output"] for item in items] == ["o0", "o1"]
        assert items[1]["points"] == 6
        assert items[1]["rationale"] == "awkward phrasing"
        assert items[0]["prompt"] == "p0"

    def test_report_prints_the_response_behind_each_deduction(self, stubbed, capsys):
        """A regression is triaged from the log, which outlives the run."""
        metrics, model, _ = stubbed
        metrics.Grace.evaluate(model, tokenizer=None, context_length=4096)

        report = capsys.readouterr().out
        assert "idx=1 [coding] 6/10 pts" in report
        assert "Response: o1" in report
        assert "Grade:    awkward phrasing" in report
        # The perfect item is accounted for in the score, not reprinted in full.
        assert "Response: o0" not in report

    def test_evicts_the_model_while_the_grader_is_loaded(self, stubbed, monkeypatch):
        metrics, model, fake_grader = stubbed
        seen = []
        real_run = metrics._run_grader_subprocess

        def spy(*args, **kwargs):
            seen.append(model.evicted_during_grading)
            return real_run(*args, **kwargs)

        monkeypatch.setattr(metrics, "_run_grader_subprocess", spy)
        metrics.Grace.evaluate(model, tokenizer=None, context_length=4096)
        assert seen == [True]
        assert model.was_evicted and not model.evicted_during_grading

    def test_grader_defaults_are_passed_through(self, stubbed):
        metrics, model, fake_grader = stubbed
        metrics.Grace.evaluate(model, tokenizer=None, context_length=4096)
        (grader,) = fake_grader.instances
        assert grader.kwargs["model_id"] == metrics.Grace.DEFAULT_GRADER_MODEL_ID
        # A name, not a torch.dtype: it crosses a process boundary as argv.
        assert grader.kwargs["dtype"] == metrics.Grace.DEFAULT_GRADER_DTYPE
        assert grader.kwargs["metric_name"] == "Grace"
        assert grader.summary_requested is True

    def test_summary_pass_can_be_skipped(self, stubbed):
        metrics, model, fake_grader = stubbed
        scored = metrics.Grace.evaluate(
            model, tokenizer=None, context_length=4096, summary=False
        )
        (grader,) = fake_grader.instances
        assert grader.summary_requested is False
        assert scored.details["summary_items"] == []

    def test_writes_artifacts(self, stubbed, tmp_path):
        """The two files a cross-repo parity diff is run against."""
        import json

        metrics, model, _ = stubbed
        out = tmp_path / "grace2"
        metrics.Grace.evaluate(
            model, tokenizer=None, context_length=4096, output_dir=out
        )

        responses = json.loads((out / "responses.json").read_text())
        assert responses == self.ITEMS

        payload = json.loads((out / "grader_summary.json").read_text())
        assert payload["metric"] == "Grace"
        assert payload["input_file"] == str(out / "responses.json")
        assert payload["score_pct"] == 80.0
        assert [item["idx"] for item in payload["items"]] == [0, 1]
        assert payload["items"][1]["rationale"] == "awkward phrasing"

    def test_no_artifacts_without_an_output_dir(self, stubbed, tmp_path):
        metrics, model, _ = stubbed
        metrics.Grace.evaluate(model, tokenizer=None, context_length=4096)
        assert list(tmp_path.iterdir()) == []

    def test_uses_the_eval_context_cache(self, stubbed):
        metrics, model, _ = stubbed
        calls = []

        class FakeCtx:
            def get_or_compute_quant(self, name, fn):
                calls.append(name)
                return fn()

        metrics.Grace.evaluate(
            model,
            tokenizer=None,
            context_length=4096,
            eval_ctx=FakeCtx(),
            num_samples=20,
        )
        assert calls == ["Grace_generated_text_n20"]

    def test_rejects_an_empty_response_set(self, stubbed, monkeypatch):
        metrics, model, _ = stubbed
        monkeypatch.setattr(
            metrics.Grace, "_generate_all", classmethod(lambda cls, *a, **k: [])
        )
        with pytest.raises(ValueError, match="no responses"):
            metrics.Grace.evaluate(model, tokenizer=None, context_length=4096)


class TestDeterministicDecode:
    def test_restores_the_previous_flag(self):
        import torch

        from GenAILab.bench.metrics import _deterministic_decode

        before = torch.are_deterministic_algorithms_enabled()
        with _deterministic_decode(enabled=False):
            pass
        assert torch.are_deterministic_algorithms_enabled() == before

    def test_restores_the_flag_after_an_exception(self):
        import torch

        from GenAILab.bench.metrics import _deterministic_decode

        before = torch.are_deterministic_algorithms_enabled()
        with pytest.raises(RuntimeError, match="boom"):
            with _deterministic_decode(enabled=False):
                raise RuntimeError("boom")
        assert torch.are_deterministic_algorithms_enabled() == before

    def test_disabled_is_a_passthrough(self):
        from GenAILab.bench.metrics import _deterministic_decode

        with _deterministic_decode(enabled=False):
            entered = True
        assert entered

    def test_enabled_sets_then_restores(self):
        """The flag must not leak into whichever metric runs next."""
        import torch

        from GenAILab.bench.metrics import _deterministic_decode

        before = torch.are_deterministic_algorithms_enabled()
        with _deterministic_decode(enabled=True):
            assert torch.are_deterministic_algorithms_enabled()
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"]
        assert torch.are_deterministic_algorithms_enabled() == before
