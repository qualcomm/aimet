# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for metric computation logic and the ``run_metrics`` loop (no model needed)."""

import pytest
import torch

from GenAILab.bench.metrics import (
    PPL,
    TextEvaluationMetric,
    run_metrics,
    _KLDivergenceCompute,
    _ReverseKLDivergenceCompute,
    _JSDivergenceCompute,
    _FlipsCompute,
)
from GenAILab.bench.yaml_config_parser import ResolvedMetric


class TestPPLLoss:
    def test_compute_loss_known(self):
        # Deterministic case: logits that perfectly predict labels
        vocab_size = 4
        seq_len = 5
        labels = torch.tensor([[0, 1, 2, 3, 0]])
        # Create logits that assign high probability to the shifted labels
        logits = torch.zeros(1, seq_len, vocab_size)
        for i in range(seq_len - 1):
            logits[0, i, labels[0, i + 1]] = 10.0
        loss = PPL._compute_loss_from_logits(logits, labels)
        # Loss should be very small since we're predicting correctly
        assert loss.item() < 1.0

    def test_compute_loss_random(self):
        logits = torch.randn(1, 10, 100)
        labels = torch.randint(0, 100, (1, 10))
        loss = PPL._compute_loss_from_logits(logits, labels)
        assert loss.item() > 0


class TestKLDivergence:
    def test_identical_distributions(self):
        data = {"logits": torch.randn(100, 4)}
        result = _KLDivergenceCompute._compute(data, data)
        assert abs(result) < 1e-5

    def test_different_distributions(self):
        fp_data = {"logits": torch.tensor([[10.0, 0.0, 0.0, 0.0]] * 50)}
        q_data = {"logits": torch.tensor([[0.0, 10.0, 0.0, 0.0]] * 50)}
        result = _KLDivergenceCompute._compute(fp_data, q_data)
        assert result > 0


class TestReverseKLDivergence:
    def test_identical_distributions(self):
        data = {"logits": torch.randn(100, 4)}
        result = _ReverseKLDivergenceCompute._compute(data, data)
        assert abs(result) < 1e-5

    def test_asymmetry(self):
        fp_data = {"logits": torch.tensor([[10.0, 1.0, 0.5, 0.1]] * 50)}
        q_data = {"logits": torch.tensor([[1.0, 5.0, 0.0, 0.0]] * 50)}
        kl_forward = _KLDivergenceCompute._compute(fp_data, q_data)
        kl_reverse = _ReverseKLDivergenceCompute._compute(fp_data, q_data)
        # KL divergence is asymmetric in general
        assert kl_forward != pytest.approx(kl_reverse, abs=1e-3)


class TestJSDivergence:
    def test_symmetric(self):
        fp_data = {"logits": torch.randn(50, 4)}
        q_data = {"logits": torch.randn(50, 4)}
        js1 = _JSDivergenceCompute._compute(fp_data, q_data)
        js2 = _JSDivergenceCompute._compute(q_data, fp_data)
        assert js1 == pytest.approx(js2, abs=1e-6)

    def test_identical_distributions(self):
        data = {"logits": torch.randn(100, 4)}
        result = _JSDivergenceCompute._compute(data, data)
        assert abs(result) < 1e-5


class TestFlips:
    def test_all_same(self):
        data = {"logits": torch.tensor([[10.0, 0.0, 0.0, 0.0]] * 50)}
        result = _FlipsCompute._compute(data, data)
        assert result == 0.0

    def test_all_different(self):
        fp_data = {"logits": torch.tensor([[10.0, 0.0, 0.0, 0.0]] * 50)}
        q_data = {"logits": torch.tensor([[0.0, 10.0, 0.0, 0.0]] * 50)}
        result = _FlipsCompute._compute(fp_data, q_data)
        assert result == 100.0

    def test_partial_flips(self):
        fp_data = {"logits": torch.tensor([[10.0, 0.0, 0.0, 0.0]] * 100)}
        q_logits = [[10.0, 0.0, 0.0, 0.0]] * 75 + [[0.0, 10.0, 0.0, 0.0]] * 25
        q_data = {"logits": torch.tensor(q_logits)}
        result = _FlipsCompute._compute(fp_data, q_data)
        assert result == pytest.approx(25.0, abs=0.1)


class TestRunMetricsExtraKwargs:
    """``extra_kwargs`` (the analysis pass's per-condition ``output_dir``) reach
    only metrics whose ``evaluate`` declares them: most metrics are keyword-only
    without ``**kwargs`` and would raise ``TypeError`` if passed one blindly."""

    class DeclaresOutputDir(TextEvaluationMetric):
        @classmethod
        def evaluate(
            cls,
            model,
            tokenizer,
            context_length,
            *,
            eval_ctx,
            output_dir=None,
            **kwargs,
        ):
            return output_dir

    class OnlyKwargs(TextEvaluationMetric):
        @classmethod
        def evaluate(cls, model, tokenizer, context_length, *, eval_ctx, **kwargs):
            return sorted(kwargs)

    @staticmethod
    def _run(metric_cls, metric_kwargs=None):
        [result] = run_metrics(
            (
                ResolvedMetric(
                    name=metric_cls.__name__,
                    metric_cls=metric_cls,
                    metric_kwargs=metric_kwargs or {},
                ),
            ),
            generator="gen",
            tokenizer=object(),
            context_length=128,
            eval_ctx="ctx",
            extra_kwargs={"output_dir": "analysis/trunc_bits8"},
        )
        return result.result

    def test_injected_when_declared(self):
        assert self._run(self.DeclaresOutputDir) == "analysis/trunc_bits8"

    def test_omitted_when_only_absorbed_by_kwargs(self):
        assert self._run(self.OnlyKwargs) == []

    def test_overrides_same_key_in_metric_kwargs(self):
        """An ``output_dir`` set on the metric in the config must not collide
        with the per-condition one (``TypeError: multiple values``); the
        condition's wins, so conditions don't overwrite each other's files."""
        assert (
            self._run(self.DeclaresOutputDir, {"output_dir": "user/dir"})
            == "analysis/trunc_bits8"
        )
