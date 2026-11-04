# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the param quantizer folding done before ONNX evaluation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from GenAILab.bench.metrics import MMLU, PPL, MMLUKLDivergence, Prompts
from GenAILab.bench.onnx.run_genai import (
    _can_fold_param_quantizers,
    _fold_param_quantizers,
)
from GenAILab.bench.yaml_config_parser import ResolvedMetric
from GenAILab.tests.unit.onnx.test_generator_utils import _tiny_sim


def _config(*metric_classes, analysis=None):
    metrics = tuple(
        ResolvedMetric(name=c.__name__, metric_cls=c) for c in metric_classes
    )
    return SimpleNamespace(metrics=metrics, analysis=analysis)


class TestCanFoldParamQuantizers:
    def test_folds_without_distance_metrics_or_analysis(self):
        assert _can_fold_param_quantizers(_config(PPL, MMLU, Prompts))

    def test_distance_metric_prevents_folding(self):
        """A distance metric runs fp_mode, which folding would silently break."""
        assert not _can_fold_param_quantizers(_config(PPL, MMLUKLDivergence))

    def test_analysis_prevents_folding(self):
        assert not _can_fold_param_quantizers(_config(PPL, analysis=MagicMock()))

    def test_no_metrics_skips_folding(self):
        """Nothing is evaluated, so folding would only cost time."""
        assert not _can_fold_param_quantizers(_config())


class TestFoldParamQuantizers:
    def test_folds_every_sim_without_changing_outputs(self, capsys):
        backbone, visual = _tiny_sim(), _tiny_sim()
        x = {"x": np.random.randn(1, 8).astype(np.float32)}
        expected = backbone.session.run(None, x)[0]

        _fold_param_quantizers({"backbone": backbone, "visual": visual})

        assert backbone._folded_param_quantizers
        assert visual._folded_param_quantizers
        np.testing.assert_allclose(
            backbone.session.run(None, x)[0], expected, atol=1e-6
        )
        assert "Folded param quantizers into the weights of backbone, visual" in (
            capsys.readouterr().out
        )

    def test_unsupported_sim_is_left_unfolded(self, capsys):
        """A sim that cannot fold is reported and skipped; the others still fold."""
        backbone, visual = _tiny_sim(), MagicMock()
        visual.fold_param_quantizers.side_effect = NotImplementedError("If branches")

        _fold_param_quantizers({"backbone": backbone, "visual": visual})

        assert backbone._folded_param_quantizers
        out = capsys.readouterr().out
        assert "Not folding visual param quantizers: If branches" in out
        assert "Folded param quantizers into the weights of backbone " in out

    def test_sim_with_nothing_to_fold_is_not_reported(self, capsys):
        backbone = _tiny_sim()
        for quantizer in backbone.qc_quantize_op_dict.values():
            quantizer.enabled = False

        _fold_param_quantizers({"backbone": backbone})

        assert "Folded" not in capsys.readouterr().out

    def test_other_errors_propagate(self):
        sim = MagicMock()
        sim.fold_param_quantizers.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            _fold_param_quantizers({"backbone": sim})
