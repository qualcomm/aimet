# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause


import os
from unittest.mock import MagicMock
import pytest

try:
    from aimet_onnx.common.amp.utils import (
        visualize_quantizer_group_sensitivity,
        visualize_pareto_curve,
        create_sensitivity_plot,
        _candidate_to_str,
        candidate_cost,
    )
    from aimet_onnx.common.defs import QuantizationDataType
except ImportError:
    from aimet_torch.common.amp.utils import (
        visualize_quantizer_group_sensitivity,
        visualize_pareto_curve,
        create_sensitivity_plot,
        _candidate_to_str,
        candidate_cost,
    )
    from aimet_torch.common.defs import QuantizationDataType


@pytest.fixture(scope="session", autouse=True)
def accuracy_list():
    return [
        (
            MagicMock(),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
            0.7,
            MagicMock(),
        ),
        (
            MagicMock(),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
            0.75,
            MagicMock(),
        ),
        (
            MagicMock(),
            ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),
            0.79,
            MagicMock(),
        ),
        (
            MagicMock(),
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            0.6,
            MagicMock(),
        ),
        (
            MagicMock(),
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            0.65,
            MagicMock(),
        ),
        (
            MagicMock(),
            ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),
            0.7,
            MagicMock(),
        ),
    ]


class TestCommonAMPUtils:
    def test_visualization_pareto_curve(self):
        pareto_list = [
            (1.0, 0.9, None, None),
            (0.99, 0.8, None, None),
            (0.98, 0.78, None, None),
            (0.97, 0.77, None, None),
            (0.92, 0.7, None, None),
            (0.5, 0.3, None, None),
        ]
        file_path = "artifacts"
        os.makedirs("artifacts", exist_ok=True)
        plot = visualize_pareto_curve(pareto_list, file_path)
        file_path = os.path.join(file_path, "pareto_curve.html")
        assert plot.hover
        assert plot.title.text == "Accuracy vs BitOps"
        assert os.path.isfile(file_path)

    def test_visualize_quantizer_group_sensitivity(self, accuracy_list):
        baseline_candidate = (
            (16, QuantizationDataType.int),
            (16, QuantizationDataType.int),
        )
        fp32_accuracy = 0.8

        results_dir = "artifacts"
        os.makedirs(results_dir, exist_ok=True)

        visualize_quantizer_group_sensitivity(
            accuracy_list, baseline_candidate, fp32_accuracy, results_dir
        )

        file_path = os.path.join(results_dir, "quantizer_group_sensitivity.html")
        assert os.path.isfile(file_path)

    def test_get_sensitivity_plot(self, accuracy_list):
        baseline_candidate = (
            (16, QuantizationDataType.int),
            (16, QuantizationDataType.int),
        )
        fp32_accuracy = 0.8

        plot = create_sensitivity_plot(accuracy_list, baseline_candidate, fp32_accuracy)
        df = plot.renderers[0].data_source.to_df()

        plotted_data = [
            (row.QuantizerGroup_Bitwidth, row.Accuracy_mean) for _, row in df.iterrows()
        ]
        real_data = [
            ((str(quantizer_group), _candidate_to_str(bw)), acc)
            for quantizer_group, bw, acc, _ in accuracy_list
        ]
        assert sorted(plotted_data) == sorted(real_data)

    def test_candidate_cost_factor(self):
        w16a16 = ((16, QuantizationDataType.int), (16, QuantizationDataType.int))
        fp16 = ((16, QuantizationDataType.float), (16, QuantizationDataType.float))
        w8a16 = ((16, QuantizationDataType.int), (8, QuantizationDataType.int))
        w4a16 = ((16, QuantizationDataType.int), (4, QuantizationDataType.int))
        w8a8 = ((8, QuantizationDataType.int), (8, QuantizationDataType.int))
        w4a8 = ((8, QuantizationDataType.int), (4, QuantizationDataType.int))
        w4afp16 = ((16, QuantizationDataType.float), (4, QuantizationDataType.int))
        a16 = ((16, QuantizationDataType.int), (None, None))
        a8 = ((8, QuantizationDataType.int), (None, None))

        assert candidate_cost(*w16a16) < candidate_cost(*fp16)
        assert candidate_cost(*w8a16) < candidate_cost(*w16a16)
        assert candidate_cost(*w8a8) < candidate_cost(*w8a16)
        assert candidate_cost(*w4a8) < candidate_cost(*w8a8)
        assert candidate_cost(*w4a16) < candidate_cost(*w8a16)
        assert candidate_cost(*w8a8) == candidate_cost(*w4a16)
        assert candidate_cost(*w4a16) < candidate_cost(*w4afp16)
        assert candidate_cost(*a8) < candidate_cost(*a16)
