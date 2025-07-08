# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import os
from unittest.mock import MagicMock
import pytest

from aimet_common.amp.utils import (
    visualize_quantizer_group_sensitivity,
    visualize_pareto_curve,
    create_sensitivity_plot,
    _candidate_to_str,
    candidate_cost,
)
from aimet_common.defs import QuantizationDataType


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
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")
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
