# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ONNX quantization recipe helpers."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import ml_dtypes
import numpy as np
import torch


class TestPrefillInputs:
    def test_bfloat16_tensors_become_ml_dtypes_arrays(self):
        """bf16 prefill tensors are kept bit-exact as ml_dtypes.bfloat16 arrays."""
        from GenAILab.bench.onnx.quant_recipes import _prefill_inputs

        hidden = torch.randn(2, 4, dtype=torch.bfloat16)
        input_ids = torch.arange(4, dtype=torch.int64)

        quantsim = MagicMock()
        quantsim.session.get_inputs.return_value = [
            SimpleNamespace(name="input_ids"),
            SimpleNamespace(name="hidden"),
        ]
        generator = MagicMock()
        generator.device = torch.device("cpu")
        generator.fp_mode.return_value = nullcontext()
        generator.prefill.return_value = [{"input_ids": input_ids, "hidden": hidden}]

        (inputs,) = _prefill_inputs(quantsim, generator, [{"x": torch.zeros(1)}], 1)

        assert inputs["input_ids"].dtype == np.int64
        np.testing.assert_array_equal(inputs["input_ids"], input_ids.numpy())
        assert inputs["hidden"].dtype == ml_dtypes.bfloat16
        np.testing.assert_array_equal(
            inputs["hidden"].astype(np.float32), hidden.float().numpy()
        )
