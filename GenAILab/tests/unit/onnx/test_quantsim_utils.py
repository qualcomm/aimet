# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ONNX quantsim utility functions."""

from unittest.mock import MagicMock

import pytest
import torch

from aimet_onnx.common.defs import float16, float32, int4
from aimet_onnx.defs import QSpec
from GenAILab.bench.precision import Granularity, WeightPrecision


class TestWeightPrecisionToQSpec:
    @pytest.mark.parametrize(
        "precision, expected",
        [
            (WeightPrecision(qtype=float16), QSpec.per_tensor(float16)),
            (WeightPrecision(qtype=float32), QSpec.per_tensor(float32)),
            (
                WeightPrecision(qtype=int4, granularity=Granularity.PCQ),
                QSpec.per_channel(int4),
            ),
            (
                WeightPrecision(
                    qtype=int4,
                    granularity=Granularity.BQ,
                    block_size=64,
                ),
                QSpec.blockwise(int4, block_size=64, symmetric=True),
            ),
            (
                WeightPrecision(
                    qtype=int4,
                    granularity=Granularity.LPBQ,
                    block_size=32,
                ),
                QSpec.lpbq(int4, block_size=32),
            ),
        ],
    )
    def test_conversion(self, precision, expected):
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import (
            weight_precision_to_qspec,
        )

        assert weight_precision_to_qspec(precision) == expected


class TestDisableFloatQuantizers:
    @pytest.mark.parametrize("precision", ["float16", "float32"])
    def test_disables_float_quantizers(self, precision):
        from aimet_onnx.defs import qtype
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import (
            _disable_float_quantizers,
        )

        quantizer = MagicMock()
        quantizer.precision.return_value = qtype.from_string(precision)
        sim = MagicMock()
        sim.qc_quantize_op_dict = {"tensor": quantizer}

        _disable_float_quantizers(sim)

        assert quantizer.enabled is False

    def test_leaves_integer_quantizers_enabled(self):
        from aimet_onnx.common.defs import int8
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import (
            _disable_float_quantizers,
        )

        quantizer = MagicMock()
        quantizer.precision.return_value = int8
        quantizer.enabled = True
        sim = MagicMock()
        sim.qc_quantize_op_dict = {"tensor": quantizer}

        _disable_float_quantizers(sim)

        assert quantizer.enabled is True


class TestGetOrtProviders:
    def test_cpu_device(self):
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import get_ort_providers

        providers = get_ort_providers(torch.device("cpu"))
        assert providers == ["CPUExecutionProvider"]

    def test_cuda_device_with_index(self):
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import get_ort_providers

        providers = get_ort_providers(torch.device("cuda", 0))
        assert len(providers) == 2
        assert providers[0] == ("CUDAExecutionProvider", {"device_id": 0})
        assert providers[1] == "CPUExecutionProvider"

    def test_cuda_device_without_index(self):
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import get_ort_providers

        providers = get_ort_providers(torch.device("cuda"))
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


class TestQuantizeEmbeddingWeights:
    def test_bfloat16_matches_float32(self):
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import (
            quantize_embedding_weights,
        )

        # An asymmetric range (as in real embeddings) gives a nonzero offset,
        # which bfloat16 arithmetic cannot add and subtract back exactly.
        torch.manual_seed(0)
        weight = (torch.randn(1024, 64) * 0.05).clamp(-0.3, 0.2).to(torch.bfloat16)
        fp32_emb = torch.nn.Embedding.from_pretrained(weight.float())
        bf16_emb = torch.nn.Embedding.from_pretrained(weight.clone())

        quantize_embedding_weights(fp32_emb, 16)
        quantize_embedding_weights(bf16_emb, 16)

        assert bf16_emb.weight.dtype == torch.bfloat16
        torch.testing.assert_close(bf16_emb.weight, fp32_emb.weight.to(torch.bfloat16))


class TestAttributePatch:
    def test_sets_and_restores_attribute(self):
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import AttributePatch

        class Obj:
            x = 10

        obj = Obj()
        with AttributePatch(obj, "x", 42):
            assert obj.x == 42
        assert obj.x == 10

    def test_creates_and_removes_new_attribute(self):
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import AttributePatch

        class Obj:
            pass

        obj = Obj()
        assert not hasattr(obj, "y")
        with AttributePatch(obj, "y", 99):
            assert obj.y == 99
        assert not hasattr(obj, "y")

    def test_restores_class_level_attribute(self):
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import AttributePatch

        class Obj:
            z = "original"

        obj = Obj()
        with AttributePatch(obj, "z", "patched"):
            assert obj.z == "patched"
        # After exit, the instance override is removed, class attr remains
        assert obj.z == "original"


class TestRemoveActivationQuantizers:
    def test_disables_activation_quantizers(self):
        from GenAILab.qai_hub_lm.backends.onnx.quantsim_utils import (
            _remove_activation_quantizers,
        )

        mock_qsim = MagicMock()
        act_op = MagicMock()
        weight_op = MagicMock()
        mock_qsim.qc_quantize_op_dict = {
            "act_tensor": act_op,
            "weight_tensor": weight_op,
        }
        mock_qsim.activation_names = {"act_tensor"}

        _remove_activation_quantizers(mock_qsim)

        act_op.reset_encoding_stats.assert_called_once()
        assert act_op.enabled is False
        weight_op.reset_encoding_stats.assert_not_called()
