# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ONNX LLM quantsim instantiation logic.

These tests verify the orchestration in LLM_ONNX.instantiate_quantsim
without loading real models — all heavy dependencies are mocked.
"""

from unittest.mock import MagicMock, call, patch

import pytest

from GenAILab.bench.precision import PrecisionConfig


class TestOnnxInstantiateQuantsimOrchestration:
    def _run_instantiate(self, precision=None, extra_patches=None):
        """Helper to run instantiate_quantsim with all heavy deps mocked."""
        topology = MagicMock(name="topology")
        block_spec = MagicMock(name="block_spec")
        lm_head_spec = MagicMock(name="lm_head_spec")
        patches = {
            "GenAILab.qai_hub_lm.backends.onnx.llm.is_huggingface_ckpt": MagicMock(
                return_value=False
            ),
            "GenAILab.qai_hub_lm.backends.onnx.llm.load_model_components_from_disk": MagicMock(
                return_value=(MagicMock(), None)
            ),
            "GenAILab.qai_hub_lm.backends.onnx.llm.AutoConfig": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm.QuantizationSimModel": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm.configure_llm": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm.weight_precision_to_qspec": MagicMock(
                side_effect=[block_spec, lm_head_spec]
            ),
            "GenAILab.qai_hub_lm.backends.onnx.llm._disable_float_quantizers": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm.get_ort_providers": MagicMock(
                return_value=["CPUExecutionProvider"]
            ),
            "GenAILab.qai_hub_lm.backends.onnx.llm.AttributePatch": MagicMock(),
            "GenAILab.qai_hub_lm.backends.onnx.llm.QUANTSIM_CONFIG": "config.json",
        }
        if extra_patches:
            patches.update(extra_patches)

        active_patches = {k: patch(k, v) for k, v in patches.items()}
        mocks = {}
        for k, p in active_patches.items():
            mocks[k] = p.start()

        try:
            from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX

            entry = LLM_ONNX.instantiate_float_model(
                model_id="/fake/path",
                context_length=32,
                sequence_length=8,
            )
            result = LLM_ONNX.instantiate_quantsim(
                entry,
                precision=precision,
                topology=topology,
            )
            return result, mocks, topology, block_spec, lm_head_spec
        finally:
            for p in active_patches.values():
                p.stop()

    def test_default_precision_when_none(self):
        result, *_ = self._run_instantiate(precision=None)
        # Should not fail — default PrecisionConfig should be used
        from GenAILab.qai_hub_lm.models.base import SimCollection

        assert isinstance(result, SimCollection)

    def test_configures_llm(self):
        precision = PrecisionConfig()
        _, mocks, topology, block_spec, lm_head_spec = self._run_instantiate(precision)
        quant_sim = mocks[
            "GenAILab.qai_hub_lm.backends.onnx.llm.QuantizationSimModel"
        ].return_value

        mocks[
            "GenAILab.qai_hub_lm.backends.onnx.llm.weight_precision_to_qspec"
        ].assert_has_calls(
            [
                call(precision.blocks["default"]),
                call(precision.lm_head),
            ]
        )
        mocks[
            "GenAILab.qai_hub_lm.backends.onnx.llm.configure_llm"
        ].assert_called_once_with(
            quant_sim,
            topology,
            kv_cache_type=precision.resolve_kv_cache_qtype(),
            backbone_weight_type=block_spec,
            lm_head_weight_type=lm_head_spec,
        )
        mocks[
            "GenAILab.qai_hub_lm.backends.onnx.llm._disable_float_quantizers"
        ].assert_called_once_with(quant_sim)

    def test_requires_topology(self):
        from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX

        with pytest.raises(TypeError, match="topology"):
            LLM_ONNX.instantiate_quantsim(MagicMock())

    def test_returns_sim_collection_with_config(self):
        result, *_ = self._run_instantiate()
        from GenAILab.qai_hub_lm.models.base import SimCollection

        assert isinstance(result, SimCollection)
        # Config should be set from AutoConfig
        assert result.config is not None
