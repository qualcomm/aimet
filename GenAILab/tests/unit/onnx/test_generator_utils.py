# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ONNX generator utility functions."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator
from GenAILab.qai_hub_lm.backends.onnx.generator_utils import _VisualONNXAdapter


def _tiny_sim():
    """Build a real QuantizationSimModel over a two-layer MLP."""
    import io

    import onnx
    from aimet_onnx.quantsim import QuantizationSimModel

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(8, 8)
            self.l2 = torch.nn.Linear(8, 4)

        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))

    x = torch.randn(1, 8)
    buf = io.BytesIO()
    torch.onnx.export(
        MLP().eval(), (x,), buf, input_names=["x"], output_names=["y"], opset_version=17
    )
    buf.seek(0)
    return QuantizationSimModel(
        onnx.load_model(buf),
        dummy_input={"x": x.numpy()},
        providers=["CPUExecutionProvider"],
    )


def _enabled_flags(sim):
    return {name: q.enabled for name, q in sim.qc_quantize_op_dict.items()}


class TestONNXFPModeMixin:
    def test_fp_mode_disables_and_restores_quantizers(self):
        from GenAILab.qai_hub_lm.backends.onnx.generator_utils import generator_factory

        backbone = _tiny_sim()
        before = _enabled_flags(backbone)
        # Guard the premise: the sim must start with something enabled to disable
        assert any(before.values())

        collection = SimCollection(backbone=backbone, config=MagicMock())
        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.qai_hub_lm.backends.onnx.generator_utils.TorchONNXInterface"
        ):
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            with gen.fp_mode():
                assert not any(_enabled_flags(backbone).values())

            # Restored to the original per-quantizer state, not blanket-enabled
            assert _enabled_flags(backbone) == before

    def test_fp_mode_covers_visual_model(self):
        from GenAILab.qai_hub_lm.backends.onnx.generator_utils import generator_factory

        backbone = _tiny_sim()
        visual = _tiny_sim()
        backbone_before = _enabled_flags(backbone)
        visual_before = _enabled_flags(visual)

        mock_config = MagicMock()
        mock_config.text_config = MagicMock()
        collection = SimCollection(
            backbone=backbone,
            visual=visual,
            embedding=MagicMock(),
            config=mock_config,
        )
        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.qai_hub_lm.backends.onnx.generator_utils.TorchONNXInterface"
        ):
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=VLM_Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            with gen.fp_mode():
                assert not any(_enabled_flags(backbone).values())
                assert not any(_enabled_flags(visual).values())

            assert _enabled_flags(backbone) == backbone_before
            assert _enabled_flags(visual) == visual_before


class TestGeneratorFactory:
    def test_wraps_backbone_in_torch_onnx_interface(self):
        from GenAILab.qai_hub_lm.backends.onnx.generator_utils import generator_factory

        mock_backbone = MagicMock()
        mock_config = MagicMock()
        collection = SimCollection(backbone=mock_backbone, config=mock_config)

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.qai_hub_lm.backends.onnx.generator_utils.TorchONNXInterface"
        ) as mock_interface:
            mock_interface.return_value = MagicMock()
            mock_interface.return_value.config = mock_config
            mock_interface.return_value.dtype = MagicMock()

            gen = generator_factory(
                sim_collection=collection,
                generator_cls=Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            mock_interface.assert_called_once_with(mock_backbone, mock_config)
            assert isinstance(gen, Generator)

    def test_passes_extras_to_llm_generator(self):
        from GenAILab.qai_hub_lm.backends.onnx.generator_utils import generator_factory

        class _ExtrasGenerator(Generator):
            def __init__(self, *args, ngram_tables=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.ngram_tables = ngram_tables

        mock_config = MagicMock()
        tables = torch.nn.ModuleList([torch.nn.Embedding(10, 4)])
        collection = SimCollection(
            backbone=MagicMock(), config=mock_config, extras={"ngram_tables": tables}
        )
        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.qai_hub_lm.backends.onnx.generator_utils.TorchONNXInterface"
        ) as mock_interface:
            mock_interface.return_value = MagicMock()
            mock_interface.return_value.config = mock_config
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=_ExtrasGenerator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
        assert gen.ngram_tables is tables

    def test_vlm_wraps_both_models(self):
        from GenAILab.qai_hub_lm.backends.onnx.generator_utils import generator_factory

        mock_backbone = MagicMock()
        mock_visual = MagicMock()
        mock_config = MagicMock()
        mock_config.text_config = MagicMock()
        embedding = MagicMock()

        collection = SimCollection(
            backbone=mock_backbone,
            visual=mock_visual,
            embedding=embedding,
            config=mock_config,
        )

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.qai_hub_lm.backends.onnx.generator_utils.TorchONNXInterface"
        ) as mock_interface:
            mock_bb_wrapped = MagicMock()
            mock_vis_wrapped = MagicMock()
            mock_interface.side_effect = [mock_bb_wrapped, mock_vis_wrapped]

            gen = generator_factory(
                sim_collection=collection,
                generator_cls=VLM_Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            assert mock_interface.call_count == 2
            assert isinstance(gen, VLM_Generator)


class TestVisualONNXAdapter:
    def _make_adapter(self, num_list_outputs=0):
        interface = MagicMock()
        interface.config = MagicMock(name="config")
        interface.device = torch.device("cpu")
        interface.dtype = torch.float32
        return _VisualONNXAdapter(interface, num_list_outputs)

    def test_forwards_properties(self):
        adapter = self._make_adapter()
        assert adapter.config is adapter.interface.config
        assert adapter.device == torch.device("cpu")
        assert adapter.dtype == torch.float32

    def test_reassembles_list_outputs(self):
        adapter = self._make_adapter(num_list_outputs=3)
        # Simulate 5 flat outputs: 2 base + 3 deepstack layers
        flat = (
            torch.randn(4, 64),  # image_embeddings
            torch.randn(1, 8),  # visual_pos_masks
            torch.randn(4, 64),  # deepstack layer 0
            torch.randn(4, 64),  # deepstack layer 1
            torch.randn(4, 64),  # deepstack layer 2
        )
        adapter.interface.return_value = flat
        result = adapter(torch.randn(1, 3, 224, 224))

        # Should be (image_embeddings, visual_pos_masks, [ds0, ds1, ds2])
        assert len(result) == 3
        assert torch.equal(result[0], flat[0])
        assert torch.equal(result[1], flat[1])
        assert isinstance(result[2], list)
        assert len(result[2]) == 3
        assert torch.equal(result[2][0], flat[2])
        assert torch.equal(result[2][2], flat[4])

    def test_no_list_outputs_passthrough(self):
        adapter = self._make_adapter(num_list_outputs=0)
        out = (torch.randn(4, 64), torch.randn(1, 8))
        adapter.interface.return_value = out
        result = adapter(torch.randn(1, 3, 224, 224))

        assert result is out
