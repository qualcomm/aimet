# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Torch generator utility functions."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from aimet_torch.quantsim import QuantizationSimModel
from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator


class TestPlaceCollection:
    def test_places_backbone(self):
        from GenAILab.qai_hub_lm.backends.torch.generator_utils import place_collection

        mock_sim = MagicMock(spec=QuantizationSimModel)
        mock_sim.model = torch.nn.Linear(4, 4)
        collection = SimCollection(backbone=mock_sim)

        with patch(
            "GenAILab.qai_hub_lm.backends.torch.generator_utils.place_model"
        ) as mock_place:
            mock_place.return_value.__enter__ = MagicMock()
            mock_place.return_value.__exit__ = MagicMock(return_value=False)
            with place_collection(collection, torch.device("cpu")):
                pass
            mock_place.assert_called()

    def test_skips_none_fields(self):
        from GenAILab.qai_hub_lm.backends.torch.generator_utils import place_collection

        collection = SimCollection(backbone=MagicMock())
        collection.backbone = None
        collection.visual = None
        collection.embedding = None

        # Should not raise even with all None fields
        with patch(
            "GenAILab.qai_hub_lm.backends.torch.generator_utils.place_model"
        ) as mock_place:
            mock_place.return_value.__enter__ = MagicMock()
            mock_place.return_value.__exit__ = MagicMock(return_value=False)
            with place_collection(collection, torch.device("cpu")):
                pass


class TestTorchFPModeMixin:
    def test_fp_mode_disables_quantizers(self):
        from GenAILab.qai_hub_lm.backends.torch.generator_utils import generator_factory

        mock_sim = MagicMock()
        mock_sim.model = torch.nn.Linear(4, 4)
        collection = SimCollection(backbone=mock_sim)

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.qai_hub_lm.backends.torch.generator_utils.remove_all_quantizers"
        ) as mock_remove:
            mock_remove.return_value.__enter__ = MagicMock()
            mock_remove.return_value.__exit__ = MagicMock(return_value=False)
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            with gen.fp_mode():
                mock_remove.assert_called()

    def test_handles_visual_model(self):
        from GenAILab.qai_hub_lm.backends.torch.generator_utils import generator_factory

        mock_backbone = MagicMock()
        mock_backbone.model = torch.nn.Linear(4, 4)
        mock_visual = MagicMock()
        mock_visual.model = torch.nn.Linear(4, 4)
        embedding = torch.nn.Embedding(10, 4)

        collection = SimCollection(
            backbone=mock_backbone, visual=mock_visual, embedding=embedding
        )

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.qai_hub_lm.backends.torch.generator_utils.remove_all_quantizers"
        ) as mock_remove:
            mock_remove.return_value.__enter__ = MagicMock()
            mock_remove.return_value.__exit__ = MagicMock(return_value=False)
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=VLM_Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            with gen.fp_mode():
                # Should be called for backbone, visual, and embedding
                assert mock_remove.call_count >= 2


class TestTorchDevicePlacementMixin:
    def test_on_device_places_models(self):
        from GenAILab.qai_hub_lm.backends.torch.generator_utils import generator_factory

        mock_sim = MagicMock(spec=QuantizationSimModel)
        mock_sim.model = torch.nn.Linear(4, 4)
        collection = SimCollection(backbone=mock_sim)

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.qai_hub_lm.backends.torch.generator_utils.place_model"
        ) as mock_place:
            mock_place.return_value.__enter__ = MagicMock()
            mock_place.return_value.__exit__ = MagicMock(return_value=False)
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            with gen.on_device(torch.device("cpu")):
                pass
            mock_place.assert_called()


class TestGeneratorFactory:
    def test_creates_generator_for_llm(self):
        from GenAILab.qai_hub_lm.backends.torch.generator_utils import generator_factory

        mock_sim = MagicMock()
        mock_sim.model = torch.nn.Linear(4, 4)
        collection = SimCollection(backbone=mock_sim)

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.qai_hub_lm.backends.torch.generator_utils.remove_all_quantizers"
        ) as mock_remove:
            mock_remove.return_value.__enter__ = MagicMock()
            mock_remove.return_value.__exit__ = MagicMock(return_value=False)
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            assert isinstance(gen, Generator)

    def test_passes_extras_to_llm_generator(self):
        from GenAILab.qai_hub_lm.backends.torch.generator_utils import generator_factory

        class _ExtrasGenerator(Generator):
            def __init__(self, *args, ngram_tables=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.ngram_tables = ngram_tables

        mock_sim = MagicMock()
        mock_sim.model = torch.nn.Linear(4, 4)
        tables = torch.nn.ModuleList([torch.nn.Embedding(10, 4)])
        collection = SimCollection(backbone=mock_sim, extras={"ngram_tables": tables})
        tok = MagicMock()
        tok.eos_token_id = 0

        gen = generator_factory(
            sim_collection=collection,
            generator_cls=_ExtrasGenerator,
            tokenizer=tok,
            sequence_length=8,
            context_length=32,
        )
        assert gen.ngram_tables is tables

    def test_creates_vlm_generator_for_vlm(self):
        from GenAILab.qai_hub_lm.backends.torch.generator_utils import generator_factory

        mock_backbone = MagicMock()
        mock_backbone.model = torch.nn.Linear(4, 4)
        mock_visual = MagicMock()
        mock_visual.model = torch.nn.Linear(4, 4)
        embedding = torch.nn.Embedding(10, 4)
        mock_config = MagicMock()

        collection = SimCollection(
            backbone=mock_backbone,
            visual=mock_visual,
            embedding=embedding,
            config=mock_config,
        )

        tok = MagicMock()
        tok.eos_token_id = 0

        with patch(
            "GenAILab.qai_hub_lm.backends.torch.generator_utils.remove_all_quantizers"
        ) as mock_remove:
            mock_remove.return_value.__enter__ = MagicMock()
            mock_remove.return_value.__exit__ = MagicMock(return_value=False)
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=VLM_Generator,
                tokenizer=tok,
                sequence_length=8,
                context_length=32,
            )
            assert isinstance(gen, VLM_Generator)
