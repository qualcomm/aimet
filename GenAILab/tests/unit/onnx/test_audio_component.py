# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Audio-component support in the ONNX backend: generic component handling in
the generator utils, :class:`ModelCacheEntry` persistence (including entries
written before audio existed), and capability gating in ``instantiate_quantsim``.
"""

import io
import shutil
from unittest.mock import MagicMock, patch

import onnx
import pytest
import torch

from GenAILab.bench.model_cache import DiskBackedModelCache, ModelCacheEntry
from GenAILab.bench.precision import PrecisionConfig
from GenAILab.qai_hub_lm.models.base import SimCollection
from GenAILab.qai_hub_lm.models.generator import VLM_Generator


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 8)
        self.l2 = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))


def _tiny_onnx() -> onnx.ModelProto:
    """A minimal but real ONNX graph, so cache round-trips exercise real IO."""
    buf = io.BytesIO()
    torch.onnx.export(
        _MLP().eval(),
        (torch.randn(1, 8),),
        buf,
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
    )
    buf.seek(0)
    return onnx.load_model(buf)


def _tiny_sim():
    """A real ONNX QuantizationSimModel over the same graph."""
    from aimet_onnx.quantsim import QuantizationSimModel

    return QuantizationSimModel(
        _tiny_onnx(),
        dummy_input={"x": torch.randn(1, 8).numpy()},
        providers=["CPUExecutionProvider"],
    )


def _enabled_flags(sim):
    return {name: q.enabled for name, q in sim.qc_quantize_op_dict.items()}


def _mock_config():
    config = MagicMock()
    config.text_config = MagicMock()
    return config


def _tokenizer():
    tok = MagicMock()
    tok.eos_token_id = 0
    return tok


# ---------------------------------------------------------------------------
# generator utils
# ---------------------------------------------------------------------------


class TestAudioOnlyGeneratorFactory:
    def test_audio_sim_is_wired_to_audio_model_kwarg(self):
        from GenAILab.qai_hub_lm.backends.onnx.generator_utils import generator_factory

        collection = SimCollection(
            backbone=MagicMock(),
            audio=MagicMock(),
            embedding=MagicMock(),
            config=_mock_config(),
        )

        with patch(
            "GenAILab.qai_hub_lm.backends.onnx.generator_utils.TorchONNXInterface"
        ) as mock_interface:
            audio_wrapped, backbone_wrapped = MagicMock(), MagicMock()
            # The encoder interfaces are built before the backbone's.
            mock_interface.side_effect = [audio_wrapped, backbone_wrapped]

            gen = generator_factory(
                sim_collection=collection,
                generator_cls=VLM_Generator,
                tokenizer=_tokenizer(),
                sequence_length=8,
                context_length=32,
            )

        assert isinstance(gen, VLM_Generator)
        assert gen.audio_model is audio_wrapped
        # An audio-only model must not acquire a vision tower.
        assert gen.vision_model is None
        assert gen.present_components() == ("audio",)

    def test_fp_mode_covers_audio_model(self):
        from GenAILab.qai_hub_lm.backends.onnx.generator_utils import generator_factory

        backbone, audio = _tiny_sim(), _tiny_sim()
        backbone_before = _enabled_flags(backbone)
        audio_before = _enabled_flags(audio)
        # Guard the premise: something must start enabled to be disabled.
        assert any(audio_before.values())

        collection = SimCollection(
            backbone=backbone,
            audio=audio,
            embedding=MagicMock(),
            config=_mock_config(),
        )

        with patch(
            "GenAILab.qai_hub_lm.backends.onnx.generator_utils.TorchONNXInterface"
        ):
            gen = generator_factory(
                sim_collection=collection,
                generator_cls=VLM_Generator,
                tokenizer=_tokenizer(),
                sequence_length=8,
                context_length=32,
            )
            with gen.fp_mode():
                assert not any(_enabled_flags(backbone).values())
                assert not any(_enabled_flags(audio).values())

            # Restored per-quantizer, not blanket-enabled.
            assert _enabled_flags(backbone) == backbone_before
            assert _enabled_flags(audio) == audio_before


# ---------------------------------------------------------------------------
# model cache
# ---------------------------------------------------------------------------


class TestModelCacheEntryComponents:
    def test_component_accessors(self):
        entry = ModelCacheEntry(backbone=MagicMock(), audio=MagicMock())
        assert entry.component("audio") is entry.audio
        assert entry.component("visual") is None
        assert entry.present_components() == ("audio",)
        with pytest.raises(KeyError):
            entry.component("nonsense")

    def test_present_components_in_canonical_order(self):
        entry = ModelCacheEntry(
            backbone=MagicMock(), audio=MagicMock(), visual=MagicMock()
        )
        assert entry.present_components() == ("visual", "audio")

    def test_round_trip_with_audio(self, tmp_path):
        cache = DiskBackedModelCache(tmp_path / "model_cache")
        cache.put(
            "audio_key",
            ModelCacheEntry(backbone=_tiny_onnx(), audio=_tiny_onnx()),
            metadata={"small_model": True},
        )

        audio_dir = tmp_path / "model_cache" / "audio_key" / "audio"
        assert (audio_dir / "model.onnx").exists()

        got = cache.get("audio_key")
        assert got is not None
        assert got.audio is not None
        assert got.visual is None
        assert got.present_components() == ("audio",)

    def test_round_trip_without_audio_writes_no_audio_dir(self, tmp_path):
        """A vision-only entry must be byte-for-byte the layout it always was."""
        cache = DiskBackedModelCache(tmp_path / "model_cache")
        cache.put(
            "visual_key",
            ModelCacheEntry(backbone=_tiny_onnx(), visual=_tiny_onnx()),
            metadata={"small_model": True},
        )

        entry_dir = tmp_path / "model_cache" / "visual_key"
        assert (entry_dir / "visual" / "model.onnx").exists()
        assert not (entry_dir / "audio").exists()

        got = cache.get("visual_key")
        assert got.visual is not None
        assert got.audio is None

    def test_legacy_entry_without_audio_dir_still_loads(self, tmp_path):
        """Backward compat: an entry written before audio existed has no
        ``audio/`` subdir, which must read back as ``audio=None`` rather than
        failing the whole cache hit."""
        cache = DiskBackedModelCache(tmp_path / "model_cache")
        cache.put(
            "legacy_key",
            ModelCacheEntry(backbone=_tiny_onnx(), audio=_tiny_onnx()),
            metadata={"small_model": True},
        )
        # Emulate the on-disk shape of a pre-audio entry.
        shutil.rmtree(tmp_path / "model_cache" / "legacy_key" / "audio")

        got = cache.get("legacy_key")
        assert got is not None
        assert got.backbone is not None
        assert got.audio is None
        assert got.present_components() == ()


# ---------------------------------------------------------------------------
# capability gating in instantiate_quantsim
# ---------------------------------------------------------------------------


_VLM_MODULE = "GenAILab.qai_hub_lm.backends.onnx.vlm"


def _run_instantiate_quantsim(model_cls, entry, precision):
    topology = MagicMock(name="topology")
    patches = {
        f"{_VLM_MODULE}.QuantizationSimModel": MagicMock(),
        f"{_VLM_MODULE}.configure_llm": MagicMock(),
        f"{_VLM_MODULE}.weight_precision_to_qspec": MagicMock(),
        f"{_VLM_MODULE}._disable_float_quantizers": MagicMock(),
        f"{_VLM_MODULE}.get_ort_providers": MagicMock(
            return_value=["CPUExecutionProvider"]
        ),
        f"{_VLM_MODULE}.AttributePatch": MagicMock(),
        f"{_VLM_MODULE}.QUANTSIM_CONFIG": "config.json",
    }
    started = {key: patch(key, value) for key, value in patches.items()}
    mocks = {key: p.start() for key, p in started.items()}
    try:
        collection = model_cls.instantiate_quantsim(
            entry,
            precision=precision,
            topology=topology,
        )
        return collection, mocks, topology
    finally:
        for p in started.values():
            p.stop()


class TestAudioOnlyQuantsimGating:
    @staticmethod
    def _audio_only_cls():
        from GenAILab.qai_hub_lm.backends.onnx.vlm import VLM_ONNX

        class _AudioOnly_ONNX(VLM_ONNX):
            COMPONENTS = ("audio",)

            @classmethod
            def instantiate_position_processor(cls):
                return None

        return _AudioOnly_ONNX

    def test_builds_audio_sim_and_no_visual_sim(self):
        entry = ModelCacheEntry(
            backbone=MagicMock(), audio=MagicMock(), config=MagicMock()
        )
        precision = PrecisionConfig()

        collection, mocks, topology = _run_instantiate_quantsim(
            self._audio_only_cls(), entry, precision
        )

        # One sim for the backbone, one for audio -- nothing for vision.
        assert mocks[f"{_VLM_MODULE}.QuantizationSimModel"].call_count == 2
        assert mocks[f"{_VLM_MODULE}.configure_llm"].call_args.args[1] is topology
        assert mocks[f"{_VLM_MODULE}._disable_float_quantizers"].call_count == 2
        assert collection.has("audio")
        assert collection.visual is None
        assert collection.present_components() == ("audio",)

    def test_no_stray_visual_precision_block(self):
        entry = ModelCacheEntry(
            backbone=MagicMock(), audio=MagicMock(), config=MagicMock()
        )
        precision = PrecisionConfig()

        _run_instantiate_quantsim(self._audio_only_cls(), entry, precision)

        assert precision.audio_weight is not None
        assert precision.audio_activations is not None
        # Defaults are only filled in for declared components, so the config
        # still hashes as if visual never existed.
        assert precision.visual_weight is None
        assert precision.visual_activations is None
        assert "visual" not in precision.to_dict()
        assert "audio" in precision.to_dict()

    def test_declared_component_with_no_graph_yields_no_sim(self):
        """A checkpoint exported before the component existed must not crash."""
        entry = ModelCacheEntry(backbone=MagicMock(), config=MagicMock())

        collection, mocks, _ = _run_instantiate_quantsim(
            self._audio_only_cls(), entry, PrecisionConfig()
        )

        assert mocks[f"{_VLM_MODULE}.QuantizationSimModel"].call_count == 1
        assert mocks[f"{_VLM_MODULE}._disable_float_quantizers"].call_count == 1
        assert collection.present_components() == ()
