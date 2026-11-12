# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""The on-sim chain hands a calibration dataset only the model kwargs it takes.

``image_size`` and ``audio_frames`` are model-level settings; a vision dataset's
loader has no ``audio_frames`` (or ``model``) parameter and an audio dataset's no
``image_size``, so injecting them blindly raises ``TypeError`` at load time.
"""

from unittest.mock import MagicMock, patch

from transformers.processing_utils import ProcessorMixin

from GenAILab.bench.datasets import MultimodalDataset
from GenAILab.bench.recipe_chain import apply_quantization_chain


class _ImageDataset(MultimodalDataset):
    calls = []

    @staticmethod
    def load_dataset(split):
        raise NotImplementedError

    @classmethod
    def load_encoded_dataset(
        cls, processor, context_length, split="train", image_size=None
    ):
        cls.calls.append({"image_size": image_size})
        return []


class _AudioDataset(MultimodalDataset):
    calls = []

    @staticmethod
    def load_dataset(split):
        raise NotImplementedError

    @classmethod
    def load_encoded_dataset(
        cls, processor, context_length, split="train", audio_frames=None, model=None
    ):
        cls.calls.append({"audio_frames": audio_frames, "model": model})
        return []


def _run(dataset_cls, generator, image_size=None, audio_frames=None):
    technique_cls = MagicMock()
    technique_cls.cacheable.return_value = False
    technique_cls.__name__ = "FakeTechnique"
    step = MagicMock()
    step.name = "FakeTechnique"
    step.technique_cls = technique_cls
    step.recipe_kwargs = {}
    step.dataset_cls = dataset_cls
    step.dataset_kwargs = {}

    dataset_cls.calls = []
    with patch("GenAILab.bench.recipe_chain.GPUMeter"):
        apply_quantization_chain(
            [step],
            MagicMock(),
            generator,
            MagicMock(spec=ProcessorMixin),
            context_length=32,
            image_size=image_size,
            profiler_kwargs={},
            profiler_capture_intermediate_data=False,
            framework="onnx",
            model_id="fake",
            precision=MagicMock(),
            model_kwargs={},
            audio_frames=audio_frames,
        )
    return dataset_cls.calls[0], step.dataset_kwargs


def test_image_dataset_gets_no_audio_kwargs():
    call, stored = _run(
        _ImageDataset, MagicMock(), image_size=(224, 224), audio_frames=2400
    )
    assert call == {"image_size": (224, 224)}
    assert "audio_frames" not in stored


def test_audio_dataset_gets_no_image_size():
    generator = MagicMock()
    call, stored = _run(
        _AudioDataset, generator, image_size=(224, 224), audio_frames=2400
    )
    assert call == {"audio_frames": 2400, "model": generator}
    assert "image_size" not in stored
    # The live model goes to the loader only, never into the JSON-reported kwargs.
    assert "model" not in stored
