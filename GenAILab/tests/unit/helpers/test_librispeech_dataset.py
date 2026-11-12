# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the LibriSpeech ASR dataset.

The raw HuggingFace dataset and the ASR processor are both mocked: nothing here
downloads audio or loads a model.
"""

import numpy as np
import pytest
import torch

from GenAILab.bench.datasets import (
    AUDIO_SAMPLING_RATE,
    Interleaved,
    LazyLibriSpeechDataset,
    LibriSpeech,
    MultimodalDataset,
    decode_audio_column,
)
from GenAILab.bench.yaml_config_parser import YAMLConfigParser


class _FakeFeatureExtractor:
    def __init__(self, n_window=50):
        self.n_window = n_window


class _FakeASRProcessor:
    """Stands in for Qwen3ASRProcessor.

    ``apply_transcription_request`` records its kwargs and returns fused inputs
    whose mel axis is a multiple of ``2 * n_window`` (as the real feature
    extractor guarantees), unless ``mel_frames`` is overridden.
    """

    def __init__(self, n_window=50, mel_frames=200, num_tokens=32):
        self.feature_extractor = _FakeFeatureExtractor(n_window)
        self.tokenizer = object()
        self.mel_frames = mel_frames
        self.num_tokens = num_tokens
        self.calls = []

    def apply_transcription_request(self, audio, language=None, **kwargs):
        audio_kwargs = (kwargs.get("processor_kwargs") or {}).get("audio_kwargs", {})
        self.calls.append(
            {"audio": audio, "language": language, **kwargs, **audio_kwargs}
        )
        return {
            "input_ids": torch.zeros((1, self.num_tokens), dtype=torch.long),
            "attention_mask": torch.ones((1, self.num_tokens), dtype=torch.long),
            "input_features": torch.zeros((1, 128, self.mel_frames)),
            "input_features_mask": torch.ones((1, self.mel_frames), dtype=torch.long),
            # An extra key the model does not take: must not be forwarded.
            "num_audio_tokens": torch.tensor([13]),
        }


def _raw_dataset(num_items=2, seconds=1.0, sampling_rate=AUDIO_SAMPLING_RATE):
    return [
        {
            "audio": {
                "array": np.zeros(int(seconds * sampling_rate), dtype=np.float32),
                "sampling_rate": sampling_rate,
            },
            "text": f"HELLO WORLD {i}",
        }
        for i in range(num_items)
    ]


class TestDecodeAudioColumn:
    def test_decoded_dict_form(self):
        array, rate = decode_audio_column(
            {"array": np.arange(4, dtype=np.float32), "sampling_rate": 16000}
        )
        assert rate == 16000
        assert array.dtype == np.float32
        assert array.tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_torchcodec_style_decoder(self):
        class _Samples:
            data = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
            sample_rate = 16000

        class _Decoder:
            def get_all_samples(self):
                return _Samples()

        array, rate = decode_audio_column(_Decoder())
        assert rate == 16000
        assert array.tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_multi_channel_is_averaged_to_mono(self):
        stereo = np.array(
            [[0.0, 2.0, 4.0, 6.0], [2.0, 2.0, 2.0, 2.0]], dtype=np.float32
        )
        array, _ = decode_audio_column({"array": stereo, "sampling_rate": 16000})
        assert array.shape == (4,)
        assert array.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_bare_array_assumes_target_rate(self):
        array, rate = decode_audio_column(np.zeros(8, dtype=np.float32))
        assert rate == AUDIO_SAMPLING_RATE
        assert array.shape == (8,)

    def test_sampling_rate_mismatch_is_an_error_not_a_silent_resample(self):
        with pytest.raises(ValueError, match="sampling rate"):
            decode_audio_column(
                {"array": np.zeros(8, dtype=np.float32), "sampling_rate": 8000}
            )

    def test_undecoded_value_without_bytes_or_path_is_an_error(self):
        with pytest.raises(ValueError, match="neither 'bytes' nor 'path'"):
            decode_audio_column({"bytes": None, "path": None})

    def test_none_is_an_error(self):
        with pytest.raises(ValueError, match="Could not extract"):
            decode_audio_column(None)


class TestLazyLibriSpeechDataset:
    def test_yields_only_the_fused_model_inputs(self):
        processor = _FakeASRProcessor()
        dataset = LazyLibriSpeechDataset(_raw_dataset(), processor, context_length=512)
        item = dataset[0]
        assert set(item) == {
            "input_ids",
            "attention_mask",
            "input_features",
            "input_features_mask",
        }
        assert item["input_features"].shape == (1, 128, 200)
        assert item["input_features_mask"].shape == (1, 200)

    def test_calibration_items_are_pure_tensors(self):
        processor = _FakeASRProcessor()
        dataset = LazyLibriSpeechDataset(_raw_dataset(), processor, context_length=512)
        assert all(isinstance(v, torch.Tensor) for v in dataset[0].values())

    def test_include_reference_attaches_the_transcript(self):
        processor = _FakeASRProcessor()
        dataset = LazyLibriSpeechDataset(
            _raw_dataset(), processor, context_length=512, include_reference=True
        )
        assert dataset[0]["reference"] == "HELLO WORLD 0"
        assert dataset[1]["reference"] == "HELLO WORLD 1"

    def test_len_and_iteration(self):
        processor = _FakeASRProcessor()
        dataset = LazyLibriSpeechDataset(_raw_dataset(3), processor, context_length=512)
        assert len(dataset) == 3
        assert len(list(dataset)) == 3

    def test_waveform_and_rate_are_passed_to_the_processor(self):
        processor = _FakeASRProcessor()
        dataset = LazyLibriSpeechDataset(
            _raw_dataset(seconds=0.5), processor, context_length=512
        )
        dataset[0]
        call = processor.calls[0]
        assert call["sampling_rate"] == AUDIO_SAMPLING_RATE
        assert call["language"] == "English"
        assert len(call["audio"]) == 1
        assert call["audio"][0].shape == (8000,)

    def test_n_window_defaults_to_the_processor_and_is_forwarded(self):
        processor = _FakeASRProcessor(n_window=25, mel_frames=200)
        dataset = LazyLibriSpeechDataset(_raw_dataset(), processor, context_length=512)
        assert dataset.n_window == 25
        assert dataset.mel_frame_multiple == 50
        dataset[0]
        assert processor.calls[0]["processor_kwargs"]["audio_kwargs"]["n_window"] == 25
        assert processor.calls[0]["n_window"] == 25

    def test_n_window_override_wins_over_the_processor(self):
        processor = _FakeASRProcessor(n_window=25)
        dataset = LazyLibriSpeechDataset(
            _raw_dataset(), processor, context_length=512, n_window=50
        )
        assert dataset.mel_frame_multiple == 100
        dataset[0]
        assert processor.calls[0]["processor_kwargs"]["audio_kwargs"]["n_window"] == 50
        assert processor.calls[0]["n_window"] == 50

    def test_default_n_window_when_processor_has_none(self):
        class _Bare:
            def apply_transcription_request(self, audio, language=None, **kwargs):
                raise AssertionError("not called")

        dataset = LazyLibriSpeechDataset(_raw_dataset(), _Bare(), context_length=512)
        assert dataset.n_window == 50
        assert dataset.mel_frame_multiple == 100

    def test_mel_frames_not_a_multiple_of_2n_window_is_rejected(self):
        # 150 frames is not a multiple of 2 * 50 = 100.
        processor = _FakeASRProcessor(n_window=50, mel_frames=150)
        dataset = LazyLibriSpeechDataset(_raw_dataset(), processor, context_length=512)
        with pytest.raises(ValueError, match="not a multiple of"):
            dataset[0]

    def test_mel_frames_multiple_of_100_is_accepted(self):
        for frames in (100, 200, 3000):
            processor = _FakeASRProcessor(n_window=50, mel_frames=frames)
            dataset = LazyLibriSpeechDataset(
                _raw_dataset(), processor, context_length=512
            )
            assert dataset[0]["input_features"].shape[-1] == frames

    def test_over_context_length_warns_but_does_not_truncate(self):
        processor = _FakeASRProcessor(num_tokens=600)
        dataset = LazyLibriSpeechDataset(_raw_dataset(2), processor, context_length=512)
        with pytest.warns(UserWarning, match="exceeding the context length"):
            item = dataset[0]
        # Truncating would desynchronise audio placeholders from mel frames.
        assert item["input_ids"].shape[-1] == 600

    def test_over_context_warning_is_emitted_once(self):
        import warnings as _warnings

        processor = _FakeASRProcessor(num_tokens=600)
        dataset = LazyLibriSpeechDataset(_raw_dataset(2), processor, context_length=512)
        with pytest.warns(UserWarning):
            dataset[0]
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            dataset[1]  # must not warn again


class TestLibriSpeechWrapper:
    def test_is_a_multimodal_dataset_and_is_registered(self):
        assert issubclass(LibriSpeech, MultimodalDataset)
        assert YAMLConfigParser.get_dataset("LibriSpeech") is LibriSpeech

    def test_source_is_pinned(self):
        assert LibriSpeech.REPO_ID == "openslr/librispeech_asr"
        assert LibriSpeech.CONFIG == "all"
        # A pinned 40-hex commit sha, not a branch name.
        assert len(LibriSpeech.REVISION) == 40
        assert all(c in "0123456789abcdef" for c in LibriSpeech.REVISION)

    def test_load_dataset_pins_repo_config_and_revision(self, monkeypatch):
        captured = {}

        class _Fake:
            column_names = ["file", "audio", "text", "speaker_id", "id"]

            def __init__(self, items):
                self.items = items

            def cast_column(self, name, feature):
                captured["cast"] = (name, feature)
                return self

            def select(self, indices):
                return _Fake([self.items[i] for i in indices])

            def __len__(self):
                return len(self.items)

        def fake_load_dataset(
            repo, data_files=None, split=None, revision=None, verification_mode=None
        ):
            captured.update(
                repo=repo,
                data_files=data_files,
                split=split,
                revision=revision,
                verification_mode=verification_mode,
            )
            return _Fake(_raw_dataset(5))

        monkeypatch.setattr("GenAILab.bench.datasets.load_dataset", fake_load_dataset)
        raw = LibriSpeech.load_dataset("test.clean")
        assert captured["repo"] == "openslr/librispeech_asr"
        # Scoped to this split's parquet shard(s) only, so `datasets` never
        # downloads/prepares the other splits of the 'all' config.
        assert captured["data_files"] == {"test.clean": "all/test.clean/*.parquet"}
        # The repo's dataset_infos.json declares every split; skip the split check.
        assert captured["verification_mode"] == "no_checks"
        assert captured["split"] == "test.clean"
        assert captured["revision"] == LibriSpeech.REVISION
        # Audio decoding is done by decode_audio_column, not by `datasets`.
        assert captured["cast"][0] == "audio"
        assert captured["cast"][1].decode is False
        assert len(raw) == 5

    def test_load_dataset_rejects_missing_columns(self, monkeypatch):
        # verification_mode="no_checks" would let a re-laid-out repo load; the
        # column check makes it fail at the load site instead.
        class _NoText:
            column_names = ["file", "audio"]

        monkeypatch.setattr(
            "GenAILab.bench.datasets.load_dataset", lambda *a, **k: _NoText()
        )
        with pytest.raises(ValueError, match=r"missing column\(s\) \['text'\]"):
            LibriSpeech.load_dataset("test.clean")

    def test_num_samples_takes_the_head_of_the_split(self, monkeypatch):
        class _Fake:
            column_names = ["file", "audio", "text", "speaker_id", "id"]

            def __init__(self, items):
                self.items = items

            def cast_column(self, name, feature):
                return self

            def select(self, indices):
                return _Fake([self.items[i] for i in indices])

            def __len__(self):
                return len(self.items)

        monkeypatch.setattr(
            "GenAILab.bench.datasets.load_dataset",
            lambda *a, **k: _Fake(_raw_dataset(5)),
        )
        assert len(LibriSpeech.load_dataset("test.clean", num_samples=3)) == 3
        # Asking for more than exists is clamped, not an error.
        assert len(LibriSpeech.load_dataset("test.clean", num_samples=99)) == 5

    def test_load_encoded_dataset_defaults_to_calibration_safe_items(self, monkeypatch):
        monkeypatch.setattr(
            LibriSpeech, "load_dataset", staticmethod(lambda *a, **k: _raw_dataset(2))
        )
        dataset = LibriSpeech.load_encoded_dataset(
            _FakeASRProcessor(), context_length=512, split="validation.clean"
        )
        assert isinstance(dataset, LazyLibriSpeechDataset)
        assert dataset.include_reference is False
        assert "reference" not in dataset[0]

    def test_load_encoded_dataset_can_request_references(self, monkeypatch):
        monkeypatch.setattr(
            LibriSpeech, "load_dataset", staticmethod(lambda *a, **k: _raw_dataset(2))
        )
        dataset = LibriSpeech.load_encoded_dataset(
            _FakeASRProcessor(),
            context_length=512,
            split="test.clean",
            include_reference=True,
        )
        assert dataset[0]["reference"] == "HELLO WORLD 0"


class _FakeChatProcessor:
    """Gemma4-shaped: no ``apply_transcription_request``, so the dataset takes
    the generic ``apply_chat_template`` branch."""

    def __init__(self, mel_frames=200, num_tokens=32):
        self.tokenizer = object()
        self.mel_frames = mel_frames
        self.num_tokens = num_tokens
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": torch.zeros((1, self.num_tokens), dtype=torch.long),
            "attention_mask": torch.ones((1, self.num_tokens), dtype=torch.long),
            # Gemma4 layout: (B, time, mel).
            "input_features": torch.zeros((1, self.mel_frames, 128)),
            "input_features_mask": torch.ones((1, self.mel_frames), dtype=torch.long),
        }


class TestGenericChatTemplateBranch:
    """The non-ASR path (Gemma4). Both pin bugs that produced a flat WER of 100
    rather than an error."""

    def test_opens_an_assistant_turn(self):
        processor = _FakeChatProcessor()
        dataset = LazyLibriSpeechDataset(_raw_dataset(), processor, context_length=512)
        dataset[0]
        assert processor.calls[0]["add_generation_prompt"] is True

    def test_audio_kwargs_are_nested_for_the_processor(self):
        processor = _FakeChatProcessor()
        dataset = LazyLibriSpeechDataset(_raw_dataset(), processor, context_length=512)
        dataset[0]
        audio_kwargs = processor.calls[0]["processor_kwargs"]["audio_kwargs"]
        assert audio_kwargs["sampling_rate"] == AUDIO_SAMPLING_RATE
        assert "sampling_rate" not in processor.calls[0]


class TestInterleavedForwarding:
    def test_audio_kwargs_reach_a_librispeech_sub_dataset(self, monkeypatch):
        monkeypatch.setattr(
            LibriSpeech, "load_dataset", staticmethod(lambda *a, **k: _raw_dataset(2))
        )
        model = object()
        wrapper = Interleaved.load_encoded_dataset(
            _FakeASRProcessor(),
            context_length=512,
            source_datasets=[{"name": "LibriSpeech", "split": "validation.clean"}],
            audio_frames=3000,
            model=model,
            # Not a LibriSpeech kwarg: must be dropped, not raise TypeError.
            image_size=(224, 224),
        )
        (sub,) = wrapper.datasets
        assert isinstance(sub, LazyLibriSpeechDataset)
        assert sub.audio_frames == 3000
        assert sub.model is model

    def test_sub_dataset_config_wins_over_parent_kwargs(self, monkeypatch):
        monkeypatch.setattr(
            LibriSpeech, "load_dataset", staticmethod(lambda *a, **k: _raw_dataset(2))
        )
        wrapper = Interleaved.load_encoded_dataset(
            _FakeASRProcessor(),
            context_length=512,
            source_datasets=[
                {
                    "name": "LibriSpeech",
                    "split": "validation.clean",
                    "audio_frames": 1000,
                }
            ],
            audio_frames=3000,
            model=object(),
        )
        assert wrapper.datasets[0].audio_frames == 1000
