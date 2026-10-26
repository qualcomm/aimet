# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import tempfile
import torch
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@pytest.fixture
def tmp_dir():
    """
    Pytest fixture to create and yield a temporary directory.
    The directory is automatically cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def add_genai_tests_path():
    """
    Pytest fixture to add the GenAILab directory to sys.path.
    """
    path = os.path.abspath(os.path.join(Path(__file__).parent, "../../../../"))
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(path)
        yield


@contextmanager
def patch_fuse_supergroups(enabled: bool):
    """
    Context manager that patches ``aimet_onnx.quantsim._fuse_supergroups`` to the
    given value for the duration of the block.
    """
    with patch("aimet_onnx.quantsim._fuse_supergroups", enabled):
        yield


@contextmanager
def force_random_weight_init(vocab_size: int | None = None):
    """Patch ``AutoModelForCausalLM.from_pretrained`` to build from config with set vocab size."""
    from transformers import AutoModelForCausalLM
    from GenAILab.qai_hub_lm.models.utils.layer_cache import _resolve_text_config

    def _from_config_stub(model_id, config=None, **kwargs):
        # Find decoder vocab_size if nested
        text_config = _resolve_text_config(config)
        if vocab_size is not None:
            text_config.vocab_size = vocab_size
        return AutoModelForCausalLM.from_config(text_config)

    with patch.object(AutoModelForCausalLM, "from_pretrained", new=_from_config_stub):
        yield


_HF_CONFIG_DIR = Path(__file__).parent / "models" / "hf_configs"


def _vendored_config_dir(model_id) -> Path | None:
    """Directory holding the verbatim ``config.json`` of HF repo ``model_id``, if vendored."""
    if not isinstance(model_id, str):
        return None
    path = _HF_CONFIG_DIR / model_id.replace("/", "--")
    return path if (path / "config.json").is_file() else None


@pytest.fixture
def offline_hf_configs(add_genai_tests_path, monkeypatch):
    """
    Serve ``config.json`` (and a stub tokenizer) of vendored HF repos from disk.

    The configs under ``models/hf_configs`` are verbatim copies of the Hub files, so
    model structure is identical to what the Hub would give. Combine with
    ``force_random_weight_init`` so that neither config nor weights hit the network.
    """
    from transformers import AutoConfig
    from GenAILab.qai_hub_lm.models.base import LLM

    real_from_pretrained = AutoConfig.from_pretrained

    def _config_from_pretrained(model_id, *args, **kwargs):
        local = _vendored_config_dir(model_id)
        return real_from_pretrained(local if local else model_id, *args, **kwargs)

    def _instantiate_tokenizer(model_id):
        local = _vendored_config_dir(model_id)
        assert local is not None, f"No vendored config for {model_id}"
        config = real_from_pretrained(local)
        return SimpleNamespace(eos_token_id=config.eos_token_id)

    def _is_huggingface_ckpt(model_id):
        return _vendored_config_dir(model_id) is not None

    monkeypatch.setattr(AutoConfig, "from_pretrained", _config_from_pretrained)
    monkeypatch.setattr(
        LLM, "instantiate_tokenizer", staticmethod(_instantiate_tokenizer)
    )
    # Treat vendored repos as HF checkpoints (it would otherwise query the Hub's model_info)
    for module in (
        "GenAILab.qai_hub_lm.backends.onnx.llm",
        "GenAILab.qai_hub_lm.backends.onnx.export_utils",
    ):
        monkeypatch.setattr(f"{module}.is_huggingface_ckpt", _is_huggingface_ckpt)


def random_chunked_dataset(vocab_size: int, context_length: int, num_chunks: int = 5):
    """Build a ``ChunkedDataset`` of random token ids, in place of a real data."""
    from GenAILab.bench.datasets import ChunkedDataset

    input_ids = torch.randint(0, vocab_size, (1, context_length * (num_chunks + 1)))
    attention_mask = torch.ones_like(input_ids)
    return ChunkedDataset(
        {"input_ids": input_ids, "attention_mask": attention_mask}, context_length
    )
