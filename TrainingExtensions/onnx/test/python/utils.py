# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest
import tempfile
import torch
from contextlib import contextmanager
from pathlib import Path
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


def random_chunked_dataset(vocab_size: int, context_length: int, num_chunks: int = 5):
    """Build a ``ChunkedDataset`` of random token ids, in place of a real data."""
    from GenAILab.bench.datasets import ChunkedDataset

    input_ids = torch.randint(0, vocab_size, (1, context_length * (num_chunks + 1)))
    attention_mask = torch.ones_like(input_ids)
    return ChunkedDataset(
        {"input_ids": input_ids, "attention_mask": attention_mask}, context_length
    )
