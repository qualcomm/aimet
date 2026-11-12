# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import io
import os
import onnx
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
    from GenAILab.qai_hub_lm.models.utils.layer_cache import resolve_text_config

    def _from_config_stub(model_id, config=None, **kwargs):
        # Find decoder vocab_size if nested
        text_config = resolve_text_config(config)
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


@contextmanager
def patch_sdpa_mask():
    """Make HF mask construction traceable under torch.jit.trace."""
    import transformers.masking_utils as mu

    orig = mu.sdpa_mask
    orig_registered = mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]

    def patched(*args, **kwargs):
        if (
            "q_length" in kwargs
            and isinstance(kwargs["q_length"], torch.Tensor)
            and kwargs["q_length"].ndim == 0
        ):
            kwargs["q_length"] = kwargs["q_length"].item()
        elif len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].ndim == 0:
            args = (args[0], args[1].item(), *args[2:])
        return orig(*args, **kwargs)

    try:
        mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = patched
        mu.sdpa_mask = patched
        yield
    finally:
        mu.sdpa_mask = orig
        mu.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] = orig_registered


def make_tiny_decoder_config(config_attr, num_hidden_layers=2, **extra):
    """Build a tiny decoder config, or skip if the architecture is unavailable."""
    import transformers

    cfg_cls = getattr(transformers, config_attr, None)
    if cfg_cls is None:
        pytest.skip(f"{config_attr} not available in this transformers version")

    cfg = cfg_cls(**extra)
    overrides = dict(
        num_hidden_layers=num_hidden_layers,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1024,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        # MoE knobs (best-effort; ignored by dense configs)
        num_experts=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        decoder_sparse_step=1,
    )
    for key, value in overrides.items():
        if hasattr(cfg, key):
            try:
                setattr(cfg, key, value)
            except (AttributeError, TypeError):
                pass
    return cfg


class PrefillWrapper(torch.nn.Module):
    """input_ids -> logits prefill wrapper for ONNX export.

    NOTE: ``use_cache=False`` keeps the graph to a prefill pass,
    block detection needs no KV-cache I/O.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits


def sample_prefill_inputs(cfg, seq_len=8):
    """Build dummy (input_ids, position_ids, attention_mask) for a prefill pass."""
    input_ids = torch.randint(3, cfg.vocab_size, (1, seq_len), dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    return input_ids, position_ids, attention_mask


def export_onnx(model, inputs, backend, input_names=None):
    """Export the wrapped model to ONNX via the torchscript or dynamo backend."""
    if input_names is None:
        input_names = ["input_ids", "position_ids", "attention_mask"]
    buf = io.BytesIO()
    with patch_sdpa_mask(), torch.no_grad():
        if backend == "dynamo":
            program = torch.export.draft_export(model, inputs, strict=False)
            torch.onnx.export(program, (), buf, input_names=input_names, dynamo=True)
        else:
            torch.onnx.export(
                model,
                inputs,
                buf,
                input_names=input_names,
                opset_version=17,
                dynamo=False,
            )
    buf.seek(0)
    return onnx.load_model(buf)
