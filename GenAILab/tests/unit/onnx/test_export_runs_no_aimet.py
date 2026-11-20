# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""The ONNX export must run no AIMET code.

The model cache key leaves AIMET out (see ``model_fingerprint``) so that an
AIMET change keeps cached exports. That is only sound while the export never
calls into AIMET. This exports a tiny model with every ``aimet_onnx`` and
``aimet_torch`` function turned into a tripwire; if the export starts depending
on AIMET, it fails here and AIMET has to join the model fingerprint.
"""

import inspect
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("onnxscript")


class _AimetCalled(AssertionError):
    pass


def _trip_aimet(monkeypatch):
    """Make every AIMET function and method raise when called.

    Methods are replaced on their class, which every importer shares. A module
    function is also bound by name in each module that did ``from aimet_x import
    f``, so every module's reference to it is replaced too, not only AIMET's own.
    Dunder methods are left alone: Python calls them for attribute lookup, class
    creation and repr whenever anything touches the object.
    """

    def tripwire(qualname):
        def tripped(*args, **kwargs):
            raise _AimetCalled(f"export called AIMET: {qualname}")

        return tripped

    aimet_modules = [
        (name, module)
        for name, module in list(sys.modules.items())
        if module is not None and name.startswith(("aimet_onnx", "aimet_torch"))
    ]
    functions = {}
    for name, module in aimet_modules:
        for attr, value in list(vars(module).items()):
            if attr.startswith("__"):
                continue
            if inspect.isfunction(value) and value.__module__ == name:
                functions[id(value)] = tripwire(f"{name}.{attr}")
            elif inspect.isclass(value) and vars(value).get("__module__") == name:
                for meth, fn in list(vars(value).items()):
                    if inspect.isfunction(fn) and not meth.startswith("__"):
                        monkeypatch.setattr(
                            value, meth, tripwire(f"{name}.{attr}.{meth}")
                        )

    for module in list(sys.modules.values()):
        if module is None:
            continue
        for attr, value in list(vars(module).items()):
            if inspect.isfunction(value) and id(value) in functions:
                monkeypatch.setattr(module, attr, functions[id(value)])
    return len(functions)


def test_tripwire_catches_imported_names(monkeypatch):
    """Control: a name ``llm.py`` imported from AIMET trips once wired."""
    from GenAILab.qai_hub_lm.backends.onnx import llm

    assert _trip_aimet(monkeypatch) > 0
    with pytest.raises(_AimetCalled, match="duplicate_shared_initializers"):
        llm.duplicate_shared_initializers(None)


def test_llm_export_runs_no_aimet(tmp_path, monkeypatch):
    from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX

    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )

    class TinyLLM(LLM_ONNX):
        @classmethod
        def instantiate_model(cls, model_id, small_model=False):
            torch.manual_seed(0)
            return transformers.AutoModelForCausalLM.from_config(config).eval()

    _trip_aimet(monkeypatch)
    entry = TinyLLM._export_to_cache_entry(
        "tiny", 32, 8, small_model=False, directory=str(tmp_path)
    )
    assert entry.backbone.graph.node
