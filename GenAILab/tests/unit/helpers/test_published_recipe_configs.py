# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Guard that every GenAI Lab config, including the ones published in the LLM
quantization recipe docs, still parses."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIGS_DIR = REPO_ROOT / "GenAILab" / "configs"
FRAMEWORKS = ("onnx", "torch")

# ``config.json`` ``model_type`` of every published ``model_id``; it selects the
# model class and the required export adaptations. A new model needs an entry.
MODEL_TYPES = {
    "google/gemma-3-1b-it": "gemma3_text",
    "google/gemma-3-270m-it": "gemma3_text",
    "google/gemma-3-4b-it": "gemma3",
    "google/gemma-4-E4B-it": "gemma4",
    "meta-llama/Llama-3.2-1B-Instruct": "llama",
    "meta-llama/Llama-3.2-3B-Instruct": "llama",
    "microsoft/Phi-3.5-mini-instruct": "phi3",
    "Qwen/Qwen2.5-0.5B-Instruct": "qwen2",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2",
    "Qwen/Qwen3-0.6B": "qwen3",
    "Qwen/Qwen3-1.7B": "qwen3",
    "Qwen/Qwen3-4B": "qwen3",
    "Qwen/Qwen3-8B": "qwen3",
    "Qwen/Qwen3.5-4B": "qwen3_5",
    "Qwen/Qwen3-VL-8B-Instruct": "qwen3_vl",
}

# Both backends register under the same names in one global registry, so each
# framework is parsed in its own interpreter, as ``python -m GenAILab`` runs them.
_PARSE_SCRIPT = """
import json, sys, tempfile, traceback
import importlib
importlib.import_module(f"GenAILab.bench.{sys.argv[1]}.run_genai")
from GenAILab.bench.yaml_config_parser import YAMLConfigParser

model_types = json.loads(sys.argv[2])
def detect_model_type(cls, model_id):
    if model_id not in model_types:
        raise KeyError(f"{model_id} missing from MODEL_TYPES in test_published_recipe_configs.py")
    return model_types[model_id]
YAMLConfigParser.detect_model_type = classmethod(detect_model_type)

errors = {}
for path in sys.argv[3:]:
    try:
        with tempfile.TemporaryDirectory() as export_dir:
            list(YAMLConfigParser.parse(path, export_base_dir=export_dir))
        errors[path] = None
    except Exception:
        errors[path] = traceback.format_exc()
print(json.dumps(errors))
"""


def _configs(framework):
    """Regression configs (``<framework>_*.yaml``) run on their own framework; the
    published recipe configs are shared, so they run on both."""
    configs = []
    for path in sorted(CONFIGS_DIR.rglob("*.yaml")):
        prefix = path.name.split("_")[0]
        if prefix == framework or prefix not in FRAMEWORKS:
            configs.append(path)
    return configs


@pytest.fixture(scope="module")
def parse_errors():
    """Parse every config once per framework: ``{framework: {path: traceback | None}}``."""
    errors = {}
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p
    )
    env["HF_HUB_OFFLINE"] = "1"  # fail on any Hub access not covered by MODEL_TYPES
    for framework in FRAMEWORKS:
        result = subprocess.run(
            [sys.executable, "-c", _PARSE_SCRIPT, framework, json.dumps(MODEL_TYPES)]
            + [str(p) for p in _configs(framework)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        errors[framework] = json.loads(result.stdout.strip().splitlines()[-1])
    return errors


@pytest.mark.parametrize(
    "framework, config",
    [
        pytest.param(framework, p, id=f"{framework}-{p.relative_to(CONFIGS_DIR)}")
        for framework in FRAMEWORKS
        for p in _configs(framework)
    ],
)
def test_config_parses(framework, config, parse_errors):
    error = parse_errors[framework][str(config)]
    assert error is None, (
        f"{config.relative_to(REPO_ROOT)} no longer parses with --framework {framework}. "
        f"Update the config (and the recipe docs table, if it is linked there):\n{error}"
    )
