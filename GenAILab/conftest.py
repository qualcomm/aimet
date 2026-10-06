# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""pytest config for test_genai.py"""

import warnings
from pathlib import Path

import onnxruntime as ort
import pytest
import yaml

# Suppress verbose onnxruntime INFO/WARNING messages (e.g. EP selection,
# graph optimisation notes) that clutter test output.  Level 3 = ERROR only.
ort.set_default_logger_severity(3)

from GenAILab.bench.fp_cache import DiskBackedFPCache
from GenAILab.bench.model_cache import DiskBackedModelCache
from GenAILab.bench.recipe_cache import RecipeCache

# All generated artifacts live under GenAILab/artifacts/ so they don't
# clutter the repo root and can be excluded from rsync with a single path.
_OUTPUT_ROOT = Path(__file__).parent / "artifacts"


def pytest_addoption(parser):
    parser.addoption("--config", action="store", default=None)
    parser.addoption("--force-export", action="store_true", default=False)
    parser.addoption(
        "--export-dir", action="store", default=str(_OUTPUT_ROOT / "exports")
    )
    parser.addoption(
        "--results-dir", action="store", default=str(_OUTPUT_ROOT / "results")
    )
    parser.addoption(
        "--fp-cache-dir", action="store", default=str(_OUTPUT_ROOT / "cache" / "fp")
    )
    parser.addoption("--clear-fp-cache", action="store_true", default=False)
    parser.addoption(
        "--model-cache-dir",
        action="store",
        default=str(_OUTPUT_ROOT / "cache" / "model"),
    )
    parser.addoption("--clear-model-cache", action="store_true", default=False)
    parser.addoption(
        "--recipe-cache-dir",
        action="store",
        default=str(_OUTPUT_ROOT / "cache" / "recipe"),
        help="Directory for recipe chain cache (use --no-recipe-cache to disable)",
    )
    parser.addoption(
        "--no-recipe-cache",
        action="store_true",
        default=False,
        help="Disable recipe chain caching",
    )
    parser.addoption(
        "--clear-recipe-cache",
        action="store_true",
        default=False,
        help="Clear the recipe cache before running",
    )


@pytest.fixture(scope="session")
def fp_cache(request):
    """Session-scoped disk-backed FP results cache.

    Shared across all parameterized test cases so that FP model outputs are
    computed at most once per (model, dataset) combination, even across
    different quantization recipes in a multi-doc config.
    """
    cache = DiskBackedFPCache(request.config.getoption("--fp-cache-dir"))
    if request.config.getoption("--clear-fp-cache"):
        cache.clear()
    return cache


@pytest.fixture(scope="session")
def model_cache(request):
    """Session-scoped disk-backed ONNX model cache.

    Shared across all parameterized test cases so that ONNX model exports are
    computed at most once per (model, sequence_length, context_length, adaptations)
    combination, even across different quantization recipes in a multi-doc config.
    """
    cache = DiskBackedModelCache(request.config.getoption("--model-cache-dir"))
    if request.config.getoption("--clear-model-cache"):
        cache.clear()
    return cache


@pytest.fixture(scope="session")
def recipe_cache(request):
    """Session-scoped recipe chain cache (enabled by default).

    Content-addressed caching of expensive recipe steps (AdaScale, SeqMSE)
    so that shared recipe prefixes across experiments are computed only once.
    Disable with --no-recipe-cache.
    """
    if request.config.getoption("--no-recipe-cache"):
        return None
    cache_dir = request.config.getoption("--recipe-cache-dir")
    cache = RecipeCache(cache_dir)
    if request.config.getoption("--clear-recipe-cache"):
        cache.clear()
    return cache


@pytest.fixture(scope="session")
def export_dir(request):
    """Base directory for exported model artifacts."""
    path = Path(request.config.getoption("--export-dir"))
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@pytest.fixture(scope="session")
def results_dir(request):
    """Directory for global profiling results (profiling_data.json/csv)."""
    path = Path(request.config.getoption("--results-dir"))
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def pytest_generate_tests(metafunc):
    if "test_config" not in metafunc.fixturenames:
        return

    config_file = metafunc.config.getoption("--config", skip=False)
    force_export = metafunc.config.getoption("--force-export", skip=False)

    if not config_file:
        raise ValueError(
            "No config file provided. Please specify a config file using the --config option,"
        )

    with open(config_file, "r") as f:
        docs = list(yaml.safe_load_all(f))
    if force_export:
        for doc in docs:
            if doc.get("export") and doc["export"] != True:
                warnings.warn(
                    "force_export is True, all artifacts will be saved to default export path."
                )
            doc["export"] = True
    test_configs = docs

    metafunc.parametrize("test_config", test_configs)
