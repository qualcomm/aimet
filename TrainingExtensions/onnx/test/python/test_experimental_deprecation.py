# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the deprecated aimet_onnx.experimental aliases of graduated packages"""

import importlib
import sys

import pytest

GRADUATED = (
    ("adascale", "adascale_optimizer"),
    ("spinquant", "passes.r1"),
    ("llm_topology", "topology"),
    ("llm_configurator", "llm_configurator"),
)


@pytest.fixture
def fresh_alias(request):
    """Drop any cached alias so the shim (and its warning) runs again."""
    old_name = f"aimet_onnx.experimental.{request.param}"
    for name in list(sys.modules):
        if name == old_name or name.startswith(f"{old_name}."):
            del sys.modules[name]
    return request.param


@pytest.mark.parametrize(
    "fresh_alias, submodule",
    GRADUATED,
    indirect=["fresh_alias"],
)
def test_deprecated_alias(fresh_alias, submodule):
    old_name = f"aimet_onnx.experimental.{fresh_alias}"
    new_name = f"aimet_onnx.{fresh_alias}"

    with pytest.warns(DeprecationWarning, match="AIMET 2.45"):
        old_pkg = importlib.import_module(old_name)

    assert old_pkg is importlib.import_module(new_name)
    assert importlib.import_module(
        f"{old_name}.{submodule}"
    ) is importlib.import_module(f"{new_name}.{submodule}")
