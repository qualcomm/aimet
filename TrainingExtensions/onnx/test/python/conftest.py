# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import platform
import sys

import pytest


# TODO: #6523 Update AttentionMaskConverter in transformers to new transformers.masking_utils
class TransformersDeprecationFilter(logging.Filter):
    """Filter to suppress malformed deprecation warnings from transformers.modeling_attn_mask_utils.

    Transformers 5.x has a bug where logger.warning_once() is called with
    (message, FutureWarning) but the message has no % placeholders,
    causing 'TypeError: not all arguments converted during string formatting'.
    """

    def filter(self, record):
        if record.name == "transformers.modeling_attn_mask_utils":
            return False
        return True


# TODO: #6523 Update AttentionMaskConverter in transformers to new transformers.masking_utils
def pytest_configure(config):
    # Add filter to suppress malformed transformers deprecation warnings
    logging.getLogger("transformers.modeling_attn_mask_utils").addFilter(
        TransformersDeprecationFilter()
    )
    config.addinivalue_line(
        "markers",
        "skip_on_windows_arm64(reason): skip test on Windows ARM64 with specified reason",
    )
    config.addinivalue_line(
        "markers",
        "skip_on_windows_amd64(reason): skip test on Windows AMD64 with specified reason",
    )
    config.addinivalue_line(
        "markers",
        "skip_on_macos(reason): skip test on MacOS with specified reason",
    )


def _is_windows_arm64():
    return sys.platform == "win32" and platform.machine().lower() in (
        "aarch64",
        "arm64",
    )


def _is_windows_amd64():
    return sys.platform == "win32" and platform.machine().lower() in (
        "amd64",
        "x86_64",
    )


def _is_macos():
    return sys.platform == "darwin" and platform.machine().lower() in (
        "aarch64",
        "arm64",
    )


@pytest.fixture(autouse=True)
def skip_on_windows_arm64(request):
    marker = request.node.get_closest_marker("skip_on_windows_arm64")
    if marker is not None:
        if _is_windows_arm64():
            reason = marker.args[0] if marker.args else "Not supported on Windows ARM64"
            pytest.skip(reason)


@pytest.fixture(autouse=True)
def skip_on_windows_amd64(request):
    marker = request.node.get_closest_marker("skip_on_windows_amd64")
    if marker is not None:
        if _is_windows_amd64():
            reason = marker.args[0] if marker.args else "Not supported on Windows AMD64"
            pytest.skip(reason)


@pytest.fixture(autouse=True)
def skip_on_macos(request):
    marker = request.node.get_closest_marker("skip_on_macos")
    if marker is not None:
        if _is_macos():
            reason = marker.args[0] if marker.args else "Not supported on MacOS"
            pytest.skip(reason)


def skip_module_on_windows_arm64(reason):
    if _is_windows_arm64():
        pytest.skip(allow_module_level=True, reason=reason)


def skip_module_on_macos(reason):
    if _is_macos():
        pytest.skip(allow_module_level=True, reason=reason)
