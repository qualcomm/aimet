# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated alias of :mod:`aimet_onnx.llm_configurator`, to be removed in AIMET 2.45"""

import sys

from .._deprecation import _alias_deprecated_package

sys.modules[__name__] = _alias_deprecated_package(
    __name__, "aimet_onnx.llm_configurator", removal_version="2.45"
)
