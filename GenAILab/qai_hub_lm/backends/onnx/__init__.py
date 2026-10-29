# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI ONNX models"""

# Register the default ONNX LLM class
from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX

# Register shared adaptations
from GenAILab.qai_hub_lm import transforms as shared_adaptations  # noqa: F401

# Register ONNX-specific adaptations
from GenAILab.qai_hub_lm.backends.onnx import adaptations  # noqa: F401

# VLM models (registered via the shared VLM backend)
from GenAILab.qai_hub_lm.backends.onnx.vlm import *  # noqa: F401, F403

# LLM models that require special handling
try:
    from GenAILab.qai_hub_lm.backends.onnx.qwen3_5 import Qwen3_5
except ImportError:
    warnings.warn(
        "Qwen 3.5 is not available. Please upgrade to a later version of transformers to use this model."
    )

from GenAILab.qai_hub_lm.backends.onnx.qwen4_exp import Qwen4Exp  # noqa: F401
