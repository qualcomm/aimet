# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen4-Exp model class"""

from GenAILab.qai_hub_lm.backends.onnx.llm import LLM_ONNX
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.models.qwen4_exp import Qwen4ExpMixin


@YAMLConfigParser.register_model("qwen4_exp_text")
@YAMLConfigParser.register_model("qwen4_exp")
class Qwen4Exp(Qwen4ExpMixin, LLM_ONNX):
    pass
