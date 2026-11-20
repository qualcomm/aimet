# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""``SUPPORTED_LANGUAGES``: the ASR model's forced-language vocabulary."""

from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator
from GenAILab.qai_hub_lm.models.qwen3_asr import Qwen3ASRGenerator

#: Qwen/Qwen3-ASR-0.6B model card, "Supported Languages" row (README.md:39 @
#: 5eb144179a02acc5e5ba31e748d22b0cf3e303b0), minus its 22 Chinese dialects.
_CARD_LANGUAGES = {
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai",
    "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish",
    "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
    "Hungarian", "Macedonian", "Romanian",
}  # fmt: skip


def test_only_the_asr_generator_declares_languages():
    assert not hasattr(Generator, "SUPPORTED_LANGUAGES")
    assert not hasattr(VLM_Generator, "SUPPORTED_LANGUAGES")


def test_qwen3_asr_languages_are_the_model_card_list():
    # Sourced from the transformers processor; pinned here so an upstream change
    # to its table fails loudly.
    languages = Qwen3ASRGenerator.SUPPORTED_LANGUAGES
    assert set(languages.values()) == _CARD_LANGUAGES
    assert languages["zh"] == "Chinese"
    assert languages["yue"] == "Cantonese"
