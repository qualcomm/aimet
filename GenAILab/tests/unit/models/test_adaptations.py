# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for model adaptations registration."""

from unittest.mock import patch

import pytest

from GenAILab.bench.yaml_config_parser import YAMLConfigParser


@pytest.fixture(autouse=True)
def _clean_registry():
    saved_adaptations = dict(YAMLConfigParser.adaptation_lookup)
    saved_default = YAMLConfigParser._default_llm_cls
    yield
    YAMLConfigParser.adaptation_lookup = saved_adaptations
    YAMLConfigParser._default_llm_cls = saved_default


class TestAdaptations:
    def test_adaptation_registered(self):
        @YAMLConfigParser.register_adaptation("TestAdapt", model_type="llama")
        class TestMixin:
            pass

        assert ("llama", "TestAdapt") in YAMLConfigParser.adaptation_lookup

    def test_universal_adaptation(self):
        @YAMLConfigParser.register_adaptation("Universal", model_type="*")
        class UniversalMixin:
            pass

        assert ("*", "Universal") in YAMLConfigParser.adaptation_lookup

    def test_adaptation_apply_creates_subclass(self):
        class BaseLLM:
            base_attr = True

        YAMLConfigParser._default_llm_cls = BaseLLM

        @YAMLConfigParser.register_adaptation("Mixin1", model_type="llama")
        class Mixin1:
            mixin_attr = True

        result_cls = YAMLConfigParser.get_model_class("llama", adaptations=["Mixin1"])
        assert issubclass(result_cls, BaseLLM)
        assert issubclass(result_cls, Mixin1)
        # Verify MRO order: mixin before base
        assert result_cls.__mro__.index(Mixin1) < result_cls.__mro__.index(BaseLLM)

    def test_multiple_adaptations_stacked(self):
        class BaseLLM:
            pass

        YAMLConfigParser._default_llm_cls = BaseLLM

        @YAMLConfigParser.register_adaptation("A")
        class MixinA:
            pass

        @YAMLConfigParser.register_adaptation("B")
        class MixinB:
            pass

        result_cls = YAMLConfigParser.get_model_class("llama", adaptations=["A", "B"])
        assert issubclass(result_cls, MixinA)
        assert issubclass(result_cls, MixinB)
        assert issubclass(result_cls, BaseLLM)


class TestAdaptationIdentity:
    """Adaptation kwargs become class attributes; the identity records them."""

    def test_records_names_and_kwargs_in_order(self):
        class BaseLLM:
            pass

        YAMLConfigParser._default_llm_cls = BaseLLM

        @YAMLConfigParser.register_adaptation("Scale")
        class Scale:
            pass

        @YAMLConfigParser.register_adaptation("Plain")
        class Plain:
            pass

        result_cls = YAMLConfigParser.get_model_class(
            "llama",
            adaptations=["Scale", "Plain"],
            adaptation_kwargs={"Scale": {"layer_multipliers": {0: 0.5}}},
        )
        assert result_cls.adaptation_identity == [
            ["Scale", [("layer_multipliers", {0: 0.5})]],
            ["Plain", []],
        ]

    def test_kwargs_change_the_identity(self):
        class BaseLLM:
            pass

        YAMLConfigParser._default_llm_cls = BaseLLM

        @YAMLConfigParser.register_adaptation("Scale")
        class Scale:
            pass

        def identity(multiplier):
            return YAMLConfigParser.get_model_class(
                "llama",
                adaptations=["Scale"],
                adaptation_kwargs={"Scale": {"layer_multipliers": {0: multiplier}}},
            ).adaptation_identity

        assert identity(0.5) != identity(0.25)
