# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion config parsing."""

import json

import pytest

from GenAILab.bench.diffusion import metrics as diffusion_metrics  # noqa: F401 — registration
from GenAILab.bench.diffusion.config import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    parse_diffusion_document,
)
from GenAILab.bench.diffusion.experiment import DiffusionEvaluationContext
from GenAILab.bench.fp_cache import DiskBackedFPCache
from GenAILab.qai_hub_lm.diffusion import StableDiffusion


@pytest.fixture
def sd_checkpoint(tmp_path):
    """A local directory that looks like an SD pipeline to model_index lookups."""
    (tmp_path / "model_index.json").write_text(
        json.dumps(
            {"_class_name": "StableDiffusionPipeline", "_diffusers_version": "0.37.1"}
        )
    )
    return str(tmp_path)


def _doc(model_id, **model_overrides):
    return {
        "model": {"model_id": model_id, **model_overrides},
        "metrics": [{"name": "CLIPScore", "num_prompts": 4}],
    }


class TestParseDiffusionDocument:
    def test_defaults(self, sd_checkpoint):
        config = parse_diffusion_document(_doc(sd_checkpoint))
        model = config.model
        assert model.model_cls is StableDiffusion
        assert model.model_type == "StableDiffusionPipeline"
        assert model.dtype == "float32"
        assert model.num_inference_steps == DEFAULT_NUM_INFERENCE_STEPS == 10
        assert model.guidance_scale == DEFAULT_GUIDANCE_SCALE
        assert model.height is None and model.scheduler is None
        assert [m.name for m in config.metrics] == ["CLIPScore"]
        assert config.metrics[0].metric_kwargs == {"num_prompts": 4}

    def test_all_model_fields(self, sd_checkpoint):
        model = parse_diffusion_document(
            _doc(
                sd_checkpoint,
                dtype="float16",
                height=512,
                width=768,
                num_inference_steps=25,
                guidance_scale=5,
                negative_prompt="blurry",
                scheduler="EulerDiscreteScheduler",
                seed=11,
            )
        ).model
        assert (model.height, model.width, model.num_inference_steps) == (512, 768, 25)
        assert model.guidance_scale == 5.0 and isinstance(model.guidance_scale, float)
        assert model.report_modifiers() == {
            "dtype": "float16",
            "num_inference_steps": 25,
            "guidance_scale": 5.0,
            "seed": 11,
            "height": 512,
            "width": 768,
            "scheduler": "EulerDiscreteScheduler",
            "negative_prompt": "blurry",
        }

    @pytest.mark.parametrize(
        "section",
        ["precision", "recipe", "dataset", "export", "eval_in_onnx", "analysis"],
    )
    def test_rejects_unsupported_sections(self, sd_checkpoint, section):
        with pytest.raises(ValueError, match=f"'{section}' is not supported"):
            parse_diffusion_document({**_doc(sd_checkpoint), section: {}})

    @pytest.mark.parametrize(
        "key", ["sequence_length", "context_length", "adaptations"]
    )
    def test_rejects_llm_model_keys(self, sd_checkpoint, key):
        with pytest.raises(ValueError, match=f"'model.{key}' does not apply"):
            parse_diffusion_document(_doc(sd_checkpoint, **{key: 1}))

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"dtype": "int8"}, "Unsupported model dtype"),
            ({"height": 0}, "must be positive"),
            ({"num_inference_steps": 2.5}, "must be an integer"),
            ({"seed": True}, "must be an integer"),
            ({"guidance_scale": "high"}, "must be a number"),
            ({"scheduler": 3}, "must be a string"),
            ({"steps": 10}, "Unrecognized keys"),
        ],
    )
    def test_rejects_bad_model_values(self, sd_checkpoint, overrides, match):
        with pytest.raises(ValueError, match=match):
            parse_diffusion_document(_doc(sd_checkpoint, **overrides))

    def test_rejects_text_metric(self, sd_checkpoint):
        doc = {**_doc(sd_checkpoint), "metrics": [{"name": "PPL"}]}
        with pytest.raises(
            ValueError, match="'PPL' does not evaluate image generation"
        ):
            parse_diffusion_document(doc)

    def test_requires_metrics(self, sd_checkpoint):
        with pytest.raises(RuntimeError, match="Metrics not specified"):
            parse_diffusion_document({"model": {"model_id": sd_checkpoint}})

    def test_unknown_section(self, sd_checkpoint):
        with pytest.raises(ValueError, match="Unrecognized sections"):
            parse_diffusion_document({**_doc(sd_checkpoint), "foo": 1})

    def test_unregistered_pipeline(self, tmp_path):
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "FluxPipeline"})
        )
        with pytest.raises(LookupError, match="Failed to configure model"):
            parse_diffusion_document(_doc(str(tmp_path)))

    def test_profiler_and_run_group(self, sd_checkpoint):
        config = parse_diffusion_document(
            {
                **_doc(sd_checkpoint),
                "profiler": {"capture_intermediate_data": True, "nvml_interval_ms": 5},
                "run_group": "abc",
            }
        )
        assert config.profiler.capture_intermediate_data
        assert config.profiler.gpu_meter_kwargs == {"nvml_interval_ms": 5}
        assert config.run_group == "abc"


class TestEvaluationContextKey:
    def test_generation_settings_change_the_fp_key(self, sd_checkpoint, tmp_path):
        cache = DiskBackedFPCache(tmp_path / "fp")

        def model_hash(**overrides):
            model = parse_diffusion_document(_doc(sd_checkpoint, **overrides)).model
            return DiffusionEvaluationContext(cache, model, "torch")._model_hash

        assert model_hash() == model_hash()
        assert model_hash() != model_hash(num_inference_steps=20)
        assert model_hash() != model_hash(seed=1)
