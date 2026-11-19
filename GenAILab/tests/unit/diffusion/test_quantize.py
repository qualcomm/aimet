# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion component quantization: config, held-out prompts, and the aimet_onnx path."""

import json

import numpy as np
import pytest
import torch

from GenAILab.bench.diffusion import metrics as diffusion_metrics
from GenAILab.bench.diffusion.config import (
    DEFAULT_CALIBRATION_PROMPTS,
    parse_diffusion_document,
)
from GenAILab.bench.diffusion.experiment import run_experiment
from GenAILab.bench.diffusion.metrics import CLIPScore, ImageDump
from GenAILab.bench.fp_cache import DiskBackedFPCache

from .conftest import FAKE_PARTI, TINY_SD

W8A16 = {
    "unet": {"weight": {"qtype": "int8", "granularity": "PCQ"}, "activations": "int16"}
}


@pytest.fixture
def sd_checkpoint(tmp_path):
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionPipeline"})
    )
    return str(tmp_path)


def _doc(model_id, **sections):
    return {
        "model": {"model_id": model_id},
        "metrics": [{"name": "CLIPScore", "num_prompts": 4}],
        **sections,
    }


class TestPrecisionConfig:
    def test_unet_w8a16(self, sd_checkpoint):
        config = parse_diffusion_document(
            _doc(sd_checkpoint, precision=W8A16, recipe={"name": "Calibration"})
        )
        assert config.is_quantized
        (name,) = config.precision
        assert name == "unet"
        assert config.precision["unet"].weight.qtype.bits == 8
        assert config.precision["unet"].activations.bits == 16
        assert config.calibration.num_prompts == DEFAULT_CALIBRATION_PROMPTS
        assert config.precision_dict() == {
            "unet": {
                "weight": {"qtype": "int8", "granularity": "PCQ"},
                "activations": "int16",
            }
        }

    def test_shorthand_and_recipe_kwargs(self, sd_checkpoint):
        config = parse_diffusion_document(
            _doc(
                sd_checkpoint,
                precision={"unet": {"weight": 8, "activations": 16}},
                recipe=[{"name": "Calibration", "num_prompts": 3, "sample_seed": 5}],
            )
        )
        assert config.precision["unet"].weight.qtype.bits == 8
        assert config.calibration.to_dict() == {"num_prompts": 3, "sample_seed": 5}

    def test_fp_document_is_not_quantized(self, sd_checkpoint):
        config = parse_diffusion_document(_doc(sd_checkpoint))
        assert not config.is_quantized and config.calibration is None
        assert config.precision_dict() == {}

    @pytest.mark.parametrize(
        "sections, match",
        [
            (
                {"precision": {"vae_decoder": W8A16["unet"]}},
                "vae_decoder is not supported",
            ),
            (
                {"precision": {"vae": W8A16["unet"]}},
                "Unknown diffusion component 'vae'",
            ),
            ({"precision": {"activations": "int16"}}, "Unknown diffusion component"),
            ({"precision": {"unet": {"weight": "int8"}}}, "needs both"),
            (
                {"precision": {"unet": {**W8A16["unet"], "kv_cache": 8}}},
                "Unrecognized keys in 'precision.unet'",
            ),
            (
                {"precision": {"unet": {"weight": "float16", "activations": "int16"}}},
                "weight' qtype must be one of",
            ),
            (
                {"precision": {"unet": {"weight": "int8", "activations": "float16"}}},
                "activations' must be one of",
            ),
            (
                {
                    "precision": {
                        "unet": {
                            "weight": {
                                "qtype": "int4",
                                "granularity": "LPBQ",
                                "block_size": 64,
                            },
                            "activations": "int16",
                        }
                    }
                },
                "granularity PCQ only",
            ),
        ],
    )
    def test_rejects_bad_precision(self, sd_checkpoint, sections, match):
        with pytest.raises(ValueError, match=match):
            parse_diffusion_document(
                _doc(sd_checkpoint, recipe={"name": "Calibration"}, **sections)
            )

    @pytest.mark.parametrize(
        "sections, match",
        [
            ({"precision": W8A16}, "needs a 'recipe'"),
            ({"recipe": {"name": "Calibration"}}, "needs a 'precision' section"),
        ],
    )
    def test_precision_and_recipe_go_together(self, sd_checkpoint, sections, match):
        with pytest.raises(ValueError, match=match):
            parse_diffusion_document(_doc(sd_checkpoint, **sections))

    @pytest.mark.parametrize(
        "recipe, match",
        [
            ({"name": "SeqMSE"}, "'Calibration' only"),
            (
                [{"name": "Calibration"}, {"name": "Calibration"}],
                "single 'Calibration'",
            ),
            ({"name": "Calibration", "num_iterations": 3}, "Unrecognized keys"),
            ({"name": "Calibration", "num_prompts": 0}, "must be positive"),
            ({"name": "Calibration", "num_prompts": "8"}, "must be an integer"),
        ],
    )
    def test_rejects_bad_recipe(self, sd_checkpoint, recipe, match):
        with pytest.raises(ValueError, match=match):
            parse_diffusion_document(
                _doc(sd_checkpoint, precision=W8A16, recipe=recipe)
            )


class TestHeldOutPrompts:
    def test_excludes_evaluated_prompts(self, fake_parti):
        evaluated = CLIPScore.evaluation_prompts(num_prompts=5)
        held_out = diffusion_metrics.held_out_parti_prompts(
            5, 0, exclude={p["prompt"] for p in evaluated}
        )
        assert len(held_out) == 5
        assert not {p["prompt"] for p in held_out} & {p["prompt"] for p in evaluated}

    def test_evaluation_prompts_use_each_metrics_defaults(self, fake_parti):
        # CLIPScore defaults to 64 prompts (all 10 fake ones), ImageDump to 8.
        assert len(CLIPScore.evaluation_prompts()) == len(FAKE_PARTI)
        assert len(ImageDump.evaluation_prompts()) == 8
        assert ImageDump.evaluation_prompts(num_prompts=2, sample_seed=1) == (
            diffusion_metrics.load_parti_prompts(2, 1)
        )

    def test_too_few_left(self, fake_parti):
        with pytest.raises(ValueError, match="only 1 PartiPrompts"):
            diffusion_metrics.held_out_parti_prompts(
                2, 0, exclude={p["prompt"] for p in FAKE_PARTI[1:]}
            )


class TestQuantizedExperiment:
    def test_rejects_float16_before_loading(self, tmp_path, monkeypatch):
        pytest.importorskip("aimet_onnx")
        from GenAILab.bench.diffusion.onnx_cache import DiffusionONNXCache
        from GenAILab.qai_hub_lm.diffusion import StableDiffusion

        loaded = []
        monkeypatch.setattr(
            StableDiffusion,
            "instantiate_float_model",
            classmethod(lambda cls, *a, **k: loaded.append(1)),
        )
        doc = {
            "model": {"model_id": TINY_SD, "dtype": "float16"},
            "precision": W8A16,
            "recipe": {"name": "Calibration"},
            "metrics": [{"name": "CLIPScore", "num_prompts": 1}],
        }
        with pytest.raises(ValueError, match="not float16"):
            run_experiment(
                doc,
                fp_cache=DiskBackedFPCache(tmp_path / "fp"),
                results_dir=tmp_path / "out",
                framework="onnx",
                onnx_cache=DiffusionONNXCache(tmp_path / "onnx"),
            )
        assert not loaded

    def test_torch_framework_rejected(self, tmp_path):
        doc = {
            "model": {"model_id": TINY_SD},
            "precision": W8A16,
            "recipe": {"name": "Calibration"},
            "metrics": [{"name": "CLIPScore", "num_prompts": 1}],
        }
        with pytest.raises(ValueError, match="aimet_onnx only"):
            run_experiment(
                doc,
                fp_cache=DiskBackedFPCache(tmp_path / "fp"),
                results_dir=tmp_path / "out",
                framework="torch",
            )

    def test_end_to_end_unet_w8a16(self, fake_parti, tmp_path, monkeypatch):
        """Quantize only the UNet, calibrate on held-out prompts, and score."""
        pytest.importorskip("aimet_onnx")
        from GenAILab.bench.diffusion import quantize as diffusion_quantize
        from GenAILab.bench.diffusion.onnx_cache import DiffusionONNXCache
        from GenAILab.qai_hub_lm.diffusion import onnx as diffusion_onnx

        scored = []
        monkeypatch.setattr(
            diffusion_metrics,
            "clip_scores",
            lambda images, texts, *a, **k: (
                scored.append((images, texts)) or [1.0] * len(texts)
            ),
        )
        calibrated = []
        original_calibrate = diffusion_quantize.calibrate

        def calibrate(generator, quantized, prompts):
            calibrated.append([p["prompt"] for p in prompts])
            # Only the UNet is a sim; the text encoder and VAE stay float sessions.
            components = generator.components
            assert isinstance(components.unet, diffusion_quantize.SimComponent)
            assert isinstance(components.text_encoder, diffusion_onnx.ORTComponent)
            assert isinstance(components.vae_decoder, diffusion_onnx.ORTComponent)
            original_calibrate(generator, quantized, prompts)

        monkeypatch.setattr(diffusion_quantize, "calibrate", calibrate)

        doc = {
            "model": {"model_id": TINY_SD, "num_inference_steps": 2},
            "precision": W8A16,
            "recipe": {"name": "Calibration", "num_prompts": 2},
            "metrics": [{"name": "CLIPScore", "num_prompts": 3}],
            "run_group": "rg",
        }
        run_experiment(
            doc,
            fp_cache=DiskBackedFPCache(tmp_path / "fp"),
            results_dir=tmp_path / "out",
            framework="onnx",
            onnx_cache=DiffusionONNXCache(tmp_path / "onnx"),
        )

        (calibration_prompts,) = calibrated
        assert len(calibration_prompts) == 2
        ((images, texts),) = scored
        assert not set(calibration_prompts) & set(texts)
        assert all(np.asarray(i).dtype == np.uint8 for i in images)

        (entry,) = json.loads((tmp_path / "out" / "profiling_data.json").read_text())[
            "StableDiffusionPipeline"
        ]
        assert entry["precision"] == {
            "unet": {
                "weight": {"qtype": "int8", "granularity": "PCQ"},
                "activations": "int16",
            }
        }
        components = entry["components"]
        assert components["unet"]["recipe_name"] == "Calibration"
        (step,) = components["unet"]["steps"]
        assert step["recipe_kwargs"] == {"num_prompts": 2, "sample_seed": 0}
        assert step["dataset_name"] == "PartiPrompts"
        assert components["text_encoder"]["steps"] == []
        assert components["vae_decoder"]["steps"] == []


class TestCalibrate:
    def test_encodings_are_computed_and_weights_folded(self, tiny_pipeline, tmp_path):
        pytest.importorskip("aimet_onnx")
        from GenAILab.bench.diffusion import quantize as diffusion_quantize
        from GenAILab.bench.diffusion.config import ComponentPrecision
        from GenAILab.bench.precision import WeightPrecision, resolve_qtype
        from GenAILab.qai_hub_lm.diffusion import (
            DiffusionGenerator,
            GenerationConfig,
            StableDiffusion,
        )
        from GenAILab.qai_hub_lm.diffusion import onnx as diffusion_onnx

        components = StableDiffusion.build_components(tiny_pipeline)
        paths = diffusion_onnx.export_components(
            StableDiffusion, components, 64, 64, tmp_path
        )
        components = diffusion_onnx.load_components(components, paths)
        float_unet = components.unet
        precision = ComponentPrecision(
            weight=WeightPrecision(qtype=resolve_qtype("int8")),
            activations=resolve_qtype("int16"),
        )
        sim = diffusion_quantize.build_quantsim(paths["unet"], precision, "cpu")
        components.unet = diffusion_quantize.SimComponent(sim, "cpu")
        generator = DiffusionGenerator(
            components,
            GenerationConfig(
                height=64, width=64, num_inference_steps=2, guidance_scale=7.5
            ),
        )

        diffusion_quantize.calibrate(generator, ["unet"], [{"prompt": "a cat"}])

        enabled = [q for q in sim.qc_quantize_op_dict.values() if q.enabled]
        assert enabled and all(q.is_initialized() for q in enabled)
        assert sim._folded_param_quantizers  # pylint: disable=protected-access
        torch_components = StableDiffusion.build_components(tiny_pipeline)
        latents, timestep, hidden = StableDiffusion.get_sample_inputs(
            "unet", torch_components, 64, 64
        )
        gen = torch.Generator().manual_seed(0)
        sample = (
            torch.randn(latents.shape, generator=gen),
            timestep + 500,
            torch.randn(hidden.shape, generator=gen),
        )
        quantized = components.unet(*sample)
        reference = float_unet(*sample)
        assert quantized.shape == reference.shape
        assert not torch.equal(quantized, reference)
        assert (
            torch.nn.functional.cosine_similarity(
                quantized.flatten(), reference.flatten(), dim=0
            )
            > 0.9
        )

        components.unet.to("cpu")  # Same device: the session is kept.
        assert sim.session is not None
        components.unet.release()
        assert sim.session is None
        torch.testing.assert_close(components.unet(*sample), quantized)
