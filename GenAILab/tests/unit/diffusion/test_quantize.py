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

from GenAILab.qai_hub_lm.diffusion.base import COMPONENTS

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

    def test_every_component(self, sd_checkpoint):
        precision = {name: W8A16["unet"] for name in COMPONENTS}
        config = parse_diffusion_document(
            _doc(sd_checkpoint, precision=precision, recipe={"name": "Calibration"})
        )
        assert set(config.precision) == set(COMPONENTS)
        assert set(config.precision_dict()) == set(COMPONENTS)

    def test_fp_document_is_not_quantized(self, sd_checkpoint):
        config = parse_diffusion_document(_doc(sd_checkpoint))
        assert not config.is_quantized and config.calibration is None
        assert config.precision_dict() == {}

    @pytest.mark.parametrize(
        "sections, match",
        [
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

    @pytest.mark.parametrize("quantized", [("unet",), COMPONENTS], ids=["unet", "all"])
    def test_end_to_end_w8a16(self, quantized, fake_parti, tmp_path, monkeypatch):
        """Quantize components, calibrate each with the others in float, and score."""
        pytest.importorskip("aimet_onnx")
        from GenAILab.bench.diffusion import quantize as diffusion_quantize
        from GenAILab.bench.diffusion.onnx_cache import DiffusionONNXCache
        from GenAILab.qai_hub_lm.diffusion import onnx as diffusion_onnx

        scored = []
        monkeypatch.setattr(
            diffusion_metrics,
            "clip_scores",
            lambda images, texts, *a, **k: (
                scored.append((images, texts, _kinds(generators[0])))
                or [1.0] * len(texts)
            ),
        )
        calibrated = {}
        generators = []
        original_calibrate = diffusion_quantize.calibrate

        def calibrate(generator, name, prompts):
            generators[:] = [generator]
            calibrated[name] = ([p["prompt"] for p in prompts], _kinds(generator))
            original_calibrate(generator, name, prompts)

        def _kinds(generator):
            return {
                n: type(generator.components.component(n)).__name__ for n in COMPONENTS
            }

        monkeypatch.setattr(diffusion_quantize, "calibrate", calibrate)

        doc = {
            "model": {"model_id": TINY_SD, "num_inference_steps": 2},
            "precision": {name: W8A16["unet"] for name in quantized},
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

        # Each component is calibrated in pipeline order, as the only sim.
        assert list(calibrated) == [n for n in COMPONENTS if n in quantized]
        for name, (prompts, kinds) in calibrated.items():
            assert len(prompts) == 2
            assert kinds == {
                n: "SimComponent" if n == name else diffusion_onnx.ORTComponent.__name__
                for n in COMPONENTS
            }
        # Every component calibrates on the same held-out prompts.
        assert len({tuple(p) for p, _ in calibrated.values()}) == 1

        ((images, texts, kinds),) = scored
        assert not set(calibrated[quantized[0]][0]) & set(texts)
        assert all(np.asarray(i).dtype == np.uint8 for i in images)
        # Evaluation runs with every quantized component's sim at once.
        assert kinds == {
            n: "SimComponent" if n in quantized else "ORTComponent" for n in COMPONENTS
        }

        (entry,) = json.loads((tmp_path / "out" / "profiling_data.json").read_text())[
            "StableDiffusionPipeline"
        ]
        assert set(entry["precision"]) == set(quantized)
        assert entry["precision"]["unet"] == {
            "weight": {"qtype": "int8", "granularity": "PCQ"},
            "activations": "int16",
        }
        for name, stats in entry["components"].items():
            if name not in quantized:
                assert stats["steps"] == []
                continue
            (step,) = stats["steps"]
            assert step["recipe_name"] == "Calibration"
            assert step["recipe_kwargs"] == {"num_prompts": 2, "sample_seed": 0}
            assert step["dataset_name"] == "PartiPrompts"


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

        diffusion_quantize.calibrate(generator, "unet", [{"prompt": "a cat"}])

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


@pytest.mark.parametrize("name", ["text_encoder", "vae_decoder"])
def test_calibration_runs_only_as_far_as_the_component(name, tiny_pipeline, tmp_path):
    """The text encoder calibrates on prompts alone; the VAE on denoised latents."""
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
    calls = {n: 0 for n in COMPONENTS}
    for n in COMPONENTS:
        if n == name:
            continue
        component = components.component(n)
        original = component.forward

        def counted(*inputs, _n=n, _original=original):
            calls[_n] += 1
            return _original(*inputs)

        component.forward = counted
    precision = ComponentPrecision(
        weight=WeightPrecision(qtype=resolve_qtype("int8")),
        activations=resolve_qtype("int16"),
    )
    sim = diffusion_quantize.build_quantsim(paths[name], precision, "cpu")
    setattr(components, name, diffusion_quantize.SimComponent(sim, "cpu"))
    generator = DiffusionGenerator(
        components,
        GenerationConfig(
            height=64, width=64, num_inference_steps=2, guidance_scale=7.5
        ),
    )

    diffusion_quantize.calibrate(generator, name, [{"prompt": "a cat"}])

    enabled = [q for q in sim.qc_quantize_op_dict.values() if q.enabled]
    assert enabled and all(q.is_initialized() for q in enabled)
    if name == "text_encoder":
        assert calls == {"text_encoder": 0, "unet": 0, "vae_decoder": 0}
    else:
        # 2 steps x (cond + uncond) UNet calls, one text-encoder call per prompt.
        assert calls == {"text_encoder": 2, "unet": 4, "vae_decoder": 0}


class TestMaskQuantizers:
    def test_causal_mask_and_masked_scores_are_not_quantized(self, tmp_path):
        pytest.importorskip("aimet_onnx")
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        from GenAILab.bench.diffusion import quantize as diffusion_quantize
        from GenAILab.bench.diffusion.config import ComponentPrecision
        from GenAILab.bench.precision import WeightPrecision, resolve_qtype

        # scores = x @ W; softmax(scores + causal_mask), as CLIP's attention.
        mask = np.triu(np.full((4, 4), np.finfo(np.float32).min, np.float32), 1)
        weight = np.random.default_rng(0).standard_normal((4, 4)).astype(np.float32)
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["x", "w"], ["scores"]),
                helper.make_node("Add", ["scores", "mask"], ["masked"]),
                helper.make_node("Softmax", ["masked"], ["y"], axis=-1),
            ],
            "attn",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4])],
            [
                numpy_helper.from_array(weight, "w"),
                numpy_helper.from_array(mask, "mask"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        path = tmp_path / "model.onnx"
        onnx.save(model, path)
        precision = ComponentPrecision(
            weight=WeightPrecision(qtype=resolve_qtype("int8")),
            activations=resolve_qtype("int16"),
        )

        sim = diffusion_quantize.build_quantsim(path, precision, "cpu")

        quantizers = sim.qc_quantize_op_dict
        assert not quantizers["mask"].enabled
        assert not quantizers["masked"].enabled
        assert quantizers["scores"].enabled  # Unmasked scores stay quantized.
        assert quantizers["y"].enabled
        x = np.random.default_rng(1).standard_normal((1, 4, 4)).astype(np.float32)
        sim.compute_encodings([{"x": x}])
        (out,) = sim.session.run(None, {"x": x})
        # Masked positions get no attention, as in float.
        assert np.all(np.triu(out[0], 1) == 0)
        np.testing.assert_allclose(out.sum(-1), 1, atol=1e-3)
