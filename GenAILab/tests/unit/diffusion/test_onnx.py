# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""ONNX export of the diffusion components and the ONNX Runtime experiment path."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("onnxruntime")

from GenAILab.bench.diffusion import metrics as diffusion_metrics
from GenAILab.bench.diffusion.experiment import run_experiment
from GenAILab.bench.diffusion.onnx_cache import DiffusionONNXCache, export_key
from GenAILab.bench.fp_cache import DiskBackedFPCache
from GenAILab.qai_hub_lm.diffusion import (
    DiffusionGenerator,
    GenerationConfig,
    StableDiffusion,
)
from GenAILab.qai_hub_lm.diffusion import onnx as diffusion_onnx
from GenAILab.qai_hub_lm.diffusion.base import COMPONENTS

from .conftest import TINY_SD

PROMPT = "a photo of an astronaut riding a horse"


@pytest.fixture(scope="module")
def exported(tiny_pipeline, tmp_path_factory):
    """The tiny pipeline's components, exported once for the module."""
    components = StableDiffusion.build_components(tiny_pipeline)
    paths = diffusion_onnx.export_components(
        StableDiffusion, components, 64, 64, tmp_path_factory.mktemp("onnx")
    )
    return components, paths


class TestExport:
    def test_writes_every_component_at_the_pinned_opset(self, exported):
        import onnx

        _, paths = exported
        assert set(paths) == set(COMPONENTS)
        for path in paths.values():
            model = onnx.load(str(path), load_external_data=False)
            opsets = {o.domain or "ai.onnx": o.version for o in model.opset_import}
            assert opsets["ai.onnx"] == diffusion_onnx.ONNX_OPSET

    def test_keeps_wrapper_io_names(self, exported):
        import onnxruntime

        _, paths = exported
        for name, path in paths.items():
            session = onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            assert tuple(i.name for i in session.get_inputs()) == (
                StableDiffusion.get_input_names(name)
            )
            assert tuple(o.name for o in session.get_outputs()) == (
                StableDiffusion.get_output_names(name)
            )

    @pytest.mark.parametrize("name", COMPONENTS)
    def test_component_matches_torch(self, exported, name):
        components, paths = exported
        inputs = StableDiffusion.get_sample_inputs(name, components, 64, 64)
        generator = torch.Generator().manual_seed(0)
        inputs = tuple(
            torch.randn(x.shape, generator=generator)
            if x.is_floating_point() and x.dim() > 1
            else torch.full_like(x, 500.0)
            if x.is_floating_point()
            else torch.randint(0, 99, x.shape, dtype=x.dtype)
            for x in inputs
        )
        with torch.no_grad():
            ref = components.component(name)(*inputs)
        got = diffusion_onnx.ORTComponent(paths[name])(*inputs)
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_rejects_bfloat16(self):
        with pytest.raises(ValueError, match="bfloat16 Conv needs ONNX opset 22"):
            diffusion_onnx.check_dtype(torch.bfloat16)


class TestORTGeneration:
    def test_matches_torch_generator(self, exported):
        components, paths = exported
        config = GenerationConfig(64, 64, num_inference_steps=4, guidance_scale=7.5)
        onnx_components = diffusion_onnx.load_components(components, paths)
        ours = DiffusionGenerator(onnx_components, config).generate(PROMPT, seed=3)
        ref = DiffusionGenerator(components, config).generate(PROMPT, seed=3)
        assert np.abs(ours.astype(np.int32) - ref.astype(np.int32)).max() <= 1

    def test_components_report_device_and_dtype(self, exported):
        components, paths = exported
        onnx_components = diffusion_onnx.load_components(components, paths)
        assert onnx_components.device == torch.device("cpu")
        assert onnx_components.dtype == torch.float32
        # Host-side helpers are carried over from the torch components.
        assert onnx_components.tokenizer is components.tokenizer
        assert onnx_components.scheduler is components.scheduler

    def test_to_moves_device_and_ignores_dtype(self, exported):
        _, paths = exported
        component = diffusion_onnx.ORTComponent(paths["unet"])
        session = component.session
        assert component.to(torch.float16) is component
        assert component.device == torch.device("cpu")
        assert component.session is session  # A dtype-only .to keeps the session.
        component.to("meta")
        assert component.device == torch.device("meta")
        assert component._session is None  # Released; rebuilt for the new providers.

    def test_release_drops_the_session(self, exported):
        _, paths = exported
        component = diffusion_onnx.ORTComponent(paths["unet"])
        component.release()  # No session yet: a no-op.
        session = component.session
        component.release()
        assert component._session is None
        assert component.session is not session  # Rebuilt on next use.

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_leaving_the_gpu_returns_arena_memory(self, exported):
        components, paths = exported
        onnx_components = diffusion_onnx.load_components(components, paths)
        config = GenerationConfig(64, 64, num_inference_steps=2, guidance_scale=7.5)
        generator = DiffusionGenerator(onnx_components, config)

        def generate_on_gpu():
            with generator.on_device("cuda"):
                generator.generate(PROMPT, seed=0)

        # The first CUDA session also pays ORT's one-time CUDA setup, which is
        # never freed; only memory held across runs would be a leak.
        generate_on_gpu()
        free_before, _ = torch.cuda.mem_get_info()
        for _ in range(2):
            generate_on_gpu()
        free_after, _ = torch.cuda.mem_get_info()
        assert all(m._session is None for _, m in onnx_components.items())
        # The shared arena hands back what each run grew.
        assert free_before - free_after < 16 * 2**20

    def test_wrong_input_count(self, exported):
        _, paths = exported
        component = diffusion_onnx.ORTComponent(paths["unet"])
        with pytest.raises(TypeError, match="unet takes 3 inputs"):
            component(torch.zeros(1, 4, 32, 32))

    @pytest.mark.parametrize(
        "device, expected",
        [
            ("cpu", ["CPUExecutionProvider"]),
            ("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ],
    )
    def test_providers(self, device, expected):
        providers = diffusion_onnx.providers_for(device)
        assert [p if isinstance(p, str) else p[0] for p in providers] == expected
        if device == "cuda":
            assert providers[0][1]["cudnn_conv_algo_search"] == "HEURISTIC"


class TestONNXCache:
    @staticmethod
    def _key(**overrides):
        return {**export_key("org/sd", "abc123", "float32", 64, 64), **overrides}

    def test_key_covers_what_changes_the_graphs(self):
        key = export_key("org/sd", "abc123", "float16", 64, 96)
        assert key["opset"] == diffusion_onnx.ONNX_OPSET
        assert {"model_id", "revision", "dtype", "height", "width", "torch"} <= set(key)
        base = DiffusionONNXCache.key_hash(self._key())
        for change in ({"dtype": "float16"}, {"height": 128}, {"revision": "def456"}):
            assert DiffusionONNXCache.key_hash(self._key(**change)) != base

    def test_exports_once(self, tmp_path):
        cache = DiffusionONNXCache(tmp_path)
        calls = []

        def export(directory):
            calls.append(directory)
            for name in COMPONENTS:
                (directory / name).mkdir()
                (directory / name / diffusion_onnx.MODEL_FILE).write_bytes(b"x")

        first = cache.get_or_export(self._key(), export)
        second = cache.get_or_export(self._key(), export)
        assert first == second and len(calls) == 1
        entry = first["unet"].parents[1]
        assert json.loads((entry / "key.json").read_text())["model_id"] == "org/sd"

    def test_failed_export_leaves_no_entry(self, tmp_path):
        cache = DiffusionONNXCache(tmp_path)

        def export(directory):
            (directory / "text_encoder").mkdir()
            raise RuntimeError("export failed")

        with pytest.raises(RuntimeError, match="export failed"):
            cache.get_or_export(self._key(), export)
        assert cache.paths(self._key()) is None
        assert list(tmp_path.iterdir()) == []

    def test_incomplete_export_is_an_error(self, tmp_path):
        cache = DiffusionONNXCache(tmp_path)
        with pytest.raises(RuntimeError, match="did not write every component"):
            cache.get_or_export(self._key(), lambda directory: None)


class TestONNXExperiment:
    def test_end_to_end_with_cache(self, fake_parti, tmp_path, monkeypatch):
        monkeypatch.setattr(
            diffusion_metrics,
            "clip_scores",
            lambda images, texts, *a, **k: [1.0] * len(texts),
        )
        exports = []
        original = diffusion_onnx.export_components
        monkeypatch.setattr(
            diffusion_onnx,
            "export_components",
            lambda *a, **k: exports.append(1) or original(*a, **k),
        )
        doc = {
            "model": {"model_id": TINY_SD, "num_inference_steps": 2},
            "metrics": [{"name": "CLIPScore", "num_prompts": 2}],
            "run_group": "rg",
        }
        onnx_cache = DiffusionONNXCache(tmp_path / "onnx")
        for _ in range(2):
            run_experiment(
                json.loads(json.dumps(doc)),
                fp_cache=DiskBackedFPCache(tmp_path / "fp"),
                results_dir=tmp_path / "out",
                framework="onnx",
                onnx_cache=onnx_cache,
            )
        assert len(exports) == 1  # The second run reuses the cached graphs.

        data = json.loads((tmp_path / "out" / "profiling_data.json").read_text())
        entries = data["StableDiffusionPipeline"]
        assert len(entries) == 2
        modifiers = entries[0]["model_modifiers"]
        assert modifiers["framework"] == "onnx"
        assert modifiers["onnx_opset"] == diffusion_onnx.ONNX_OPSET
        assert modifiers["scheduler"] == StableDiffusion.DEFAULT_SCHEDULER

    def test_bfloat16_rejected_before_loading(self, tmp_path, monkeypatch):
        loaded = []
        monkeypatch.setattr(
            StableDiffusion,
            "instantiate_float_model",
            classmethod(lambda cls, *a, **k: loaded.append(1)),
        )
        doc = {
            "model": {"model_id": TINY_SD, "dtype": "bfloat16"},
            "metrics": [{"name": "CLIPScore", "num_prompts": 1}],
        }
        with pytest.raises(ValueError, match="bfloat16"):
            run_experiment(
                doc,
                fp_cache=DiskBackedFPCache(tmp_path / "fp"),
                results_dir=tmp_path / "out",
                framework="onnx",
                onnx_cache=DiffusionONNXCache(tmp_path / "onnx"),
            )
        assert not loaded

    def test_framework_is_part_of_the_fp_cache_key(self):
        from GenAILab.bench.diffusion.config import parse_diffusion_document
        from GenAILab.bench.diffusion.experiment import DiffusionEvaluationContext

        model = parse_diffusion_document(
            {"model": {"model_id": TINY_SD}, "metrics": [{"name": "CLIPScore"}]}
        ).model
        fp_cache = DiskBackedFPCache.__new__(DiskBackedFPCache)
        hashes = {
            framework: DiffusionEvaluationContext(
                fp_cache, model, framework
            )._model_hash
            for framework in ("torch", "onnx")
        }
        assert hashes["torch"] != hashes["onnx"]

    def test_unknown_framework(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown framework 'tflite'"):
            run_experiment({}, fp_cache=None, results_dir=tmp_path, framework="tflite")

    def test_onnx_needs_a_cache(self, tmp_path):
        with pytest.raises(ValueError, match="needs an onnx_cache"):
            run_experiment({}, fp_cache=None, results_dir=tmp_path, framework="onnx")


class TestONNXEntryPoint:
    def test_runs_and_writes_onnx_results(self, fake_parti, tmp_path, monkeypatch):
        import yaml

        from GenAILab.bench.onnx import run_diffusion

        monkeypatch.setattr(
            diffusion_metrics,
            "clip_scores",
            lambda images, texts, *a, **k: [1.0] * len(texts),
        )
        config = tmp_path / "cfg.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model": {"model_id": TINY_SD, "num_inference_steps": 1},
                    "metrics": [{"name": "CLIPScore", "num_prompts": 1}],
                }
            )
        )
        args = run_diffusion.build_parser().parse_args(
            [
                "--config", str(config),
                "--results-dir", str(tmp_path / "results"),
                "--fp-cache-dir", str(tmp_path / "fp"),
                "--onnx-cache-dir", str(tmp_path / "onnx"),
            ]
        )  # fmt: skip
        assert run_diffusion.run(args) == 0
        data = json.loads((tmp_path / "results" / "profiling_data.json").read_text())
        (entry,) = data["StableDiffusionPipeline"]
        assert entry["model_modifiers"]["framework"] == "onnx"
        assert any((tmp_path / "onnx").iterdir())
