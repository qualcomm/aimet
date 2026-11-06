# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Diffusion metrics and the end-to-end diffusion experiment runner."""

import json
from collections import Counter

import numpy as np
import pytest
import torch

from GenAILab.bench.diffusion import metrics as diffusion_metrics
from GenAILab.bench.diffusion.metrics import (
    CLIPScore,
    ImageDump,
    clip_scores,
    make_grid,
    stratified_sample,
)
from GenAILab.bench.diffusion.experiment import run_experiment
from GenAILab.bench.fp_cache import DiskBackedFPCache

from .conftest import TINY_SD

#: Real CLIP architecture and tokenizer, small enough for CPU. (The
#: tiny-random-clip checkpoint's vocab is smaller than its tokenizer's.)
SMALL_CLIP = "openai/clip-vit-base-patch32"

_FAKE_PARTI = [
    {"prompt": f"{category} prompt {i}", "category": category}
    for category, count in [("Animals", 6), ("Food", 3), ("People", 1)]
    for i in range(count)
]


@pytest.fixture
def fake_parti(monkeypatch):
    """Serve PartiPrompts from a 10-row stand-in instead of the Hub."""
    monkeypatch.setattr(
        diffusion_metrics,
        "load_parti_prompts",
        lambda num_prompts=64, sample_seed=0: stratified_sample(
            _FAKE_PARTI, num_prompts, sample_seed
        ),
    )


class TestStratifiedSample:
    def test_proportional_by_largest_remainder(self):
        # Quotas 3.0 / 1.5 / 0.5: the one leftover goes to the tied remainders'
        # first category by name.
        sample = stratified_sample(_FAKE_PARTI, 5, sample_seed=0)
        assert Counter(r["category"] for r in sample) == {"Animals": 3, "Food": 2}

    def test_deterministic_per_seed(self):
        assert stratified_sample(_FAKE_PARTI, 5, 0) == stratified_sample(
            _FAKE_PARTI, 5, 0
        )
        assert stratified_sample(_FAKE_PARTI, 5, 0) != stratified_sample(
            _FAKE_PARTI, 5, 1
        )

    def test_more_than_available_returns_everything(self):
        assert len(stratified_sample(_FAKE_PARTI, 100, 0)) == len(_FAKE_PARTI)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            stratified_sample(_FAKE_PARTI, 0, 0)


class TestCLIPScore:
    def test_matches_manual_computation(self):
        from transformers import CLIPModel, CLIPProcessor

        rng = np.random.default_rng(0)
        images = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        texts = ["a dog", "a red car on a street", "noise"]
        scores = clip_scores(images, texts, SMALL_CLIP, batch_size=2)

        model = CLIPModel.from_pretrained(SMALL_CLIP).eval()
        processor = CLIPProcessor.from_pretrained(SMALL_CLIP)
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(**inputs)
        image = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        expected = (100 * (image * text).sum(-1)).clamp(min=0)
        np.testing.assert_allclose(scores, expected.numpy(), rtol=1e-4, atol=1e-3)

    def test_matching_caption_scores_higher(self):
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (224, 224), "white")
        ImageDraw.Draw(image).ellipse((40, 40, 184, 184), fill="red")
        image = np.asarray(image)
        red, blue = clip_scores(
            [image, image], ["a red circle", "a blue square"], SMALL_CLIP
        )
        assert red > blue


class TestDiffusionRunner:
    def test_end_to_end(self, fake_parti, tmp_path):
        doc = {
            "model": {"model_id": TINY_SD, "num_inference_steps": 2},
            "metrics": [
                {"name": "CLIPScore", "num_prompts": 4, "clip_model": SMALL_CLIP},
                {"name": "ImageDump", "num_prompts": 4},
            ],
            "run_group": "rg",
        }
        results = run_experiment(
            doc,
            fp_cache=DiskBackedFPCache(tmp_path / "fp"),
            results_dir=tmp_path / "out",
        )
        assert [r.metric_name for r in results] == ["CLIPScore", "ImageDump"]
        clip = results[0]
        assert 0 <= clip.result <= 100
        assert clip.details["num_prompts"] == 4
        assert set(clip.details["by_category"]) == {"Animals", "Food", "People"}
        assert all(r.profiler is not None for r in results)

        image_dir = tmp_path / "out" / "images" / "tiny-sd-pipe_rg"
        assert sorted(p.name for p in image_dir.iterdir()) == [
            "000.png",
            "001.png",
            "002.png",
            "003.png",
            "grid.png",
        ]

        data = json.loads((tmp_path / "out" / "profiling_data.json").read_text())
        (entry,) = data["StableDiffusionPipeline"]
        assert entry["model_id"] == TINY_SD
        assert entry["run_group"] == "rg"
        assert set(entry["components"]) == {"text_encoder", "unet", "vae_decoder"}
        assert entry["model_modifiers"] == {
            "dtype": "float32",
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "seed": 0,
            "height": 64,
            "width": 64,
            "scheduler": "DDIMScheduler",
        }
        assert entry["CLIPScore"]["scoring_version"] == 1
        assert "elapsed_ms" in entry["CLIPScore"]
        assert (tmp_path / "out" / "profiling_data.csv").exists()

    def test_metrics_share_generated_images(self, fake_parti, tmp_path, monkeypatch):
        from GenAILab.qai_hub_lm.diffusion.generator import DiffusionGenerator

        prompts = []
        original = DiffusionGenerator.generate
        monkeypatch.setattr(
            DiffusionGenerator,
            "generate",
            lambda self, prompt, seed: prompts.append(prompt)
            or original(self, prompt, seed),
        )
        monkeypatch.setattr(
            diffusion_metrics,
            "clip_scores",
            lambda images, texts, *a, **k: [1.0] * len(texts),
        )
        doc = {
            "model": {"model_id": TINY_SD, "num_inference_steps": 1},
            "metrics": [
                {"name": "CLIPScore", "num_prompts": 3},
                {"name": "ImageDump", "num_prompts": 3},
            ],
        }
        run_experiment(
            doc, fp_cache=DiskBackedFPCache(tmp_path / "fp"), results_dir=tmp_path
        )
        assert (
            len(prompts) == 3
        )  # Generated once, scored and dumped from the same images.

    def test_image_dump_config_output_dir_wins(self, fake_parti, tmp_path):
        doc = {
            "model": {"model_id": TINY_SD, "num_inference_steps": 1},
            "metrics": [
                {
                    "name": "ImageDump",
                    "num_prompts": 2,
                    "output_dir": str(tmp_path / "mine"),
                }
            ],
        }
        run_experiment(
            doc, fp_cache=DiskBackedFPCache(tmp_path / "fp"), results_dir=tmp_path
        )
        assert (tmp_path / "mine" / "grid.png").exists()


class TestEntryPoint:
    """``python -m GenAILab.bench.torch.run_diffusion``."""

    @staticmethod
    def _run(tmp_path, docs, *extra):
        import yaml

        from GenAILab.bench.torch import run_diffusion

        config = tmp_path / "cfg.yaml"
        config.write_text(yaml.safe_dump_all(docs))
        args = run_diffusion.build_parser().parse_args(
            [
                "--config", str(config),
                "--results-dir", str(tmp_path / "results"),
                "--fp-cache-dir", str(tmp_path / "fp"),
                *extra,
            ]
        )  # fmt: skip
        return run_diffusion.run(args)

    def test_runs_every_document(self, fake_parti, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            diffusion_metrics,
            "clip_scores",
            lambda images, texts, *a, **k: [1.0] * len(texts),
        )
        doc = {
            "model": {"model_id": TINY_SD, "num_inference_steps": 1},
            "metrics": [{"name": "CLIPScore", "num_prompts": 2}],
        }
        assert self._run(tmp_path, [doc, None, doc]) == 0
        data = json.loads((tmp_path / "results" / "profiling_data.json").read_text())
        assert len(data["StableDiffusionPipeline"]) == 2
        out = capsys.readouterr().out
        assert "2 passed, 0 failed" in out
        assert "GenAI Scorecard Summary" in out

    def test_failure_does_not_stop_later_documents(self, fake_parti, tmp_path, capsys):
        bad = {"model": {"model_id": TINY_SD}, "recipe": {"name": "Calibration"}}
        good = {
            "model": {"model_id": TINY_SD, "num_inference_steps": 1},
            "metrics": [{"name": "ImageDump", "num_prompts": 1}],
        }
        assert self._run(tmp_path, [bad, good]) == 1
        out = capsys.readouterr().out
        assert "1 passed, 1 failed" in out
        assert f"FAILED [1/2] {TINY_SD}" in out

    def test_fail_fast(self, tmp_path, capsys):
        bad = {"model": {"model_id": TINY_SD}, "recipe": {"name": "Calibration"}}
        assert self._run(tmp_path, [bad, bad], "--fail-fast") == 1
        assert "0 passed, 1 failed" in capsys.readouterr().out


def test_make_grid_layout():
    images = [np.full((2, 3, 3), i, dtype=np.uint8) for i in range(5)]
    grid = np.asarray(make_grid(images, columns=2))
    assert grid.shape == (6, 6, 3)
    assert grid[0, 3, 0] == 1 and grid[4, 0, 0] == 4 and grid[4, 3, 0] == 0


def test_image_dump_requires_output_dir():
    with pytest.raises(ValueError, match="needs an output_dir"):
        ImageDump.evaluate(object(), eval_ctx=None)


def test_clip_score_version():
    assert CLIPScore.SCORING_VERSION == 1
