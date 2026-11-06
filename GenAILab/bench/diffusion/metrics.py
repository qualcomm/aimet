# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation metrics for text-to-image diffusion models."""

from __future__ import annotations

import gc
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from GenAILab.bench.eval_context import EvaluationContext
from GenAILab.bench.metrics import EvaluationMetric
from GenAILab.bench.profiler import ScoredResult
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.diffusion.generator import DiffusionGenerator

PARTI_PROMPTS_REPO = "nateraw/parti-prompts"
#: Pinned so the prompt subset, and therefore the score, cannot drift upstream.
PARTI_PROMPTS_REVISION = "944b156abfdad7627c3221b5ec4f6a6fb060a197"
DEFAULT_NUM_PROMPTS = 64
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"


class ImageGenerationMetric(EvaluationMetric):
    """Base for metrics that score a :class:`DiffusionGenerator`'s images.

    ``evaluate(generator, *, eval_ctx, **kwargs)`` returns a float or a
    :class:`ScoredResult`. Generated images are memoized in ``eval_ctx`` per
    prompt set, so metrics sharing a prompt set generate once.
    """

    #: Marks a metric as usable with a text-to-image model.
    IMAGE_GENERATION = True

    @classmethod
    def evaluate(
        cls, generator: DiffusionGenerator, *, eval_ctx: EvaluationContext, **kwargs
    ):
        raise NotImplementedError


def load_parti_prompts(
    num_prompts: int = DEFAULT_NUM_PROMPTS, sample_seed: int = 0
) -> list[dict]:
    """A fixed, category-stratified subset of PartiPrompts.

    Each category contributes in proportion to its share of the full set,
    rounded by largest remainder. (At the default 64 prompts every one of the 12
    categories is represented.) Within a category, prompts are drawn with
    ``random.Random(sample_seed)``.

    Returns dicts with ``prompt`` and ``category``, ordered by category then draw.
    """
    from datasets import load_dataset

    rows = load_dataset(
        PARTI_PROMPTS_REPO, split="train", revision=PARTI_PROMPTS_REVISION
    )
    return stratified_sample(
        [{"prompt": r["Prompt"], "category": r["Category"]} for r in rows],
        num_prompts,
        sample_seed,
    )


def stratified_sample(
    rows: list[dict], num_prompts: int, sample_seed: int
) -> list[dict]:
    """Sample ``num_prompts`` of ``rows`` stratified by their ``category``."""
    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be positive, got {num_prompts}.")
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    if num_prompts >= len(rows):
        return [row for c in sorted(by_category) for row in by_category[c]]

    quotas = {c: num_prompts * len(v) / len(rows) for c, v in by_category.items()}
    counts = {c: int(q) for c, q in quotas.items()}
    leftover = num_prompts - sum(counts.values())
    for c in sorted(quotas, key=lambda c: (counts[c] - quotas[c], c))[:leftover]:
        counts[c] += 1

    rng = random.Random(sample_seed)
    sample = []
    for category in sorted(by_category):
        sample.extend(rng.sample(by_category[category], counts[category]))
    return sample


def generate_images(
    generator: DiffusionGenerator,
    prompts: list[dict],
    eval_ctx: EvaluationContext | None,
    collection_name: str,
) -> list[np.ndarray]:
    """Generate one image per prompt, memoized in ``eval_ctx`` under ``collection_name``."""

    def collect():
        base = generator.config.seed
        return [
            generator.generate(p["prompt"], seed=base + i)
            for i, p in enumerate(tqdm(prompts, desc="Generating images"))
        ]

    if eval_ctx is None:
        return collect()
    return eval_ctx.get_or_compute_quant(collection_name, collect)


def _prompt_set_name(num_prompts: int, sample_seed: int) -> str:
    return f"parti_prompts_n{num_prompts}_s{sample_seed}_images"


@YAMLConfigParser.register_metric
class CLIPScore(ImageGenerationMetric):
    """Mean CLIP image-text similarity over a fixed PartiPrompts subset.

    Per image, ``100 * max(cos(image_embed, text_embed), 0)`` -- the definition
    torchmetrics uses -- with prompts truncated to CLIP's 77 tokens. Higher is
    better. ``details`` breaks the score down by PartiPrompts category and lists
    every prompt's score.
    """

    SCORING_VERSION = 1

    @classmethod
    def evaluate(
        cls,
        generator: DiffusionGenerator,
        *,
        eval_ctx: EvaluationContext = None,
        num_prompts: int = DEFAULT_NUM_PROMPTS,
        sample_seed: int = 0,
        clip_model: str = DEFAULT_CLIP_MODEL,
    ) -> ScoredResult:
        prompts = load_parti_prompts(num_prompts, sample_seed)
        images = generate_images(
            generator, prompts, eval_ctx, _prompt_set_name(num_prompts, sample_seed)
        )
        scores = clip_scores(
            images, [p["prompt"] for p in prompts], clip_model, generator.device
        )

        by_category: dict[str, list[float]] = defaultdict(list)
        for prompt, score in zip(prompts, scores):
            by_category[prompt["category"]].append(score)
        return ScoredResult(
            result=float(np.mean(scores)),
            details={
                "clip_model": clip_model,
                "num_prompts": len(prompts),
                "by_category": {
                    c: float(np.mean(v)) for c, v in sorted(by_category.items())
                },
                "per_prompt": [
                    {"prompt": p["prompt"], "category": p["category"], "score": s}
                    for p, s in zip(prompts, scores)
                ],
            },
        )


@torch.inference_mode()
def clip_scores(
    images: list[np.ndarray],
    texts: list[str],
    clip_model: str = DEFAULT_CLIP_MODEL,
    device: torch.device | str = "cpu",
    batch_size: int = 16,
) -> list[float]:
    """Per-pair CLIP score, ``100 * max(cos, 0)``, for uint8 ``(H, W, 3)`` images."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model)
    scores = []
    try:
        for start in range(0, len(images), batch_size):
            inputs = processor(
                text=texts[start : start + batch_size],
                images=list(images[start : start + batch_size]),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            image_embeds = _features(
                model.get_image_features(pixel_values=inputs["pixel_values"])
            )
            text_embeds = _features(
                model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            )
            cosine = torch.nn.functional.cosine_similarity(
                image_embeds, text_embeds, dim=-1
            )
            scores.extend((100 * cosine.clamp(min=0)).float().cpu().tolist())
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores


def _features(output) -> torch.Tensor:
    """Projected embeddings from ``get_*_features``.

    transformers 5 returns a ``BaseModelOutputWithPooling`` whose
    ``pooler_output`` holds the projection; earlier versions return the tensor.
    """
    return output if isinstance(output, torch.Tensor) else output.pooler_output


@YAMLConfigParser.register_metric
class ImageDump(ImageGenerationMetric):
    """Save generated images, plus a contact sheet, for visual inspection.

    Writes ``<output_dir>/<i>.png`` and ``grid.png``. Without an ``output_dir``
    the experiment runner supplies one under the results directory. Shares its images with
    a ``CLIPScore`` over the same prompt set, so pairing the two costs nothing.
    Returns the number of images written.
    """

    @classmethod
    def evaluate(
        cls,
        generator: DiffusionGenerator,
        *,
        eval_ctx: EvaluationContext = None,
        num_prompts: int = 8,
        sample_seed: int = 0,
        output_dir: str | Path | None = None,
    ) -> ScoredResult:
        from PIL import Image

        if output_dir is None:
            raise ValueError("ImageDump needs an output_dir.")
        prompts = load_parti_prompts(num_prompts, sample_seed)
        images = generate_images(
            generator, prompts, eval_ctx, _prompt_set_name(num_prompts, sample_seed)
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            Image.fromarray(image).save(output_dir / f"{i:03d}.png")
        make_grid(images).save(output_dir / "grid.png")
        print(f"Saved {len(images)} images to {output_dir}")
        return ScoredResult(
            result=len(images),
            details={
                "output_dir": str(output_dir),
                "prompts": [p["prompt"] for p in prompts],
            },
        )


def make_grid(images: list[np.ndarray], columns: int = 4):
    """Tile equally-sized ``(H, W, 3)`` uint8 images into one PIL image."""
    from PIL import Image

    columns = min(columns, len(images))
    rows = -(-len(images) // columns)
    height, width, _ = images[0].shape
    grid = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, columns)
        grid[r * height : (r + 1) * height, c * width : (c + 1) * width] = image
    return Image.fromarray(grid)
