# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Metrics for GenAI testing"""

import contextlib
import gc
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import warnings
import yaml
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizer,
    GenerationConfig,
    TextStreamer,
    set_seed,
)
from transformers.processing_utils import ProcessorMixin
from transformers.generation.stopping_criteria import StoppingCriteriaList

from GenAILab.bench.yaml_config_parser import YAMLConfigParser, ResolvedMetric
from GenAILab.bench.eval_context import EvaluationContext
from GenAILab.bench.utils.prompt_utils import load_text_prompts, thinking_kwargs
from GenAILab.bench.utils.generation_utils import (
    build_generation_config,
    ContextLengthStoppingCriteria,
)
from GenAILab.qai_hub_lm.models.generator import Generator, VLM_Generator
from GenAILab.bench.profiler import GPUMeter, MetricResult, ScoredResult
from GenAILab.qai_hub_lm.scoring.grace.grace import (
    GRACE_VERSION,
    load_eval_prompts,
    select_balanced,
)
from GenAILab.qai_hub_lm.scoring.grace.grader import (
    MAX_POINTS,
)
from GenAILab.qai_hub_lm.scoring.grace.report import (
    detail_items,
)
from .datasets import (
    Wikitext,
    TinyMMLU as TinyMMLUDataset,
    MMLU as MMLUDataset,
    MMMLU as MMMLUDataset,
    MMLUPro as MMLUProDataset,
    MMMU as MMMUDataset,
    ERQA as ERQADataset,
    Where2Place as Where2PlaceDataset,
    LibriSpeech as LibriSpeechDataset,
)


class EvaluationMetric(ABC):
    """Base class for GenAI evaluation metrics.

    SCORING_VERSION identifies the metric's scoring semantics; bump it when
    they change (prompt, tokenization, filtering, aggregation). Absent/1
    means unchanged.
    """

    SCORING_VERSION: int = 1


class TextEvaluationMetric(EvaluationMetric):
    """Generic GenAI text evaluation metric"""

    @classmethod
    @abstractmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext,
    ) -> float | list[str]:
        """Perform evaluation on provided model"""


class DistanceMetric(EvaluationMetric):
    """Base class for metrics that compare quantized model outputs against FP baseline.

    This class is **modality-agnostic**.  Concrete distance metrics opt into a
    modality by also inheriting the appropriate evaluation base:

    *   Text-only:       ``class MMLUKLDiv(DistanceMetric, TextEvaluationMetric)``
        — receives the unwrapped tokenizer on VLMs.
    *   Multimodal:      ``class MMMUFlips(DistanceMetric)``
        — receives the full processor.

    Subclasses use the :class:`EvaluationContext` passed via the ``eval_ctx``
    keyword argument to cache and share intermediate results (e.g. logits)
    across multiple distance metrics without redundant forward passes.

    FP results are persisted to disk and shared across quantization recipes
    and pytest sessions.  Quant results are cached in-memory for the duration
    of a single test.
    """

    @classmethod
    @abstractmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext,
        **kwargs,
    ) -> float:
        """Compute a distance metric between quantized and FP model outputs."""


@YAMLConfigParser.register_metric
class PPL(TextEvaluationMetric):
    """PPL evaluation metric"""

    @staticmethod
    def _compute_loss_from_logits(
        output_logits: torch.Tensor, input_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Helper function to compute loss"""

        # Get the outputs and move it to CPU. Assumes that index 0 is logits as
        lm_logits = output_logits.cpu()

        # Trim the last logit off lm_logits, and the first token off input_tokens
        shift_logits = lm_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
        shift_labels = input_tokens[..., 1:].contiguous().to(shift_logits.device)

        loss_fn = torch.nn.CrossEntropyLoss()
        neg_log_likelihood = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return neg_log_likelihood

    @classmethod
    @torch.no_grad()
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        batch_size: int = 1,
        num_iterations: int = None,
    ) -> float:
        dataset = Wikitext.load_encoded_dataset(tokenizer, context_length, "test")
        dataloader = DataLoader(dataset, batch_size=batch_size)

        neg_log_likelihoods = []
        for i, batch in tqdm(
            enumerate(dataloader),
            total=num_iterations or len(dataloader),
            desc="Evaluating PPL",
        ):
            if num_iterations is not None and i >= num_iterations:
                break

            batch["input_ids"] = batch["input_ids"].to(model.device)
            outputs = model(input_ids=batch["input_ids"][0])
            neg_log_likelihoods.append(
                cls._compute_loss_from_logits(outputs[0], batch["input_ids"])
            )
            del outputs

        ppl = torch.exp(torch.stack(neg_log_likelihoods).mean())
        return float(ppl)


class GenericMMLU(TextEvaluationMetric):
    """Generic MMLU evaluation metric. Should work with any MMLU dataset."""

    @classmethod
    def get_collection_name(cls):
        """Get the collection name. Used for indexing into the EvaluationContext."""
        return f"{cls.__name__}_choice_logits"

    @staticmethod
    @abstractmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer, context_length: int
    ) -> DataLoader:
        """Get the dataloader associated with this MMLU evaluator."""

    @classmethod
    def collect_choice_logits(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        **kwargs,
    ) -> dict:
        """Run the model over this MMLU variant and collect per-sample data.

        Returns a dict with:

        * ``"logits"`` – ``Tensor(N, 4)`` of raw logits at the A/B/C/D token
          positions (before any softmax).
        * ``"labels"`` – ``Tensor(N,)`` of correct-answer indices (0–3).
        """
        kwargs.pop("image_size", None)
        dataloader = cls.get_dataloader(tokenizer, context_length, **kwargs)

        def tokenize_letter(letter: str):
            return torch.Tensor(
                tokenizer(letter, add_special_tokens=False)["input_ids"]
            ).to(dtype=torch.int)

        choices = tuple(tokenize_letter(letter) for letter in ("A", "B", "C", "D"))

        all_logits = []
        all_labels = []

        for batch in tqdm(
            dataloader, total=len(dataloader), desc=f"Collecting {cls.__name__} logits"
        ):
            batch["input_ids"] = (
                torch.Tensor(batch["input_ids"])
                .to(dtype=torch.int, device=model.device)
                .unsqueeze(0)
            )
            outputs = model(input_ids=batch["input_ids"])

            last_logit = (
                outputs[0][..., -1, :]
                .contiguous()
                .to(dtype=torch.float32, device="cpu")
                .flatten()
            )

            choice_logits = torch.tensor([last_logit[c].item() for c in choices])
            all_logits.append(choice_logits)

            label_token = torch.Tensor(batch["label"]).to(dtype=torch.int)
            label_idx = next(
                i for i, c in enumerate(choices) if torch.equal(c, label_token)
            )
            all_labels.append(label_idx)

            del outputs

        return {
            "logits": torch.stack(all_logits),
            "labels": torch.tensor(all_labels, dtype=torch.long),
        }

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        **kwargs,
    ) -> float:
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; MMLU logits will not be cached."
            )

        def collect_qt():
            return cls.collect_choice_logits(model, tokenizer, context_length, **kwargs)

        data = (
            eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect_qt)
            if eval_ctx
            else collect_qt()
        )
        preds = data["logits"].argmax(dim=-1)
        correct = (preds == data["labels"]).sum().item()
        return float(correct / len(data["labels"])) * 100


@YAMLConfigParser.register_metric
class TinyMMLU(GenericMMLU):
    @staticmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer, context_length: int
    ) -> DataLoader:
        dataset = TinyMMLUDataset.load_encoded_dataset(
            tokenizer, context_length, "test"
        )
        return DataLoader(dataset)


@YAMLConfigParser.register_metric
class MMLU(GenericMMLU):
    @staticmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        num_fewshot: int = 5,
    ) -> DataLoader:
        dataset = MMLUDataset.load_encoded_dataset(
            tokenizer, context_length, "test", num_fewshot=num_fewshot
        )
        return DataLoader(dataset)


@YAMLConfigParser.register_metric
class MMLU1000(GenericMMLU):
    @staticmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        num_fewshot: int = 5,
    ) -> DataLoader:
        dataset = MMLUDataset.load_encoded_dataset(
            tokenizer, context_length, "test", num_fewshot=num_fewshot
        )
        return DataLoader(Subset(dataset, torch.arange(1000)))


@YAMLConfigParser.register_metric
class MMMLU(GenericMMLU):
    @staticmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        split: str,
        num_fewshot: int = 5,
    ) -> DataLoader:
        dataset = MMMLUDataset.load_encoded_dataset(
            tokenizer, context_length, split, num_fewshot
        )
        return DataLoader(dataset)


@YAMLConfigParser.register_metric
class MMLUPro(TextEvaluationMetric):
    """MMLU Pro evaluation metric with 10 answer choices (A-J) using generative CoT."""

    SYSTEM_PROMPT = (
        "The following are multiple choice questions (with answers). "
        'Think step by step and then finish your answer with "the answer is (X)" '
        "where X is the correct letter choice."
    )

    @classmethod
    def get_collection_name(cls):
        """Get the collection name. Used for indexing into the EvaluationContext."""
        return f"{cls.__name__}_generated_answers"

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract answer from generated text using multi-stage regex patterns.

        Follows the MMLU Pro reference implementation with 3 fallback patterns:
        1. "answer is (A)" or "answer is A"
        2. "Answer: A" or "answer: A"
        3. Last standalone capital letter A-J
        """
        # First pattern: "answer is (A)" or "answer is A"
        pattern = r"answer is \(?([A-J])\)?"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Second pattern: "Answer: A" or "answer: A"
        match = re.search(r".*[aA]nswer:\s*([A-J])", text)
        if match:
            return match.group(1).upper()

        # Third pattern: last standalone letter A-J
        pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(0).upper()

        return None

    @classmethod
    def _generate_and_extract(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        dataloader: DataLoader,
    ) -> dict:
        """Generate answers with CoT reasoning and extract predictions.

        Returns a dict with:
        * ``"predictions"`` – List of extracted answers (or None if extraction failed)
        * ``"labels"`` – List of correct answers as letters
        """
        from GenAILab.bench.utils.generation_utils import build_generation_config

        # Build generation config for CoT (requires longer generations)
        generation_config = build_generation_config(
            model,
            tokenizer,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.0,
        )
        model.generation_config = generation_config

        predictions = []
        labels = []

        for batch in tqdm(
            dataloader, total=len(dataloader), desc="Generating MMLU Pro answers"
        ):
            # Get tokenized inputs from dataset
            input_ids = (
                torch.Tensor(batch["input_ids"])
                .to(dtype=torch.int, device=model.device)
                .unsqueeze(0)
            )
            attention_mask = (
                torch.Tensor(batch["attention_mask"])
                .to(dtype=torch.int, device=model.device)
                .unsqueeze(0)
            )
            label_token = torch.Tensor(batch["label"]).to(dtype=torch.int)

            # Get the prompt length for later extraction
            prompt_length = input_ids.shape[-1]

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )

            # Decode generated text (skip the prompt tokens)
            generated_tokens = outputs[0, prompt_length:]
            generated_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            # Extract answer from generated text
            pred = cls._extract_answer(generated_text)

            # Decode label token to letter
            label_letter = tokenizer.decode(
                label_token, skip_special_tokens=True
            ).strip()

            predictions.append(pred)
            labels.append(label_letter)

        return {
            "predictions": predictions,
            "labels": labels,
        }

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        num_fewshot: int = 5,
        num_samples: int = None,
        **kwargs,
    ) -> float:
        """Evaluate MMLU Pro accuracy with generative CoT reasoning.

        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            context_length: Maximum context length
            eval_ctx: Evaluation context for caching
            num_fewshot: Number of few-shot examples (default: 5)
            num_samples: Number of samples to evaluate (default: None, evaluates all ~12k samples)
        """
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; MMLU Pro results will not be cached."
            )

        dataset = MMLUProDataset.load_encoded_dataset(
            tokenizer, context_length, "test", num_fewshot=num_fewshot
        )
        if num_samples is not None:
            dataset = Subset(dataset, torch.arange(min(num_samples, len(dataset))))
        dataloader = DataLoader(dataset)

        def collect_qt():
            return cls._generate_and_extract(model, tokenizer, dataloader)

        data = (
            eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect_qt)
            if eval_ctx
            else collect_qt()
        )

        # Compute accuracy
        predictions = data["predictions"]
        labels = data["labels"]

        correct = 0
        wrong = 0
        for pred, label in zip(predictions, labels):
            if pred is None:
                # Failed extraction - treat as wrong
                wrong += 1
            elif pred == label:
                correct += 1
            else:
                wrong += 1

        if correct + wrong == 0:
            return 0.0

        return float(correct / (correct + wrong)) * 100


# ---------------------------------------------------------------------------
# MMLU distance metrics
# ---------------------------------------------------------------------------


class _MMLUDistanceBase(DistanceMetric, TextEvaluationMetric):
    """Shared MMLU data collection for all MMLU-based distance metrics.

    Subclasses only need to implement :meth:`_compute`.  The underlying MMLU
    forward passes (both FP and quantized) are run at most once and cached via
    the :class:`EvaluationContext`.  Data collection is delegated to
    :meth:`MMLU.collect_choice_logits` so the iteration logic lives in one
    place, and the quant collection is shared with the :class:`MMLU` accuracy
    metric when both appear in the same test config.
    """

    @classmethod
    def _get_mmlu_data(cls, model, tokenizer, context_length, eval_ctx, num_fewshot=5):
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; MMLU logits will not be cached."
            )

        # Use the same collection name as MMLU.evaluate so quant results are shared.
        collection = MMLU.get_collection_name()

        def collect_fp():
            with model.fp_mode():
                return MMLU.collect_choice_logits(
                    model, tokenizer, context_length, num_fewshot=num_fewshot
                )

        def collect_qt():
            return MMLU.collect_choice_logits(
                model, tokenizer, context_length, num_fewshot=num_fewshot
            )

        fp = (
            eval_ctx.get_or_compute_fp(collection, collect_fp)
            if eval_ctx
            else collect_fp()
        )
        q = (
            eval_ctx.get_or_compute_quant(collection, collect_qt)
            if eval_ctx
            else collect_qt()
        )
        return fp, q

    @classmethod
    @abstractmethod
    def _compute(cls, fp_data: dict, q_data: dict) -> float:
        """Compute the metric from collected FP and quantized MMLU data."""

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        num_fewshot: int = 5,
        **kwargs,
    ):
        fp, q = cls._get_mmlu_data(
            model, tokenizer, context_length, eval_ctx, num_fewshot
        )
        return cls._compute(fp, q)


# ---------------------------------------------------------------------------
# Shared distance-metric computation mixins
# ---------------------------------------------------------------------------


class _KLDivergenceCompute:
    """KL divergence KL(P_fp || P_quant) over answer choice distributions."""

    @classmethod
    def _compute(cls, fp_data, q_data):
        p = torch.nn.functional.softmax(fp_data["logits"], dim=-1)
        log_q = torch.nn.functional.log_softmax(q_data["logits"], dim=-1)
        return torch.nn.functional.kl_div(log_q, p, reduction="batchmean").item()


class _ReverseKLDivergenceCompute:
    """Reverse KL divergence KL(P_quant || P_fp) over answer choice distributions."""

    @classmethod
    def _compute(cls, fp_data, q_data):
        q = torch.nn.functional.softmax(q_data["logits"], dim=-1)
        log_p = torch.nn.functional.log_softmax(fp_data["logits"], dim=-1)
        return torch.nn.functional.kl_div(log_p, q, reduction="batchmean").item()


class _FlipsCompute:
    """Percentage of samples where quantized and FP predictions disagree."""

    @classmethod
    def _compute(cls, fp_data, q_data):
        fp_preds = fp_data.get("preds", fp_data["logits"].argmax(dim=-1))
        q_preds = q_data.get("preds", q_data["logits"].argmax(dim=-1))
        return (fp_preds != q_preds).float().mean().item() * 100


class _JSDivergenceCompute:
    """Jensen-Shannon divergence between FP and quantized distributions."""

    @classmethod
    def _compute(cls, fp_data, q_data):
        p = torch.nn.functional.softmax(fp_data["logits"], dim=-1)
        q = torch.nn.functional.softmax(q_data["logits"], dim=-1)
        m = 0.5 * (p + q)
        kl_pm = torch.nn.functional.kl_div(m.log(), p, reduction="batchmean")
        kl_qm = torch.nn.functional.kl_div(m.log(), q, reduction="batchmean")
        return (0.5 * (kl_pm + kl_qm)).item()


@YAMLConfigParser.register_metric
class MMLUKLDivergence(_KLDivergenceCompute, _MMLUDistanceBase):
    """KL divergence KL(P_fp || P_quant) over MMLU answer choice distributions."""


@YAMLConfigParser.register_metric
class MMLUReverseKLDivergence(_ReverseKLDivergenceCompute, _MMLUDistanceBase):
    """Reverse KL divergence KL(P_quant || P_fp) over MMLU answer choice distributions."""


@YAMLConfigParser.register_metric
class MMLUFlips(_FlipsCompute, _MMLUDistanceBase):
    """Percentage of MMLU samples where quantized and FP predictions disagree."""


@YAMLConfigParser.register_metric
class MMLUJSDivergence(_JSDivergenceCompute, _MMLUDistanceBase):
    """Jensen-Shannon divergence between FP and quantized MMLU distributions."""


# ---------------------------------------------------------------------------
# MMMU metrics (multimodal)
# ---------------------------------------------------------------------------


@YAMLConfigParser.register_metric
class MMMU(EvaluationMetric):
    """Generic MMMU evaluation metric for multimodal models.

    v2 (2026-07): "Answer:" moved to an assistant turn (LazyMMMUDataset) and
    letter tokens resolved via the space-prefixed form; not comparable to v1.
    """

    SCORING_VERSION = 2

    @classmethod
    def get_collection_name(cls):
        """Get the collection name. Used for indexing into the EvaluationContext."""
        return f"{cls.__name__}_choice_logits"

    @staticmethod
    def get_dataset(processor, context_length, image_size=None, **kwargs):
        return MMMUDataset.load_encoded_dataset(
            processor, context_length, split="validation", image_size=image_size
        )

    @staticmethod
    def _token_id(tokenizer, letter):
        """Resolve the vocab id for an answer letter (space-prefixed, then bare fallback)."""
        tok_ids = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
        if len(tok_ids) != 1:
            tok_ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
        return tok_ids[0]

    @classmethod
    def collect_choice_logits(cls, model, processor, context_length, **kwargs) -> dict:
        """Run the model over MMMU and collect per-sample choice logits.

        Returns a dict with:

        * ``"logits"`` – ``Tensor(N, max_options)`` of raw logits at each
          answer-letter token position, padded with ``-inf`` for samples that
          have fewer options than the maximum.
        * ``"preds"``  – ``Tensor(N,)`` of predicted answer indices.
        * ``"labels"`` – ``Tensor(N,)`` of correct-answer indices.
        """
        dataset = cls.get_dataset(processor, context_length, **kwargs)

        tokenizer = getattr(processor, "tokenizer", processor)

        all_logits = []  # variable-length per sample, padded later
        all_preds = []
        all_labels = []

        for sample in tqdm(dataset, desc=f"Collecting {cls.__name__} logits"):
            num_options = sample.pop("num_options", 4)
            label = sample.pop("label")

            inputs = {
                k: v.to(model.device)
                for k, v in sample.items()
                if isinstance(v, torch.Tensor)
            }
            outputs = model(**inputs)

            last_logit = (
                outputs[0][..., -1, :]
                .contiguous()
                .to(dtype=torch.float32, device="cpu")
                .flatten()
            )

            # Only compare logits for the actual number of options
            choice_letters = [chr(65 + i) for i in range(num_options)]
            choice_ids = [cls._token_id(tokenizer, c) for c in choice_letters]
            choice_logits = torch.tensor([last_logit[c].item() for c in choice_ids])

            all_logits.append(choice_logits)
            all_preds.append(choice_logits.argmax().item())
            all_labels.append(ord(label.strip().upper()) - ord("A"))

            del outputs, inputs
            torch.cuda.empty_cache()

        # Pad logits to the maximum number of options with -inf so they can be
        # stacked into a single tensor.  Softmax(-inf) == 0 so padded positions
        # contribute nothing to KL / JS divergence computations.
        max_options = max(l.size(0) for l in all_logits) if all_logits else 4
        padded = []
        for logit in all_logits:
            pad_len = max_options - logit.size(0)
            if pad_len > 0:
                logit = torch.cat([logit, logit.new_full((pad_len,), float("-inf"))])
            padded.append(logit)

        return {
            "logits": torch.stack(padded),
            "preds": torch.tensor(all_preds, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
        }

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        **kwargs,
    ) -> float:
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; MMLU logits will not be cached."
            )

        def collect_qt():
            return cls.collect_choice_logits(model, processor, context_length, **kwargs)

        data = (
            eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect_qt)
            if eval_ctx
            else collect_qt()
        )
        correct = (data["preds"] == data["labels"]).sum().item()
        return float(correct / len(data["labels"])) * 100


# ---------------------------------------------------------------------------
# MMMU distance metrics
# ---------------------------------------------------------------------------


class _MMMUDistanceBase(DistanceMetric):
    """Shared MMMU data collection for all MMMU-based distance metrics"""

    @classmethod
    def _get_mmmu_data(
        cls, model, processor, context_length, eval_ctx, image_size=None
    ):
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; MMLU logits will not be cached."
            )

        collection = MMMU.get_collection_name()

        def collect_fp():
            with model.fp_mode():
                return MMMU.collect_choice_logits(
                    model, processor, context_length, image_size=image_size
                )

        def collect_qt():
            return MMMU.collect_choice_logits(
                model, processor, context_length, image_size=image_size
            )

        fp = (
            eval_ctx.get_or_compute_fp(collection, collect_fp)
            if eval_ctx
            else collect_fp()
        )
        q = (
            eval_ctx.get_or_compute_quant(collection, collect_qt)
            if eval_ctx
            else collect_qt()
        )
        return fp, q

    @classmethod
    @abstractmethod
    def _compute(cls, fp_data: dict, q_data: dict) -> float:
        """Compute the metric from collected FP and quantized MMMU data."""

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        num_fewshot: int = 5,
        image_size: tuple[int, int] | None = None,
        **kwargs,
    ):
        fp, q = cls._get_mmmu_data(
            model, processor, context_length, eval_ctx, image_size=image_size
        )
        return cls._compute(fp, q)


@YAMLConfigParser.register_metric
class MMMUKLDivergence(_KLDivergenceCompute, _MMMUDistanceBase):
    """KL divergence KL(P_fp || P_quant) over MMMU answer choice distributions."""


@YAMLConfigParser.register_metric
class MMMUReverseKLDivergence(_ReverseKLDivergenceCompute, _MMMUDistanceBase):
    """Reverse KL divergence KL(P_quant || P_fp) over MMMU answer choice distributions."""


@YAMLConfigParser.register_metric
class MMMUFlips(_FlipsCompute, _MMMUDistanceBase):
    """Percentage of MMMU samples where quantized and FP predictions disagree."""


@YAMLConfigParser.register_metric
class MMMUJSDivergence(_JSDivergenceCompute, _MMMUDistanceBase):
    """Jensen-Shannon divergence between FP and quantized MMMU distributions."""


# ---------------------------------------------------------------------------
# ERQA metrics (multimodal, multi-image MCQ -- same choice-logit shape as MMMU)
# ---------------------------------------------------------------------------


@YAMLConfigParser.register_metric
class ERQA(EvaluationMetric):
    """Embodied Reasoning QA: multi-image, 4-way A-D MCQ, test-only (400 rows).

    Choice-logit scoring, identical mechanism to MMMU -- reuses MMMU's
    letter-token resolution rather than duplicating it.
    """

    SCORING_VERSION = 1

    @classmethod
    def get_collection_name(cls):
        return f"{cls.__name__}_choice_logits"

    @staticmethod
    def get_dataset(processor, context_length, image_size=None, **kwargs):
        return ERQADataset.load_encoded_dataset(
            processor, context_length, split="test", image_size=image_size
        )

    @classmethod
    def collect_choice_logits(cls, model, processor, context_length, **kwargs) -> dict:
        """Run the model over ERQA and collect per-sample choice logits.

        Same shape as ``MMMU.collect_choice_logits`` -- see there for the
        field-by-field explanation. ERQA is always a 4-way A-D MCQ (unlike
        MMMU's variable option count).
        """
        dataset = cls.get_dataset(processor, context_length, **kwargs)

        tokenizer = getattr(processor, "tokenizer", processor)

        all_logits = []
        all_preds = []
        all_labels = []

        for sample in tqdm(dataset, desc=f"Collecting {cls.__name__} logits"):
            num_options = sample.pop("num_options", 4)
            label = sample.pop("label")

            inputs = {
                k: v.to(model.device)
                for k, v in sample.items()
                if isinstance(v, torch.Tensor)
            }
            outputs = model(**inputs)

            last_logit = (
                outputs[0][..., -1, :]
                .contiguous()
                .to(dtype=torch.float32, device="cpu")
                .flatten()
            )

            choice_letters = [chr(65 + i) for i in range(num_options)]
            choice_ids = [MMMU._token_id(tokenizer, c) for c in choice_letters]
            choice_logits = torch.tensor([last_logit[c].item() for c in choice_ids])

            all_logits.append(choice_logits)
            all_preds.append(choice_logits.argmax().item())
            all_labels.append(ord(label.strip().upper()) - ord("A"))

            del outputs, inputs
            torch.cuda.empty_cache()

        return {
            "logits": torch.stack(all_logits),
            "preds": torch.tensor(all_preds, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
        }

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        **kwargs,
    ) -> float:
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; ERQA logits will not be cached."
            )

        def collect_qt():
            return cls.collect_choice_logits(model, processor, context_length, **kwargs)

        data = (
            eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect_qt)
            if eval_ctx
            else collect_qt()
        )
        correct = (data["preds"] == data["labels"]).sum().item()
        return float(correct / len(data["labels"])) * 100


class _ERQADistanceBase(DistanceMetric):
    """Shared ERQA data collection for all ERQA-based distance metrics"""

    @classmethod
    def _get_erqa_data(
        cls, model, processor, context_length, eval_ctx, image_size=None
    ):
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; ERQA logits will not be cached."
            )

        collection = ERQA.get_collection_name()

        def collect_fp():
            with model.fp_mode():
                return ERQA.collect_choice_logits(
                    model, processor, context_length, image_size=image_size
                )

        def collect_qt():
            return ERQA.collect_choice_logits(
                model, processor, context_length, image_size=image_size
            )

        fp = (
            eval_ctx.get_or_compute_fp(collection, collect_fp)
            if eval_ctx
            else collect_fp()
        )
        q = (
            eval_ctx.get_or_compute_quant(collection, collect_qt)
            if eval_ctx
            else collect_qt()
        )
        return fp, q

    @classmethod
    @abstractmethod
    def _compute(cls, fp_data: dict, q_data: dict) -> float:
        """Compute the metric from collected FP and quantized ERQA data."""

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        image_size: tuple[int, int] | None = None,
        **kwargs,
    ):
        fp, q = cls._get_erqa_data(
            model, processor, context_length, eval_ctx, image_size=image_size
        )
        return cls._compute(fp, q)


@YAMLConfigParser.register_metric
class ERQAKLDivergence(_KLDivergenceCompute, _ERQADistanceBase):
    """KL divergence KL(P_fp || P_quant) over ERQA answer choice distributions."""


@YAMLConfigParser.register_metric
class ERQAReverseKLDivergence(_ReverseKLDivergenceCompute, _ERQADistanceBase):
    """Reverse KL divergence KL(P_quant || P_fp) over ERQA answer choice distributions."""


@YAMLConfigParser.register_metric
class ERQAFlips(_FlipsCompute, _ERQADistanceBase):
    """Percentage of ERQA samples where quantized and FP predictions disagree."""


@YAMLConfigParser.register_metric
class ERQAJSDivergence(_JSDivergenceCompute, _ERQADistanceBase):
    """Jensen-Shannon divergence between FP and quantized ERQA distributions."""


class TimedStreamer(TextStreamer):
    """TextStreamer that records prefill and decode timing stats."""

    def __init__(self, *args, num_input_tokens: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = 0
        self.first_token_time = None
        self.end_time = None
        self.start_time = time.perf_counter()

    def put(self, value):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.num_output_tokens += value.numel()
        super().put(value)

    def end(self):
        self.end_time = time.perf_counter()
        super().end()
        self._print_stats()

    def _print_stats(self):
        ttft = (
            self.first_token_time - self.start_time if self.first_token_time else None
        )
        decode_tokens = max(self.num_output_tokens - 1, 0)
        decode_time = (
            (self.end_time - self.first_token_time) if self.first_token_time else 0
        )

        print(f"\n--- Generation Stats ---")
        print(f"  Input tokens:  {self.num_input_tokens}")
        print(f"  Output tokens: {self.num_output_tokens}")
        if ttft is not None:
            print(
                f"  TTFT:          {ttft:.3f}s  ({self.num_input_tokens / ttft:.1f} prefill tok/s)"
            )
        if decode_time > 0 and decode_tokens > 0:
            print(
                f"  Decode:        {decode_time:.3f}s  ({decode_tokens / decode_time:.1f} tok/s)"
            )
        print(f"  Total:         {self.end_time - self.start_time:.3f}s")


@YAMLConfigParser.register_metric
class Interactive(TextEvaluationMetric):
    @staticmethod
    def _get_generation_config(model, tokenizer, **overrides) -> GenerationConfig:
        """Build a GenerationConfig with EOS tokens merged from model config and tokenizer."""
        return build_generation_config(model, tokenizer, **overrides)

    @staticmethod
    def _build_stopping_criteria(model: Generator, verbose: bool = False):
        criteria = [
            ContextLengthStoppingCriteria(
                context_length=model.context_length,
                sequence_lengths=model.sequence_lengths,
                verbose=verbose,
            )
        ]
        return StoppingCriteriaList(criteria) if criteria else None

    @staticmethod
    def get_system_prompt() -> str:
        return "You are a helpful AI assistant."

    @classmethod
    def generate_output(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        unformatted_prompt: str = None,
        formatted_prompt: str = None,
        generation_config: GenerationConfig = None,
        highlight_output: bool = False,
    ) -> str:
        if formatted_prompt is None and unformatted_prompt is None:
            raise ValueError(
                "Either unformatted_prompt or formatted_prompt must be provided."
            )
        if formatted_prompt is not None and unformatted_prompt is not None:
            raise ValueError(
                "Only one of unformatted_prompt or formatted_prompt should be provided."
            )

        if formatted_prompt is None:
            formatted_prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": cls.get_system_prompt()},
                    {"role": "user", "content": unformatted_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        tokenized_user_input = tokenizer(formatted_prompt, return_tensors="pt").to(
            model.device
        )

        model.generation_config = (
            generation_config
            if generation_config is not None
            else cls._get_generation_config(model, tokenizer)
        )

        print(formatted_prompt, end="")
        if highlight_output:
            print("\033[0;31m", end="")  # Start red color for output

        streamer = TimedStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            num_input_tokens=tokenized_user_input["input_ids"].shape[-1],
        )
        stopping_criteria = cls._build_stopping_criteria(model, verbose=True)
        outputs = model.generate(
            inputs=tokenized_user_input["input_ids"],
            attention_mask=tokenized_user_input["attention_mask"],
            generation_config=model.generation_config,
            stopping_criteria=stopping_criteria,
            streamer=streamer,
        )

        if highlight_output:
            print("\033[0m")  # Reset color after highlighted output

        # Detokenize and return the generated string
        generated_tokens = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        return generated_text

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
    ) -> float:
        while True:
            user_input_prompt = input("Enter your prompt or 'exit' to quit: ")
            if user_input_prompt == "exit":
                break
            cls.generate_output(model, tokenizer, unformatted_prompt=user_input_prompt)
        return float("nan")


@YAMLConfigParser.register_metric
class TrickyPrompts(Interactive):
    prompts = {
        "phi3": [
            "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\nWhat is Gravity?<|end|>\n<|assistant|>\nGravity is a fundamental force of nature that attracts two bodies with mass towards each other. It is described by Isaac Newton'",
            "<|system|>\nYou are a helpful AI assistant.<|end|>\n<|user|>\nWhat is Gravity?<|end|>\n<|assistant|>\nGravity is a fundamental force of nature that attracts two bodies with mass towards each other. It is described by Isaac Newton's theory in the 17th century and is a key component in Albert Einstein'",
        ]
    }

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
    ) -> list[str]:
        generated_text = []
        for prompt in TrickyPrompts.prompts.get(model.config.model_type, []):
            print("===============================")
            generated_text.append(
                cls.generate_output(
                    model,
                    tokenizer,
                    formatted_prompt=prompt,
                    generation_config=cls._get_generation_config(
                        model,
                        tokenizer,
                        max_new_tokens=2,
                        do_sample=False,
                    ),
                    highlight_output=True,
                )
            )
        print("===============================")
        return generated_text


@YAMLConfigParser.register_metric
class Prompts(Interactive):
    @classmethod
    def get_collection_name(cls):
        return f"{cls.__name__}_generated_text"

    @classmethod
    def _generate_all(cls, model, tokenizer):
        prompts = load_text_prompts()
        generated_text = []
        for prompt in prompts:
            print("===============================")
            generated_text.append(
                cls.generate_output(
                    model=model,
                    tokenizer=tokenizer,
                    unformatted_prompt=prompt,
                    generation_config=cls._get_generation_config(
                        model, tokenizer, do_sample=False
                    ),
                )
            )
        print("===============================")
        return {"prompts": prompts, "generated_text": generated_text}

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
    ) -> list[str]:
        def collect():
            return cls._generate_all(model, tokenizer)

        if eval_ctx is not None:
            data = eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect)
        else:
            data = collect()

        return data["generated_text"]


@YAMLConfigParser.register_metric
class MultimodalPrompts(EvaluationMetric):
    PROMPTS_FILE = Path(__file__).parent / "prompts" / "multimodal_prompts.yaml"
    IMAGE_DIR = Path(__file__).parent / "prompts" / "sample_images"

    @classmethod
    def get_collection_name(cls):
        return f"{cls.__name__}_generated_text"

    @classmethod
    def _load_prompts(cls):
        with open(cls.PROMPTS_FILE) as f:
            return yaml.safe_load(f)

    @classmethod
    def _generate_all(cls, model, processor):
        from PIL import Image

        if model.generation_config is None:
            model.generation_config = GenerationConfig()

        tokenizer = getattr(processor, "tokenizer", processor)
        prompts = cls._load_prompts()
        generated_text = []

        for entry in prompts:
            image_file = entry["image"]
            prompt_text = entry["prompt"]
            print("===============================")
            image_path = cls.IMAGE_DIR / image_file
            image = Image.open(image_path).convert("RGB")
            if model.image_size is not None:
                image = image.resize(model.image_size)

            content = [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]
            messages = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            inputs.pop("mm_token_type_ids", None)

            generation_config = Interactive._get_generation_config(
                model,
                tokenizer,
                do_sample=False,
            )

            streamer = TimedStreamer(
                tokenizer=tokenizer,
                skip_prompt=True,
                num_input_tokens=inputs["input_ids"].shape[-1],
            )
            stopping_criteria = Interactive._build_stopping_criteria(
                model, verbose=True
            )
            print(text, end="")
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )

            generated_tokens = (
                outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            )
            result = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            generated_text.append(result)

        print("===============================")
        return {"prompts": prompts, "generated_text": generated_text}

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        **kwargs,
    ) -> list[str]:
        if not isinstance(model, VLM_Generator):
            raise ValueError("MultimodalPrompts metric requires a VL model.")

        def collect():
            return cls._generate_all(model, processor)

        if eval_ctx is not None:
            data = eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect)
        else:
            data = collect()

        return data["generated_text"]


# ---------------------------------------------------------------------------
# Where2Place metric (multimodal, free-form pointing -- not choice-logit shaped)
# ---------------------------------------------------------------------------


@YAMLConfigParser.register_metric
class Where2Place(EvaluationMetric):
    """RoboPoint-style free-space pointing accuracy.

    Each question asks the model to name several candidate points in a
    described free-space region, formatted as ``[(x1, y1), (x2, y2), ...]``
    with coordinates normalized to [0, 1]. Scored by the standard RoboPoint
    convention: a sample counts as correct if *any* predicted point falls
    inside the annotated ground-truth mask.

    v1: generation-based (``model.generate`` + free-text point parsing), not
    choice-logit-based -- there is no MCQ letter here. Consequently there is no
    KL/Flips/JS distance-metric family: those mixins diff a choice-logit
    distribution, and a continuous point output has none.
    """

    SCORING_VERSION = 1

    _POINT_RE = re.compile(r"\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)")

    @classmethod
    def get_collection_name(cls):
        return f"{cls.__name__}_generated_points"

    @staticmethod
    def get_dataset(processor, context_length, image_size=None, **kwargs):
        return Where2PlaceDataset.load_encoded_dataset(
            processor, context_length, split="train", image_size=image_size
        )

    @classmethod
    def _parse_points(cls, text: str) -> list[tuple[float, float]]:
        """Extract normalized (x, y) points from free-form generated text.

        The prompt's own "(x1, y1)" instruction template has no digits, so
        scanning the whole decoded string for "(number, number)" pairs (rather
        than anchoring to the text after a final "[") does not pick up false
        positives from the instruction itself. Out-of-range coordinates are
        clamped to [0, 1] rather than discarded, mirroring
        ``grace.grader.parse_rating``'s clamp-not-discard convention.
        """
        points = []
        for x_str, y_str in cls._POINT_RE.findall(text):
            x = min(1.0, max(0.0, float(x_str)))
            y = min(1.0, max(0.0, float(y_str)))
            points.append((x, y))
        return points

    @staticmethod
    def _any_point_in_mask(points: list[tuple[float, float]], mask: np.ndarray) -> bool:
        mask_height, mask_width = mask.shape
        for x, y in points:
            px = min(mask_width - 1, int(x * mask_width))
            py = min(mask_height - 1, int(y * mask_height))
            if mask[py, px]:
                return True
        return False

    @classmethod
    def _generate_all(cls, model, processor, context_length, **kwargs) -> dict:
        if model.generation_config is None:
            model.generation_config = GenerationConfig()

        tokenizer = getattr(processor, "tokenizer", processor)
        dataset = cls.get_dataset(processor, context_length, **kwargs)

        hits = []
        for sample in tqdm(dataset, desc=f"Collecting {cls.__name__} generations"):
            mask = sample.pop("mask")
            sample.pop("mask_size", None)

            inputs = {
                k: v.to(model.device)
                for k, v in sample.items()
                if isinstance(v, torch.Tensor)
            }
            generation_config = Interactive._get_generation_config(
                model, tokenizer, do_sample=False
            )
            stopping_criteria = Interactive._build_stopping_criteria(model)
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )

            generated_tokens = (
                outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            )
            new_tokens = generated_tokens[0][inputs["input_ids"].shape[-1] :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            points = cls._parse_points(text)
            hits.append(bool(points) and cls._any_point_in_mask(points, mask))

            del outputs, inputs
            torch.cuda.empty_cache()

        return {"hits": torch.tensor(hits, dtype=torch.bool)}

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        **kwargs,
    ) -> float:
        if not isinstance(model, VLM_Generator):
            raise ValueError("Where2Place metric requires a VL model.")
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; Where2Place generations will not be cached."
            )

        def collect_qt():
            return cls._generate_all(model, processor, context_length, **kwargs)

        data = (
            eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect_qt)
            if eval_ctx
            else collect_qt()
        )
        return data["hits"].float().mean().item() * 100


@YAMLConfigParser.register_metric
class AutogradedPrompts(TextEvaluationMetric):
    """Grade generated responses with a small LLM as a 4-way classifier (A/B/C/D).

    For each prompt/response pair, a single forward pass is run through the
    grader model. The logits at the last token position are read and argmax is
    taken over the four letter-token IDs. Points are assigned per the harness
    config and the final score is reported as a percentage of max possible.
    """

    HARNESS_FILE = (
        Path(__file__).parent / "prompts" / "text_prompts_autograder_harness.yaml"
    )
    LETTERS = ("A", "B", "C", "D")
    DEFAULT_HARNESS_VERSION = "v1"

    @classmethod
    def _load_harness(cls, version: str = DEFAULT_HARNESS_VERSION):
        with open(cls.HARNESS_FILE) as f:
            harness = yaml.safe_load(f)
        return harness[version]

    @staticmethod
    def _get_letter_token_ids(tokenizer) -> list[int]:
        ids = []
        for letter in AutogradedPrompts.LETTERS:
            tok_ids = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
            if len(tok_ids) != 1:
                tok_ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
            if len(tok_ids) != 1:
                raise ValueError(
                    f"Letter {letter!r} tokenizes to {len(tok_ids)} tokens; "
                    f"grader needs single-token letters."
                )
            ids.append(tok_ids[0])
        if len(set(ids)) != 4:
            raise ValueError(f"Letter token ids collided: {ids}")
        return ids

    @classmethod
    def _score_one(
        cls,
        grader_model,
        grader_tokenizer,
        grading_prompt,
        prompt,
        response,
        letter_ids,
    ) -> str:
        text = grading_prompt.replace("{prompt}", prompt).replace(
            "{response}", response
        )
        messages = [
            {"role": "user", "content": text},
            {"role": "assistant", "content": ""},
        ]
        formatted = grader_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            **thinking_kwargs(grader_tokenizer.chat_template),
        )
        input_ids = grader_tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(grader_model.device)
        outputs = grader_model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :].float().cpu()
        choice_logits = {
            letter: logits[tok_id].item()
            for letter, tok_id in zip(cls.LETTERS, letter_ids)
        }
        return max(choice_logits, key=choice_logits.get)

    @classmethod
    @torch.no_grad()
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        harness_version: str = DEFAULT_HARNESS_VERSION,
        **kwargs,
    ) -> float:
        def collect():
            return Prompts._generate_all(model, tokenizer)

        if eval_ctx is not None:
            data = eval_ctx.get_or_compute_quant(Prompts.get_collection_name(), collect)
        else:
            data = collect()

        prompts = data["prompts"]
        responses = data["generated_text"]

        harness = cls._load_harness(harness_version)
        grader_model_id = harness["model_id"]
        grading_prompt = harness["grading_prompt"]
        letter_points = harness["letter_points"]
        max_points = max(letter_points.values())

        from transformers import AutoModelForCausalLM, AutoTokenizer

        with model.on_device(torch.device("cpu")):
            grader_tokenizer = AutoTokenizer.from_pretrained(grader_model_id)
            grader_model = AutoModelForCausalLM.from_pretrained(
                grader_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            grader_model.eval()
            letter_ids = cls._get_letter_token_ids(grader_tokenizer)

            total_points = 0
            for prompt, response in tqdm(
                zip(prompts, responses),
                total=len(prompts),
                desc="Autograding responses",
            ):
                pred = cls._score_one(
                    grader_model,
                    grader_tokenizer,
                    grading_prompt,
                    prompt,
                    response,
                    letter_ids,
                )
                total_points += letter_points[pred]

            del grader_model
            del grader_tokenizer

            gc.collect()
            torch.cuda.empty_cache()

        return 100.0 * total_points / (max_points * len(prompts))


@YAMLConfigParser.register_metric
class AutogradedMultimodalPrompts(EvaluationMetric):
    """Grade VLM responses with an external VLM as a 4-way classifier (A/B/C/D).

    The grader model receives both the source image and the generated response.
    Scoring uses single-forward-pass argmax over letter tokens, same as
    AutogradedPrompts.
    """

    HARNESS_FILE = (
        Path(__file__).parent / "prompts" / "multimodal_prompts_autograder_harness.yaml"
    )
    IMAGE_DIR = Path(__file__).parent / "prompts" / "sample_images"
    LETTERS = ("A", "B", "C", "D")
    DEFAULT_HARNESS_VERSION = "v1"

    @classmethod
    def _load_harness(cls, version: str = DEFAULT_HARNESS_VERSION):
        with open(cls.HARNESS_FILE) as f:
            harness = yaml.safe_load(f)
        return harness[version]

    @staticmethod
    def _get_letter_token_ids(tokenizer) -> list[int]:
        ids = []
        for letter in AutogradedMultimodalPrompts.LETTERS:
            tok_ids = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"]
            if len(tok_ids) != 1:
                tok_ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
            if len(tok_ids) != 1:
                raise ValueError(
                    f"Letter {letter!r} tokenizes to {len(tok_ids)} tokens; "
                    f"grader needs single-token letters."
                )
            ids.append(tok_ids[0])
        if len(set(ids)) != 4:
            raise ValueError(f"Letter token ids collided: {ids}")
        return ids

    @classmethod
    def _score_one(
        cls,
        grader_model,
        grader_processor,
        grading_prompt,
        prompt_text,
        response,
        image,
        letter_ids,
    ):
        text = grading_prompt.replace("{prompt}", prompt_text).replace(
            "{response}", response
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            },
            {"role": "assistant", "content": ""},
        ]
        formatted = grader_processor.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            **thinking_kwargs(grader_processor.chat_template),
        )
        inputs = grader_processor(text=[formatted], images=[image], return_tensors="pt")
        inputs = {k: v.to(grader_model.device) for k, v in inputs.items()}

        outputs = grader_model(**inputs)
        logits = outputs.logits[0, -1, :].float().cpu()
        choice_logits = {
            letter: logits[tok_id].item()
            for letter, tok_id in zip(cls.LETTERS, letter_ids)
        }
        return max(choice_logits, key=choice_logits.get)

    @classmethod
    @torch.no_grad()
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        harness_version: str = DEFAULT_HARNESS_VERSION,
        **kwargs,
    ) -> float:
        if not isinstance(model, VLM_Generator):
            raise ValueError("AutogradedMultimodalPrompts requires a VL model.")

        def collect():
            return MultimodalPrompts._generate_all(model, processor)

        if eval_ctx is not None:
            data = eval_ctx.get_or_compute_quant(
                MultimodalPrompts.get_collection_name(), collect
            )
        else:
            data = collect()

        prompts = data["prompts"]
        responses = data["generated_text"]

        harness = cls._load_harness(harness_version)
        grader_model_id = harness["model_id"]
        grading_prompt = harness["grading_prompt"]
        letter_points = harness["letter_points"]
        max_points = max(letter_points.values())

        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        with model.on_device(torch.device("cpu")):
            grader_processor = AutoProcessor.from_pretrained(grader_model_id)
            grader_model = AutoModelForImageTextToText.from_pretrained(
                grader_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            grader_model.eval()
            grader_tokenizer = getattr(grader_processor, "tokenizer", grader_processor)
            letter_ids = cls._get_letter_token_ids(grader_tokenizer)

            total_points = 0
            for entry, response in tqdm(
                zip(prompts, responses),
                total=len(prompts),
                desc="Autograding multimodal responses",
            ):
                image_path = cls.IMAGE_DIR / entry["image"]
                image = Image.open(image_path).convert("RGB")

                pred = cls._score_one(
                    grader_model,
                    grader_processor,
                    grading_prompt,
                    entry["prompt"],
                    response,
                    image,
                    letter_ids,
                )
                total_points += letter_points[pred]

            del grader_model
            del grader_processor

            gc.collect()
            torch.cuda.empty_cache()

        return 100.0 * total_points / (max_points * len(prompts))


@contextlib.contextmanager
def _deterministic_decode(enabled: bool = True):
    """Make a greedy decode reproducible across hosts, then restore the flag.

    Greedy decoding argmaxes over logits, so cuBLAS/cuDNN nondeterminism can
    flip a near-tie and diverge the whole generation. The torch flag is restored
    on exit so a later metric is not held to deterministic kernels it may not
    have (quantsim custom ops in particular).

    ``CUBLAS_WORKSPACE_CONFIG`` only takes effect if read before the CUDA
    context is created, so if CUDA is already initialized
    ``use_deterministic_algorithms(True)`` may raise on the first matmul; pass
    ``deterministic=False`` to fall back to nondeterministic kernels (aggregate
    scores stay comparable, individual responses may not).
    """
    if not enabled:
        yield
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _run_grader_subprocess(
    responses_json: str | Path,
    summary_json: str | Path,
    *,
    model_id: str,
    dtype: str,
    device_map: str | None = None,
    allow_cpu: bool = False,
    summary: bool = True,
    metric_name: str = "Grace",
) -> dict:
    """Grade ``responses_json`` in a child process; return the summary dict.

    The grader runs out-of-process because torch's caching allocator keeps the
    35B grader's segments reserved for the life of the process even after every
    tensor is freed. ONNX Runtime allocates from the driver rather than through
    torch, so it cannot reuse them and fails to rebuild its session afterwards.
    Process exit is the only reliable way to hand that memory back.

    Both paths belong to the caller: ``responses_json`` must already exist, and
    the child writes :func:`build_summary`'s report to ``summary_json``.
    """
    # The child needs the parent's GPU footprint as small as possible, or
    # device_map="auto" offloads layers to the host and grading crawls.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(
            f"Parent GPU state before grader: "
            f"{(total - free) / 1024**3:.2f} GiB used / {total / 1024**3:.2f} GiB total"
        )

    cmd = [
        sys.executable,
        "-m",
        "GenAILab.bench.grade_responses",
        str(responses_json),
        "--output-json",
        str(summary_json),
        "--model",
        model_id,
        "--dtype",
        dtype,
        "--metric-name",
        metric_name,
    ]
    if device_map:
        cmd += ["--device-map", device_map]
    if allow_cpu:
        cmd.append("--allow-cpu")
    if not summary:
        cmd.append("--no-summary")
    print(f"Running grader: {' '.join(cmd)}")
    # Inherits stdout/stderr so the grading progress bar stays visible.
    subprocess.run(cmd, check=True)
    return json.loads(Path(summary_json).read_text(encoding="utf-8"))


def format_grader_summary(summary: dict, items: list[dict]) -> str:
    """Render the grader summary dict as the human-readable report.

    ``items`` is the joined per-prompt record from
    :func:`~GenAILab.qai_hub_lm.scoring.grace.report.detail_items`;
    the prompt and the response it produced are printed for every item that lost
    points, so a regression can be read off the log alone. Perfect items are
    listed only in the score, and ``num_forced`` is recorded in the stats file
    but not printed.

    Also used by the analysis report, which passes ``MetricResult.details``
    (the same summary dict, with the joined records under ``items``).
    """
    lines = [
        "=" * 60,
        f"Grader: {summary.get('grader_model', 'unknown')}",
        f"Responses graded: {summary.get('num_items', 0)}",
        "=" * 60,
        "",
    ]
    lines.append(
        f"Overall score: {summary.get('score_pct', 0.0):.1f}%  "
        f"({summary.get('total_points', 0)}/{summary.get('max_points', 0)} pts)"
    )
    if summary.get("num_unparsed"):
        lines.append(
            f"GRADER FAILURE: {summary['num_unparsed']} item(s) produced no "
            f"readable rating and were scored 0. The score above is a floor, not "
            f"a measurement — fix the grader and re-run before trusting it."
        )
    category_scores = summary.get("category_scores") or {}
    if category_scores:
        lines.append("")
        lines.append("By category:")
        lines.extend(
            f"  {name:15s} {entry['score_pct']:5.1f}%  (n={entry['num_scored']})"
            for name, entry in sorted(
                category_scores.items(), key=lambda kv: kv[1]["score_pct"]
            )
        )
    # What the model actually said, and why it was marked down, for every
    # non-perfect grade: enough to triage a regression from the log alone.
    penalized = sorted(
        (item for item in items if item.get("points", MAX_POINTS) < MAX_POINTS),
        key=lambda item: item["points"],
    )
    if penalized:
        lines.append("")
        lines.append("Deductions:")
        for item in penalized:
            lines += [
                "",
                f"  idx={item['idx']} [{item.get('category', '?')}] "
                f"{item['points']}/{MAX_POINTS} pts",
                f"  Prompt:   {item['prompt']}",
                f"  Response: {item['output']}",
                f"  Grade:    {item['rationale'] or '(no rationale)'}",
            ]
    summary_items = summary.get("summary_items") or []
    if summary_items:
        lines.append("")
        lines.append("Summary:")
        lines.extend(
            f"  {number}. {text}" for number, text in enumerate(summary_items, start=1)
        )
    return "\n".join(lines)


@YAMLConfigParser.register_metric
class Grace(TextEvaluationMetric):
    """Grace: free-form responses graded by an LLM on a 0-10 rubric.

    Grace is "Grading Response Accuracy Evaluation". One response is generated
    per prompt in the built-in 10-categories-x-10-prompts set, then a grader LLM
    writes a one-line rationale and a ``Rating: [[N]]`` for each. A closing pass
    distils the rationales into the recurring failure modes. The reported score
    is total points as a percentage of ``MAX_POINTS`` x items; an item the
    grader failed to rate scores 0 and *stays in the denominator*, so a broken
    grader shows up as a low score rather than an inflated one.

    Scores only compare across runs if the prompt set, the rubric and the
    generation path all match, so this deliberately does not reuse
    :class:`Prompts` / :class:`Interactive`: those prepend a system prompt, leave
    thinking enabled, add special tokens on top of the chat template, and decode
    the prompt back along with the response.

    Every response and its rationale ride in the result's ``details``, so a score
    can be explained from the stats file alone.

    The name is unversioned: ``GRACE_VERSION`` rides in ``SCORING_VERSION``, so
    a bump shows up as data instead of renaming the results key.
    """

    SCORING_VERSION: int = GRACE_VERSION

    DEFAULT_GRADER_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
    DEFAULT_GRADER_DTYPE = "bfloat16"
    DEFAULT_MAX_NEW_TOKENS = 2048
    DEFAULT_SEED = 42
    DTYPES = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    @classmethod
    def get_collection_name(cls, num_samples: int = 0) -> str:
        """Cache key for the generated responses.

        ``num_samples`` is part of the key: a shortened run's responses are not
        a valid cache hit for a full one.
        """
        suffix = f"_n{num_samples}" if num_samples else ""
        return f"{cls.__name__}_generated_text{suffix}"

    @staticmethod
    def _format_prompt(tokenizer: PreTrainedTokenizer, prompt: str) -> str:
        """Apply the model's chat template to a raw user prompt.

        A bare user turn, no system prompt. Thinking is disabled to match the
        on-device Genie path and the FP baselines: a thinking model otherwise
        spends its whole token budget on a reasoning trace, so the graded
        response is the trace rather than an answer. Non-thinking templates
        ignore the unused variable.
        """
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert isinstance(formatted, str)
        return formatted

    @classmethod
    @torch.no_grad()
    def _generate_all(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        num_samples: int = 0,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: int = DEFAULT_SEED,
        deterministic: bool = True,
    ) -> list[dict]:
        """One ``{idx, category, prompt, output}`` record per prompt."""
        prompts = load_eval_prompts()
        if num_samples and num_samples > 0:
            # Records are grouped by category, so a prefix slice would leave a
            # short smoke run reporting only the first category or two.
            prompts = select_balanced(prompts, num_samples)

        set_seed(seed)
        generation_config = build_generation_config(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
        )
        model.generation_config = generation_config
        stopping_criteria = Interactive._build_stopping_criteria(model)

        items: list[dict] = []
        with _deterministic_decode(deterministic):
            for entry in tqdm(prompts, desc="Generating responses"):
                formatted = cls._format_prompt(tokenizer, entry.prompt)
                tokenized = tokenizer(
                    formatted,
                    return_tensors="pt",
                    add_special_tokens=False,
                    return_token_type_ids=False,
                )
                input_ids = tokenized["input_ids"][:, -context_length:].to(model.device)
                attention_mask = tokenized["attention_mask"][:, -context_length:].to(
                    model.device
                )
                outputs = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                )
                output_ids = (
                    outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                )
                # New tokens only: the prompt is not part of the graded response.
                new_tokens = output_ids[0][input_ids.shape[1] :]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                items.append(
                    {
                        # The prompt's own idx, not its position in a subset, so
                        # a shortened run still joins against the full set.
                        "idx": entry.idx,
                        "category": entry.category,
                        "prompt": entry.prompt,
                        "output": response.strip(),
                    }
                )
        return items

    @classmethod
    @torch.no_grad()
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        num_samples: int = 0,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        seed: int = DEFAULT_SEED,
        deterministic: bool = True,
        grader_model_id: str = DEFAULT_GRADER_MODEL_ID,
        grader_dtype: str = DEFAULT_GRADER_DTYPE,
        grader_device_map: str = "auto",
        allow_cpu: bool = False,
        summary: bool = True,
        output_dir: str | Path | None = None,
        **kwargs,
    ) -> ScoredResult:
        # Before generating, not at the point of use: a bad value should not cost
        # a full generation pass to surface.
        if grader_dtype not in cls.DTYPES:
            raise ValueError(
                f"Unsupported grader_dtype {grader_dtype!r}; "
                f"expected one of {sorted(cls.DTYPES)}."
            )

        def collect():
            return cls._generate_all(
                model,
                tokenizer,
                context_length,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                seed=seed,
                deterministic=deterministic,
            )

        if eval_ctx is not None:
            items = eval_ctx.get_or_compute_quant(
                cls.get_collection_name(num_samples), collect
            )
        else:
            items = collect()

        if not items:
            raise ValueError("Grace generated no responses to grade.")

        with contextlib.ExitStack() as stack:
            # The responses file is how the items reach the grader process, so
            # one is needed even when the caller does not want the artifacts.
            out_dir = (
                Path(output_dir)
                if output_dir is not None
                else Path(stack.enter_context(tempfile.TemporaryDirectory()))
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            responses_path = out_dir / "responses.json"
            summary_path = out_dir / "grader_summary.json"
            responses_path.write_text(
                json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Wrote {len(items)} responses to {responses_path}")

            # Evict the model under test to CPU: the grader is a 35B MoE, and
            # the two do not fit on one GPU together. The grader itself runs in
            # a child process, so that the sim rebuild on the way out of this
            # block gets the GPU back -- torch never returns the grader's
            # reserved segments to the driver within a process, and ORT cannot
            # allocate from torch's cache.
            with model.on_device(torch.device("cpu")):
                grader_summary = _run_grader_subprocess(
                    responses_path,
                    summary_path,
                    model_id=grader_model_id,
                    dtype=grader_dtype,
                    device_map=grader_device_map,
                    allow_cpu=allow_cpu,
                    summary=summary,
                    # Also the results key, so label and key cannot drift.
                    metric_name=cls.__name__,
                )

        # Responses and rationales ride along in the stats file, which is the
        # copy that leaves the machine. Without them a dropped score can only
        # be explained by re-running, and generation is the expensive half.
        graded_with_text = detail_items(items, grader_summary["items"])
        print(format_grader_summary(grader_summary, graded_with_text))
        details = {
            key: grader_summary[key]
            for key in (
                "grader_model",
                "num_items",
                "total_points",
                "max_points",
                "num_unparsed",
                "num_forced",
                "summary_items",
                "category_scores",
            )
        }
        details["items"] = graded_with_text
        return ScoredResult(result=grader_summary["score_pct"], details=details)


# ---------------------------------------------------------------------------
# ASR error rates (WER / CER)
# ---------------------------------------------------------------------------

#: Apostrophe-like codepoints folded to ASCII ``'`` before scoring. Apostrophes
#: are *kept* (LibriSpeech references contain "don't", "o'clock").
_APOSTROPHE_VARIANTS = "’‘ʼ´`"

#: Codepoints that become a space (so "well-known" scores as two words, the way
#: an ASR system that emits a space would).
_SEPARATOR_CHARS = "-‐‑‒–—―_/\\"


def normalize_transcript(text: str) -> str:
    """Normalize an ASR reference/hypothesis for error-rate scoring.

    NFKC, apostrophes folded to ASCII, lowercased, hyphens/dashes/underscores/
    slashes to space, remaining punctuation and symbols dropped (letters, digits,
    marks and whitespace survive, so non-English text is not mangled), whitespace
    collapsed.

    Part of the scoring contract: changing it requires a ``SCORING_VERSION`` bump.
    """
    text = unicodedata.normalize("NFKC", str(text))
    for variant in _APOSTROPHE_VARIANTS:
        text = text.replace(variant, "'")
    text = text.lower()
    text = "".join(" " if ch in _SEPARATOR_CHARS else ch for ch in text)
    text = "".join(
        ch
        for ch in text
        if ch == "'"
        or ch.isspace()
        or not unicodedata.category(ch).startswith(("P", "S"))
    )
    return " ".join(text.split())


def levenshtein_distance(reference: list, hypothesis: list) -> int:
    """Minimum edit distance (substitutions + insertions + deletions).

    Plain two-row dynamic programming over arbitrary sequences (word lists for
    WER, character lists for CER), unit cost per edit. O(len(ref) * len(hyp))
    time, O(min(len)) memory. Implemented here so scoring has no third-party
    dependency (no ``jiwer``) and is auditable in place.
    """
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    # hypothesis is now the shorter sequence -> the DP row is the short one.
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_token in enumerate(hypothesis, start=1):
            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + (ref_token != hyp_token),  # substitution/match
            )
        previous = current
    return previous[-1]


def word_tokens(text: str) -> list[str]:
    """Whitespace-separated tokens of the normalized text (WER units)."""
    normalized = normalize_transcript(text)
    return normalized.split() if normalized else []


def char_tokens(text: str) -> list[str]:
    """Characters of the normalized text, spaces included (CER units)."""
    return list(normalize_transcript(text))


def corpus_error_rate(
    references: list[str], hypotheses: list[str], unit: str = "word"
) -> float:
    """Corpus-level error rate in percent.

    Corpus-level (the ASR standard): edits and reference lengths are summed over
    the set and divided once, not averaged per utterance. ``unit`` selects
    ``"word"`` or ``"char"``.

    If the whole reference set normalizes to zero tokens the rate is 0.0 when the
    hypotheses are empty too, else 100.0.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Got {len(references)} references but {len(hypotheses)} hypotheses; "
            f"they must correspond 1:1."
        )
    tokenize = {"word": word_tokens, "char": char_tokens}[unit]

    total_edits = 0
    total_reference = 0
    total_hypothesis = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = tokenize(reference)
        hyp_tokens = tokenize(hypothesis)
        total_edits += levenshtein_distance(ref_tokens, hyp_tokens)
        total_reference += len(ref_tokens)
        total_hypothesis += len(hyp_tokens)

    if total_reference == 0:
        return 0.0 if total_hypothesis == 0 else 100.0
    return 100.0 * total_edits / total_reference


class _ASRErrorRateBase(EvaluationMetric):
    """Shared transcription + error-rate scoring for ASR metrics.

    Generative: each utterance is greedily decoded and compared against its
    reference with :func:`corpus_error_rate`. Transcriptions are cached under one
    shared collection name, so WER and CER together decode once. Lower is better.
    """

    SCORING_VERSION = 1

    #: Error-rate unit: "word" (WER) or "char" (CER).
    UNIT = "word"

    #: One shared cache key for all ASR error-rate metrics -- deliberately not
    #: derived from cls.__name__, so WER and CER reuse the same decode pass.
    COLLECTION_NAME = "ASRTranscriptions_generated_text"

    DEFAULT_SPLIT = "test.clean"
    #: Bound the eval set: full test.clean is 2620 utterances of greedy decode.
    DEFAULT_NUM_SAMPLES = 256
    DEFAULT_LANGUAGE = "English"
    MAX_NEW_TOKENS = 256

    @classmethod
    def get_collection_name(cls) -> str:
        """Cache key for the shared transcription pass."""
        return cls.COLLECTION_NAME

    @classmethod
    def get_dataset(
        cls,
        processor,
        context_length,
        split: str | None = None,
        num_samples: int | None = None,
        language: str | None = None,
        audio_frames: int | None = None,
        model=None,
        **kwargs,
    ):
        return LibriSpeechDataset.load_encoded_dataset(
            processor,
            context_length,
            split=split or cls.DEFAULT_SPLIT,
            num_samples=(
                cls.DEFAULT_NUM_SAMPLES if num_samples is None else num_samples
            ),
            language=cls.DEFAULT_LANGUAGE if language is None else language,
            audio_frames=audio_frames,
            model=model,
            include_reference=True,
        )

    @staticmethod
    def decode_transcript(tokenizer, token_ids) -> str:
        """Detokenize generated ids into a bare transcript.

        Special tokens are kept so ``<asr_text>`` can anchor the split: with
        automatic language detection the model emits its own header first.
        Part of the scoring contract.
        """
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        if "<asr_text>" in text:
            text = text.split("<asr_text>")[-1]
        text = re.sub(r"<\|[^|>]*\|>", " ", text)
        text = re.sub(r"</?asr_[a-z_]*>", " ", text)
        return text.strip()

    @classmethod
    @torch.no_grad()
    def transcribe_all(cls, model, processor, context_length, **kwargs) -> dict:
        """Greedily transcribe the eval set.

        Returns ``{"references": [...], "hypotheses": [...]}``.
        """
        dataset = cls.get_dataset(processor, context_length, model=model, **kwargs)
        tokenizer = getattr(processor, "tokenizer", processor)

        generation_config = build_generation_config(
            model,
            tokenizer,
            do_sample=False,
            max_new_tokens=cls.MAX_NEW_TOKENS,
        )
        model.generation_config = generation_config

        references = []
        hypotheses = []
        for sample in tqdm(dataset, desc=f"Transcribing ({cls.__name__})"):
            reference = sample.pop("reference")
            inputs = {
                key: value.to(model.device)
                for key, value in sample.items()
                if isinstance(value, torch.Tensor)
            }
            num_prompt_tokens = inputs["input_ids"].shape[-1]
            outputs = model.generate(**inputs, generation_config=generation_config)
            tokens = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            generated = tokens[0][num_prompt_tokens:]

            references.append(reference)
            hypotheses.append(cls.decode_transcript(tokenizer, generated))

            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {"references": references, "hypotheses": hypotheses}

    @classmethod
    def evaluate(
        cls,
        model: Generator,
        processor: ProcessorMixin,
        context_length: int,
        *,
        eval_ctx: EvaluationContext = None,
        **kwargs,
    ) -> float:
        if eval_ctx is None:
            warnings.warn(
                "No EvaluationContext provided; ASR transcriptions will not be "
                "cached and WER/CER will each decode the eval set."
            )

        def collect():
            return cls.transcribe_all(model, processor, context_length, **kwargs)

        data = (
            eval_ctx.get_or_compute_quant(cls.get_collection_name(), collect)
            if eval_ctx
            else collect()
        )
        return corpus_error_rate(data["references"], data["hypotheses"], unit=cls.UNIT)


@YAMLConfigParser.register_metric
class WER(_ASRErrorRateBase):
    """Word error rate (%) on LibriSpeech. Lower is better."""

    UNIT = "word"


@YAMLConfigParser.register_metric
class CER(_ASRErrorRateBase):
    """Character error rate (%) on LibriSpeech, spaces included. Lower is better."""

    UNIT = "char"


def run_metrics(
    metrics: tuple[ResolvedMetric, ...],
    generator: Generator,
    tokenizer,
    context_length: int,
    eval_ctx: EvaluationContext,
    *,
    image_size=None,
    audio_frames: int | None = None,
    gpu_meter_kwargs: dict | None = None,
    capture_intermediate_data: bool = False,
    extra_kwargs: dict | None = None,
) -> list[MetricResult]:
    """Evaluate each resolved metric against one generator/tokenizer, in order.

    The runner calls it once for the top-level metrics; an analysis pass calls
    it once per condition for its metric subset.

    ``extra_kwargs`` (e.g. a per-condition ``output_dir``) is injected only
    for metrics whose ``evaluate`` explicitly declares that parameter name --
    never relies on ``**kwargs`` absorbing it, since most metrics don't
    accept arbitrary extra kwargs and would raise ``TypeError`` if it were
    injected blindly. It overrides the same key in the metric's config
    kwargs: each analysis condition must write to its own directory.
    """
    gpu_meter_kwargs = gpu_meter_kwargs or {}
    results = []
    with torch.no_grad():
        for metric in metrics:
            metric_cls = metric.metric_cls
            with GPUMeter(
                capture_intermediate_data=False, **gpu_meter_kwargs
            ) as profiler:
                extra_metric_kwargs = {}
                if not issubclass(metric_cls, TextEvaluationMetric):
                    extra_metric_kwargs["image_size"] = image_size
                    extra_metric_kwargs["audio_frames"] = audio_frames
                if extra_kwargs:
                    declared = inspect.signature(metric_cls.evaluate).parameters
                    extra_metric_kwargs.update(
                        {k: v for k, v in extra_kwargs.items() if k in declared}
                    )
                tokenizer_arg = (
                    tokenizer.tokenizer
                    if isinstance(tokenizer, ProcessorMixin)
                    and issubclass(metric_cls, TextEvaluationMetric)
                    else tokenizer
                )
                result = metric_cls.evaluate(
                    generator,
                    tokenizer_arg,
                    context_length,
                    eval_ctx=eval_ctx,
                    **{**metric.metric_kwargs, **extra_metric_kwargs},
                )
                # Unwrap so the log line and the stats row read the same
                # whether or not the metric reported a breakdown.
                details = None
                if isinstance(result, ScoredResult):
                    result, details = result.result, result.details
                print(f"{metric_cls.__name__} result: {result}")

            results.append(
                MetricResult(
                    metric_name=metric_cls.__name__,
                    result=result,
                    profiler=profiler if capture_intermediate_data else None,
                    scoring_version=metric_cls.SCORING_VERSION,
                    details=details,
                )
            )
    return results
