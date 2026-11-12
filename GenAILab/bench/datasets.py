# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Datasets for GenAI testing"""

from abc import ABC, abstractmethod
import ast
import gc
import inspect
import re
import warnings

from datasets import (
    Audio,
    Dataset as HFDataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, pipeline

from GenAILab.bench.determinism import set_seed
from GenAILab.bench.utils.generation_utils import build_generation_config
from GenAILab.bench.utils.prompt_utils import (
    CALIBRATION_PROMPTS_FILE,
    load_text_prompts,
    thinking_kwargs,
)
from GenAILab.bench.yaml_config_parser import YAMLConfigParser
from GenAILab.qai_hub_lm.schema import (
    InterleavedSpec,
    WikitextSpec,
    TinyMMLUSpec,
    MMLUSpec,
    MMLUProSpec,
    MMMLUSpec,
    MMMUSpec,
    C4Spec,
    AOKVQASpec,
    ERQASpec,
    Where2PlaceSpec,
    HypersimSpec,
    GeneratedDatasetSpec,
)

# LibriSpeechSpec is imported from the schema submodule rather than the schema
# package root: the package ``__init__`` re-export list is owned elsewhere and
# this keeps the audio work self-contained. Both paths are the same class.
from GenAILab.qai_hub_lm.schema.dataset import LibriSpeechSpec


class Dataset(ABC):
    """Generic GenAI Dataset class"""

    @staticmethod
    @abstractmethod
    def load_dataset(split: str):
        """Load dataset from huggingface"""

    @classmethod
    @abstractmethod
    def load_encoded_dataset(
        cls, tokenizer: PreTrainedTokenizer, context_length: int, split: str
    ):
        """Load encoded and chunked dataset"""


class TextDataset(Dataset):
    """Dataset that accepts a tokenizer (PreTrainedTokenizer)."""


class MultimodalDataset(Dataset):
    """Dataset that accepts a processor (ProcessorMixin) for text + image inputs."""


class InterleavedDatasetWrapper(torch.utils.data.Dataset):
    """Interleaves entries from multiple datasets in round-robin order.

    Items are drawn alternately: dataset_0[0], dataset_1[0], dataset_0[1],
    dataset_1[1], ... Once a dataset is exhausted it is skipped, and the
    remaining datasets keep being drawn until all are exhausted. The total
    length is therefore the sum of all sub-dataset lengths (callers that want a
    smaller calibration set bound iteration separately, e.g. via the recipe's
    ``num_iterations``).
    """

    def __init__(self, datasets: list):
        self.datasets = datasets
        # Precompute the flat draw order as (dataset_idx, item_idx) pairs:
        # round-robin across datasets, skipping any that are exhausted, until
        # every dataset has been fully consumed.
        lengths = [len(d) for d in datasets]
        counters = [0] * len(datasets)
        self._order: list[tuple[int, int]] = []
        while any(counters[i] < lengths[i] for i in range(len(datasets))):
            for i in range(len(datasets)):
                if counters[i] < lengths[i]:
                    self._order.append((i, counters[i]))
                    counters[i] += 1

    def __len__(self):
        return len(self._order)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self._order)
        if not 0 <= index < len(self._order):
            raise IndexError(index)
        ds_idx, item_idx = self._order[index]
        return self.datasets[ds_idx][item_idx]


@YAMLConfigParser.register_dataset(InterleavedSpec)
class Interleaved(MultimodalDataset):
    """Meta-dataset that interleaves entries from multiple sub-datasets.

    YAML usage::

        dataset:
          name: Interleaved
          source_datasets:
            - name: Wikitext
              split: train
            - name: AOKVQA
              split: train

    Parent kwargs like ``image_size`` and ``audio_frames`` are automatically
    forwarded to multimodal sub-datasets whose ``load_encoded_dataset``
    signature accepts them.
    """

    #: Parent kwargs forwarded to multimodal sub-datasets that accept them.
    FORWARDED_KWARGS = ("image_size", "audio_frames", "model")

    @staticmethod
    def load_dataset(split: str):
        raise NotImplementedError("Interleaved does not support load_dataset directly.")

    @classmethod
    def load_encoded_dataset(cls, tokenizer, context_length, source_datasets, **kwargs):
        if not source_datasets or not isinstance(source_datasets, list):
            raise ValueError(
                "Interleaved dataset requires a 'source_datasets' list with at least one entry."
            )
        loaded = []
        for sub_config in source_datasets:
            sub_config = sub_config.copy()
            sub_name = sub_config.pop("name")
            sub_cls = YAMLConfigParser.get_dataset(sub_name)
            # Text datasets need just the tokenizer; multimodal datasets
            # need the full processor and get the parent kwargs they accept
            # (an image dataset takes image_size, an audio one audio_frames).
            if issubclass(sub_cls, TextDataset):
                sub_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
            else:
                sub_tokenizer = tokenizer
                params = inspect.signature(sub_cls.load_encoded_dataset).parameters
                for key in cls.FORWARDED_KWARGS:
                    if key in kwargs and key in params:
                        sub_config.setdefault(key, kwargs[key])
            loaded.append(
                sub_cls.load_encoded_dataset(
                    sub_tokenizer, context_length, **sub_config
                )
            )
        return InterleavedDatasetWrapper(loaded)


class ChunkedDataset(torch.utils.data.Dataset):
    """Internal helper class to chunk input IDs to static graph context length"""

    def __init__(self, tokenized_data: dict[str, torch.Tensor], size_per_chunk: int):
        self.tokenized_data = tokenized_data
        self.size_per_chunk = size_per_chunk

    def __len__(self):
        return len(self.tokenized_data["input_ids"][0]) // self.size_per_chunk

    def __getitem__(self, index: int):
        # Raise IndexError for out-of-range indices so that bare iteration over
        # this map-style dataset (e.g. itertools.islice without a DataLoader)
        # terminates correctly. Without this, an out-of-range slice silently
        # returns an empty (1, 0) tensor and iteration never stops.
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start_index = index * self.size_per_chunk
        end_index = (index + 1) * self.size_per_chunk
        return {
            "input_ids": self.tokenized_data["input_ids"][:, start_index:end_index],
            "attention_mask": self.tokenized_data["attention_mask"][
                :, start_index:end_index
            ],
        }


@YAMLConfigParser.register_dataset(WikitextSpec)
class Wikitext(TextDataset):
    """Wikitest dataset"""

    @staticmethod
    def load_dataset(split: str):
        return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    @classmethod
    def load_encoded_dataset(
        cls, tokenizer: PreTrainedTokenizer, context_length: int, split: str
    ):
        dataset_split = cls.load_dataset(split)
        join_token = (
            "\n\n"
            if split == "test" or tokenizer.bos_token is None
            else tokenizer.bos_token
        )
        encoded_dataset_split = tokenizer(
            join_token.join(dataset_split["text"]),
            return_tensors="pt",
            add_special_tokens=True,
        )

        return ChunkedDataset(encoded_dataset_split, context_length)


@YAMLConfigParser.register_dataset(TinyMMLUSpec)
class TinyMMLU(TextDataset):
    """TinyMMLU dataset"""

    @staticmethod
    def load_dataset(split: str):
        return load_dataset("tinyBenchmarks/tinyMMLU", split=split)

    @classmethod
    def load_encoded_dataset(
        cls, tokenizer: PreTrainedTokenizer, context_length: int, split: str
    ):
        dataset_split = cls.load_dataset(split)

        def tokenize(samples):
            tokenized_question = tokenizer(
                samples["input_formatted"],
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            tokenized_question = {
                k: list(map(lambda field: field[-context_length:], v))
                for k, v in tokenized_question.items()
            }

            tokenized_answer = tokenizer(
                list(map(lambda answer: chr(ord("A") + answer), samples["answer"])),
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            result = tokenized_question
            result.update({"label": tokenized_answer["input_ids"]})

            return result

        return dataset_split.map(
            tokenize,
            batched=True,
            remove_columns=[
                "question",
                "subject",
                "choices",
                "answer",
                "input_formatted",
            ],
        )


@YAMLConfigParser.register_dataset(MMLUSpec)
class MMLU(TextDataset):
    """MMLU Dataset"""

    @classmethod
    def _format_question(cls, question: str, choices: list[str]):
        return f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

    @classmethod
    def _format_question_and_answer(
        cls, question: str, choices: list[str], answer: str
    ):
        return cls._format_question(question, choices) + f" {answer}"

    @classmethod
    def load_fewshot(cls, num_fewshot: int = 5, fewshot_split: str = "dev"):
        if num_fewshot == 0:
            return {}

        fewshot_split = load_dataset("cais/mmlu", name="all", split=fewshot_split)
        grouped_fewshot_questions = {}

        def group_fewshot_questions(sample):
            question = sample["question"]
            choices = sample["choices"]
            subject = sample["subject"]
            answer = chr(ord("A") + sample["answer"])

            if len(grouped_fewshot_questions.get(subject, [])) >= num_fewshot:
                return

            if subject not in grouped_fewshot_questions:
                grouped_fewshot_questions[subject] = []

            grouped_fewshot_questions[subject].append(
                cls._format_question_and_answer(question, choices, answer)
            )

        fewshot_split.map(group_fewshot_questions)

        for subject, questions in grouped_fewshot_questions.items():
            if len(questions) < num_fewshot:
                raise ValueError(
                    f"Not enough samples available in split {fewshot_split} to satisfy {num_fewshot} fewshot samples."
                )

        def combine_questions(subject, questions):
            formatted_subject = subject.replace("_", " ")
            formatted_string = f"The following are multiple choice questions (with answers) about {formatted_subject}.\n\n"
            for question in questions:
                formatted_string += question
                formatted_string += "\n\n"
            return formatted_string

        formatted_fewshot_questions = {
            subject: combine_questions(subject, questions)
            for subject, questions in grouped_fewshot_questions.items()
        }
        return formatted_fewshot_questions

    @staticmethod
    def load_dataset(split: str = "test"):
        if split != "test":
            raise ValueError("MMLU dataset only supports test split.")
        return load_dataset("cais/mmlu", name="all", split=split)

    @classmethod
    def load_encoded_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        split: str,
        num_fewshot: int = 5,
        fewshot_split: str = "dev",
    ):
        dataset_split = cls.load_dataset(split)
        fewshot_subject_headers = cls.load_fewshot(num_fewshot, fewshot_split)

        def tokenize(sample):
            question = sample["question"]
            choices = sample["choices"]
            subject = sample["subject"]

            formatted_question = list(
                map(
                    lambda question, choices: cls._format_question(question, choices),
                    question,
                    choices,
                )
            )
            fewshot_formatted_question = (
                list(
                    map(
                        lambda subject, question: str(
                            fewshot_subject_headers[subject] + question
                        ),
                        subject,
                        formatted_question,
                    )
                )
                if num_fewshot > 0
                else formatted_question
            )

            tokenized_question = tokenizer(
                fewshot_formatted_question,
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            tokenized_question = {
                k: list(map(lambda field: field[-context_length:], v))
                for k, v in tokenized_question.items()
            }

            tokenized_answer = tokenizer(
                list(map(lambda answer: chr(ord("A") + answer), sample["answer"])),
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            result = tokenized_question
            result.update({"label": tokenized_answer["input_ids"]})

            return result

        return dataset_split.map(
            tokenize,
            batched=True,
            remove_columns=[
                "question",
                "subject",
                "choices",
                "answer",
            ],
        )


@YAMLConfigParser.register_dataset(MMMLUSpec)
class MMMLU(TextDataset):
    """MMLU Dataset"""

    @classmethod
    def _format_question(cls, question: str, choices: tuple[str]):
        return f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

    @classmethod
    def _format_question_and_answer(
        cls, question: str, choices: list[str], answer: str
    ):
        return cls._format_question(question, choices) + f" {answer}"

    @classmethod
    def load_fewshot(cls, dataset_split, num_fewshot: int = 5):
        if num_fewshot == 0:
            return {}

        grouped_fewshot_questions: dict[str, list[str]] = {}

        def group_fewshot_questions(sample: dict[str, str]):
            question = sample["Question"]
            choices = (sample["A"], sample["B"], sample["C"], sample["D"])
            subject = sample["Subject"]
            answer = sample["Answer"]

            # We need one extra question to make sure that we can create an appropriately formatted string even if one
            # of the fewshot questions is encountered.
            if len(grouped_fewshot_questions.get(subject, [])) >= num_fewshot + 1:
                return

            if subject not in grouped_fewshot_questions:
                grouped_fewshot_questions[subject] = []

            grouped_fewshot_questions[subject].append(
                cls._format_question_and_answer(question, choices, answer)
            )

        dataset_split.map(group_fewshot_questions)

        for subject, questions in grouped_fewshot_questions.items():
            if len(questions) < num_fewshot:
                raise ValueError(
                    f"Not enough samples available in split to satisfy {num_fewshot} fewshot samples."
                )

        return grouped_fewshot_questions

    @staticmethod
    def load_dataset(split: str = "default"):
        return load_dataset("openai/MMMLU", name=split, split="test")

    @classmethod
    def load_encoded_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        split: str,
        num_fewshot: int = 5,
    ):
        dataset_split = cls.load_dataset(split)
        grouped_fewshot_questions = cls.load_fewshot(dataset_split, num_fewshot)

        def tokenize(sample: dict[str, list[str]]):
            question = sample["Question"]
            A = sample["A"]
            B = sample["B"]
            C = sample["C"]
            D = sample["D"]
            subject = sample["Subject"]

            formatted_question = list(
                map(
                    lambda question, A, B, C, D: cls._format_question(
                        question, (A, B, C, D)
                    ),
                    question,
                    A,
                    B,
                    C,
                    D,
                )
            )

            def assemble_fewshot_question(formatted_question: str, subject: str):
                subject_fewshot_questions = grouped_fewshot_questions[subject]

                formatted_string = ""
                num_fewshot_questions_added = 0
                for fewshot_question in subject_fewshot_questions:
                    if num_fewshot_questions_added >= num_fewshot:
                        break
                    if formatted_question in fewshot_question:
                        continue

                    formatted_string += fewshot_question
                    formatted_string += "\n\n"
                    num_fewshot_questions_added += 1

                formatted_string += formatted_question
                return formatted_string

            fewshot_formatted_question = list(
                map(assemble_fewshot_question, formatted_question, subject)
            )

            tokenized_question = tokenizer(
                fewshot_formatted_question,
                return_token_type_ids=False,
                add_special_tokens=True,
            )

            tokenized_question = {
                k: list(map(lambda field: field[-context_length:], v))
                for k, v in tokenized_question.items()
            }

            tokenized_answer = tokenizer(
                sample["Answer"],
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            result = tokenized_question
            result.update({"label": tokenized_answer["input_ids"]})

            return result

        return dataset_split.map(
            tokenize,
            batched=True,
            remove_columns=[
                "Question",
                "A",
                "B",
                "C",
                "D",
                "Answer",
                "Subject",
                "Unnamed: 0",
            ],
        )


@YAMLConfigParser.register_dataset(MMLUProSpec)
class MMLUPro(TextDataset):
    """MMLU Pro Dataset with 10 answer choices (A-J)."""

    @classmethod
    def _format_question(cls, question: str, choices: list[str]):
        """Format question with 10 options in MMLU Pro style (without 'Answer:')."""
        formatted = f"Question:\n{question.strip()}\nOptions:\n"
        for i, choice in enumerate(choices):
            formatted += f"{chr(ord('A') + i)}. {choice}"
            if i < len(choices) - 1:
                formatted += "\n"
        return formatted

    @classmethod
    def _format_question_and_answer(
        cls, question: str, choices: list[str], answer: str, cot_content: str = None
    ):
        """Format question with answer and optional CoT reasoning for few-shot examples."""
        formatted = f"Question:\n{question.strip()}\nOptions:\n"
        for i, choice in enumerate(choices):
            formatted += f"{chr(ord('A') + i)}. {choice}\n"

        # Add the answer with CoT if available
        formatted += "Answer:"
        if cot_content:
            # Use CoT content if available, formatting it properly
            cot = cot_content.replace(
                "A: Let's think step by step.", " Let's think step by step."
            )
            formatted += cot
        else:
            formatted += f" {answer}"
        return formatted

    @classmethod
    def load_fewshot(cls, num_fewshot: int = 5, fewshot_split: str = "validation"):
        """Load few-shot examples grouped by category with CoT reasoning.

        Returns a dict mapping category -> formatted few-shot header string.
        """
        if num_fewshot == 0:
            return {}

        fewshot_data = load_dataset("TIGER-Lab/MMLU-Pro", split=fewshot_split)
        grouped_fewshot_questions = {}

        for sample in fewshot_data:
            category = sample["category"]

            if len(grouped_fewshot_questions.get(category, [])) >= num_fewshot:
                continue

            if category not in grouped_fewshot_questions:
                grouped_fewshot_questions[category] = []

            question = sample["question"]
            choices = sample["options"]
            answer_idx = sample["answer_index"]
            answer = chr(ord("A") + answer_idx)
            cot_content = sample.get("cot_content", "")

            grouped_fewshot_questions[category].append(
                cls._format_question_and_answer(question, choices, answer, cot_content)
            )

        # Verify we have enough examples per category
        for category, questions in grouped_fewshot_questions.items():
            if len(questions) < num_fewshot:
                warnings.warn(
                    f"Only {len(questions)} examples available for category '{category}', "
                    f"requested {num_fewshot}. Using all available."
                )

        def combine_questions(category, questions):
            formatted_string = f"The following are multiple choice questions (with answers) about {category}.\n\n"
            for question in questions:
                formatted_string += question
                formatted_string += "\n\n"
            return formatted_string

        formatted_fewshot_questions = {
            category: combine_questions(category, questions)
            for category, questions in grouped_fewshot_questions.items()
        }
        return formatted_fewshot_questions

    @staticmethod
    def load_dataset(split: str = "test"):
        """Load MMLU Pro dataset from HuggingFace."""
        if split not in ["test", "validation"]:
            raise ValueError(
                "MMLU Pro dataset only supports test and validation splits."
            )
        return load_dataset("TIGER-Lab/MMLU-Pro", split=split)

    @classmethod
    def load_encoded_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        split: str,
        num_fewshot: int = 5,
        fewshot_split: str = "validation",
    ):
        """Load dataset with fully tokenized inputs for generative evaluation.

        Returns a HuggingFace dataset where each sample contains:
        - 'input_ids': Tokenized prompt including system prompt, question, and "Answer:"
        - 'attention_mask': Attention mask for the input
        - 'label': The correct answer as a single-token tensor
        """
        SYSTEM_PROMPT = (
            "The following are multiple choice questions (with answers). "
            'Think step by step and then finish your answer with "the answer is (X)" '
            "where X is the correct letter choice."
        )

        dataset_split = cls.load_dataset(split)
        fewshot_headers = (
            cls.load_fewshot(num_fewshot, fewshot_split) if num_fewshot > 0 else {}
        )

        def tokenize(sample):
            question = sample["question"]
            choices = sample["options"]
            category = sample["category"]

            # Build user prompt with optional few-shot examples
            user_prompt = list(
                map(
                    lambda cat, q, c: (
                        (fewshot_headers.get(cat, "") if num_fewshot > 0 else "")
                        + f"Question:\n{q.strip()}\nOptions:\n"
                        + "\n".join(
                            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(c)
                        )
                    ),
                    category,
                    question,
                    choices,
                )
            )

            # Format as chat with system prompt and "Answer:" in assistant turn
            formatted_prompts = []
            for prompt in user_prompt:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "Answer:"},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    continue_final_message=True,
                )
                formatted_prompts.append(text)

            # Tokenize the formatted prompts
            tokenized_question = tokenizer(
                formatted_prompts,
                return_token_type_ids=False,
                add_special_tokens=False,  # Chat template already added them
            )

            # Truncate to context_length
            tokenized_question = {
                k: list(map(lambda field: field[-context_length:], v))
                for k, v in tokenized_question.items()
            }

            # Tokenize answer letters
            tokenized_answer = tokenizer(
                list(
                    map(
                        lambda answer_idx: chr(ord("A") + answer_idx),
                        sample["answer_index"],
                    )
                ),
                return_token_type_ids=False,
                add_special_tokens=False,
                return_tensors="pt",
            )

            result = tokenized_question
            result.update({"label": tokenized_answer["input_ids"]})

            return result

        return dataset_split.map(
            tokenize,
            batched=True,
            remove_columns=[
                "question",
                "options",
                "answer",
                "answer_index",
                "category",
                "src",
                "question_id",
                "cot_content",
            ],
        )


class LazyMMMUDataset(torch.utils.data.Dataset):
    """Lazy MMMU dataset, to avoid ballooning memory costs once samples are processed."""

    def __init__(self, raw_dataset, processor, context_length, image_size=None):
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.context_length = context_length
        self.image_size = tuple(image_size) if image_size is not None else None

    def __len__(self):
        return len(self.raw_dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        sample = self.raw_dataset[index]

        question = sample["question"]
        options = (
            ast.literal_eval(sample["options"])
            if isinstance(sample["options"], str)
            else sample["options"]
        )

        # Collect all images by slot (1-indexed)
        image_slots = {i: sample.get(f"image_{i}") for i in range(1, 8)}
        valid_images = [img for img in image_slots.values() if img is not None]

        # Format answer choices
        choices_text = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )

        # Build content blocks, splitting on <image N> placeholders.
        # Each reference (even repeated ones like <image 1> appearing twice)
        # becomes an image content block, and the corresponding image is
        # appended to ordered_images so the processor sees a 1:1 mapping
        # between placeholders and pixel data.
        content = []
        ordered_images = []
        parts = re.split(r"(<image \d+>)", question)
        for part in parts:
            m = re.match(r"<image (\d+)>", part)
            if m:
                img_idx = int(m.group(1))
                img = image_slots.get(img_idx)
                if img is not None:
                    content.append({"type": "image"})
                    ordered_images.append(img)
                else:
                    # Reference to a non-existent image — keep as text
                    content.append({"type": "text", "text": part})
            elif part.strip():
                content.append({"type": "text", "text": part})

        # If no placeholders were found but images exist, prepend them
        if not ordered_images and valid_images:
            for img in valid_images:
                content.insert(0, {"type": "image"})
            ordered_images = list(valid_images)

        content.append({"type": "text", "text": f"\n{choices_text}"})

        # "Answer:" in an assistant turn so continue_final_message continues
        # the model's response, not the user's text (bump MMMU.SCORING_VERSION if changed).
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "Answer:"},
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            **thinking_kwargs(self.processor.chat_template),
        )
        if self.image_size is not None:
            ordered_images = [image.resize(self.image_size) for image in ordered_images]

        # Use the full processor with images so that image placeholder tokens
        # are expanded to match the actual image dimensions.
        inputs = self.processor(
            text=[text],
            images=ordered_images if ordered_images else None,
            return_tensors="pt",
        )

        # Truncate text tokens to context_length
        inputs["input_ids"] = inputs["input_ids"][:, -self.context_length :]
        inputs["attention_mask"] = inputs["attention_mask"][:, -self.context_length :]

        # Store answer and number of choices for the metric
        inputs["label"] = sample["answer"]
        inputs["num_options"] = len(options)

        # Remove token type IDS if they exist - recomputed on the fly
        inputs.pop("mm_token_type_ids", None)

        return inputs


@YAMLConfigParser.register_dataset(MMMUSpec)
class MMMU(MultimodalDataset):
    """MMMU Dataset."""

    @staticmethod
    def load_dataset(split: str = "validation"):
        # Necessary because all the subjects are separate datasets
        # Some questions are free-response, but we only want to evaluate on multiple-choice
        configs = get_dataset_config_names("MMMU/MMMU")
        all_ds = [load_dataset("MMMU/MMMU", config, split=split) for config in configs]
        combined = concatenate_datasets(all_ds)
        return combined.filter(lambda x: x["question_type"] == "multiple-choice")

    @classmethod
    def load_encoded_dataset(
        cls, processor, context_length, split="validation", image_size=None
    ):
        raw_dataset = cls.load_dataset(split)
        return LazyMMMUDataset(
            raw_dataset, processor, context_length, image_size=image_size
        )


@YAMLConfigParser.register_dataset(C4Spec)
class C4(TextDataset):
    """C4 dataset"""

    @staticmethod
    def load_dataset(split: str = "en", num_samples: int = 2048):
        stream = load_dataset("allenai/c4", name=split, split="train", streaming=True)
        return HFDataset.from_list(list(stream.take(num_samples)))

    @classmethod
    def load_encoded_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        split: str = "en",
        num_samples: int = 2048,
    ):
        dataset_split = cls.load_dataset(split, num_samples)
        join_token = "\n\n" if tokenizer.bos_token is None else tokenizer.bos_token
        encoded_dataset_split = tokenizer(
            join_token.join(dataset_split["text"]),
            return_tensors="pt",
            add_special_tokens=True,
        )

        return ChunkedDataset(encoded_dataset_split, context_length)


class LazyAOKVQADataset(torch.utils.data.Dataset):
    """Lazy A-OKVQA dataset for multimodal backbone calibration.

    Each sample is an image + question processed through the full VLM processor,
    producing fused input_ids, attention_mask, pixel_values, and image_grid_thw.

    For the train split, the correct answer choice and the first rationale are
    appended as an assistant turn, giving the quantizer longer, more
    representative sequences to calibrate on.
    """

    def __init__(
        self,
        raw_dataset,
        processor,
        context_length,
        image_size=None,
        include_answer=False,
    ):
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.context_length = context_length
        self.image_size = tuple(image_size) if image_size is not None else None
        self.include_answer = include_answer

    def __len__(self):
        return len(self.raw_dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        sample = self.raw_dataset[index]

        image = sample["image"]
        question = sample["question"]
        choices = sample["choices"]

        choices_text = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(choices)
        )

        content = [
            {"type": "image"},
            {"type": "text", "text": f"{question}\n{choices_text}\nAnswer:"},
        ]

        messages = [{"role": "user", "content": content}]

        if self.include_answer:
            answer_idx = sample["correct_choice_idx"]
            answer_letter = chr(65 + answer_idx)
            answer_text = f"{answer_letter}. {choices[answer_idx]}"
            rationales = sample.get("rationales", [])
            if rationales:
                answer_text += f"\n\nReasoning: {rationales[0]}"
            messages.append({"role": "assistant", "content": answer_text})

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not self.include_answer,
        )

        if self.image_size is not None:
            image = image.resize(self.image_size)

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        inputs["input_ids"] = inputs["input_ids"][:, -self.context_length :]
        inputs["attention_mask"] = inputs["attention_mask"][:, -self.context_length :]
        inputs.pop("mm_token_type_ids", None)

        return inputs


@YAMLConfigParser.register_dataset(AOKVQASpec)
class AOKVQA(MultimodalDataset):
    """A-OKVQA dataset for multimodal backbone calibration (CC BY 4.0)."""

    @staticmethod
    def load_dataset(split: str = "train"):
        return load_dataset("HuggingFaceM4/A-OKVQA", split=split)

    @classmethod
    def load_encoded_dataset(
        cls,
        processor,
        context_length,
        split="train",
        image_size=None,
    ):
        raw_dataset = cls.load_dataset(split)
        return LazyAOKVQADataset(
            raw_dataset,
            processor,
            context_length,
            image_size=image_size,
            include_answer=(split == "train"),
        )


class LazyHypersimDataset(torch.utils.data.Dataset):
    """Lazy Hypersim dataset for multimodal visual-encoder calibration.

    Hypersim has no question/answer fields (unlike AOKVQA) -- it's a plain
    image corpus, domain-matched to embodied/indoor-scene VLMs rather than
    AOKVQA's general web-photo distribution. Calibration only needs
    representative forward passes, so every example uses one fixed generic
    instruction paired with the image; there's nothing to continue, so this
    is always a single user turn (no assistant-turn variant like AOKVQA's
    ``include_answer``).
    """

    _PROMPT = "Describe this image in detail."

    def __init__(self, raw_dataset, processor, context_length, image_size=None):
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.context_length = context_length
        self.image_size = tuple(image_size) if image_size is not None else None

    def __len__(self):
        return len(self.raw_dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        sample = self.raw_dataset[index]
        image = sample["image"]

        content = [
            {"type": "image"},
            {"type": "text", "text": self._PROMPT},
        ]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if self.image_size is not None:
            image = image.resize(self.image_size)

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        inputs["input_ids"] = inputs["input_ids"][:, -self.context_length :]
        inputs["attention_mask"] = inputs["attention_mask"][:, -self.context_length :]
        inputs.pop("mm_token_type_ids", None)

        return inputs


@YAMLConfigParser.register_dataset(HypersimSpec)
class Hypersim(MultimodalDataset):
    """Hypersim dataset for multimodal visual-encoder calibration.

    The original Hypersim RGB image release (apple/ml-hypersim) is licensed
    CC BY-SA 3.0 -- permissive, no non-commercial restriction; the Evermotion
    3D mesh assets referenced in that repo carry separate commercial terms
    but are not part of the public image release used here. This loads the
    ``Vigen1/hypersim-mini`` HF Hub mirror (plain RGB ``image`` column,
    ``mask``/``scene``/``camera``/``frame`` ignored for calibration); that
    mirror is tagged ``license: other`` on the Hub and disclaims responsibility
    for compliance with the original terms -- the CC BY-SA 3.0 source terms
    above are what actually governs the images.
    """

    @staticmethod
    def load_dataset(split: str = "train"):
        return load_dataset("Vigen1/hypersim-mini", split=split)

    @classmethod
    def load_encoded_dataset(
        cls,
        processor,
        context_length,
        split="train",
        image_size=None,
    ):
        raw_dataset = cls.load_dataset(split)
        return LazyHypersimDataset(
            raw_dataset,
            processor,
            context_length,
            image_size=image_size,
        )


class LazyERQADataset(torch.utils.data.Dataset):
    """Lazy ERQA (Embodied Reasoning QA) dataset.

    Each example carries a variable-length ``images`` list. Unlike MMMU/AOKVQA,
    ERQA's ``question`` text has no ``<image N>`` placeholder tokens at all --
    images are referenced in prose ("the first image", "the second image") in
    list order, and its ``visual_indices`` field is degenerate (``[0] * len
    (images)``, confirmed against live data -- not a real index into a larger
    pool). So images are simply presented in list order ahead of the question
    text, which already has the A-D choices and an answer-format instruction
    embedded inline.

    Each image gets a short "Image N:" text label ahead of it. This isn't just
    cosmetic: with zero separating tokens between consecutive images, the
    fused sequence's visual-token positions merge into one contiguous run,
    and the ONNX generator's slicing (``Qwen3VL_Generator._find_image_boundaries``
    /``slice_inputs_for_inference``) can only split *between* runs, not within
    one -- so an undivided multi-image run breaks its per-image chunking/
    padding to the traced deepstack shape (confirmed: this is what caused a
    real "Got: 98 Expected: 256" deepstack shape mismatch on ONNX). MMMU avoids
    this incidentally, since its questions already interleave real prose
    between ``<image N>`` references.
    """

    def __init__(self, raw_dataset, processor, context_length, image_size=None):
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.context_length = context_length
        self.image_size = tuple(image_size) if image_size is not None else None

    def __len__(self):
        return len(self.raw_dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        sample = self.raw_dataset[index]

        question = sample["question"]
        ordered_images = list(sample["images"])

        content = []
        for i in range(len(ordered_images)):
            content.append({"type": "text", "text": f"Image {i + 1}:"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": question})

        # "Answer:" in an assistant turn so continue_final_message continues
        # the model's response (bump ERQA.SCORING_VERSION if changed).
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "Answer:"},
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            **thinking_kwargs(self.processor.chat_template),
        )
        if self.image_size is not None:
            ordered_images = [image.resize(self.image_size) for image in ordered_images]

        inputs = self.processor(
            text=[text],
            images=ordered_images if ordered_images else None,
            return_tensors="pt",
        )

        inputs["input_ids"] = inputs["input_ids"][:, -self.context_length :]
        inputs["attention_mask"] = inputs["attention_mask"][:, -self.context_length :]

        # ERQA is always a 4-way A-D multiple choice; store the answer letter
        # and choice count for the metric.
        inputs["label"] = sample["answer"]
        inputs["num_options"] = 4

        inputs.pop("mm_token_type_ids", None)

        return inputs


@YAMLConfigParser.register_dataset(ERQASpec)
class ERQA(MultimodalDataset):
    """ERQA (Embodied Reasoning QA) -- multi-image, 4-way MCQ, test-only."""

    @staticmethod
    def load_dataset(split: str = "test"):
        return load_dataset("FlagEval/ERQA", split=split)

    @classmethod
    def load_encoded_dataset(
        cls, processor, context_length, split="test", image_size=None
    ):
        raw_dataset = cls.load_dataset(split)
        return LazyERQADataset(
            raw_dataset, processor, context_length, image_size=image_size
        )


class LazyWhere2PlaceDataset(torch.utils.data.Dataset):
    """Lazy Where2Place dataset: single-image pointing task.

    Unlike every other dataset here, Where2Place is not a ``datasets``-loadable
    parquet/imagefolder dataset -- its real content (point_questions.jsonl +
    images/ + masks/) has to be read off a local snapshot of the raw HF repo
    (see ``Where2Place.load_dataset``). Each example is a free-form-generation
    prompt (no MCQ letter, no continuation trick), so ``add_generation_prompt``
    is used instead of MMMU/AOKVQA's ``continue_final_message``/"Answer:" turn.

    Each question's own ``text`` already instructs the model to answer with a
    *list* of points (``[(x1, y1), (x2, y2), ...]``, normalized 0-1) -- this is
    RoboPoint's own convention (confirmed against live data) and is used
    verbatim, not overridden with a single-point instruction.
    """

    def __init__(self, examples, processor, context_length, image_size=None):
        self.examples = examples
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.context_length = context_length
        self.image_size = tuple(image_size) if image_size is not None else None

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        example = self.examples[index]

        from PIL import Image

        image = Image.open(example["image_path"]).convert("RGB")
        mask = np.array(Image.open(example["mask_path"]).convert("L")) > 0
        # Predicted points are normalized [0,1] regardless of any model-input
        # resize below, so de-normalization must use the mask's own pixel
        # dimensions, not the (possibly resized) model-input image.
        mask_height, mask_width = mask.shape

        if self.image_size is not None:
            image = image.resize(self.image_size)

        content = [
            {"type": "image"},
            {"type": "text", "text": example["question"]},
        ]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        inputs["input_ids"] = inputs["input_ids"][:, -self.context_length :]
        inputs["attention_mask"] = inputs["attention_mask"][:, -self.context_length :]
        inputs.pop("mm_token_type_ids", None)

        # Non-tensor metadata for the metric -- mask stays a plain numpy array
        # (not a tensor) so it's skipped by the tensor-only device-move filter
        # metrics use when moving a sample onto the model's device.
        inputs["mask"] = mask
        inputs["mask_size"] = (mask_width, mask_height)

        return inputs


@YAMLConfigParser.register_dataset(Where2PlaceSpec)
class Where2Place(MultimodalDataset):
    """Where2Place -- single-image free-space pointing task (train-only, no
    official test split; 100 rows -- confirmed against a live load, not the
    HF viewer's auto-converted parquet stat, which misreports the row count).
    Sourced from the raw HF repo (not a ``datasets``-loadable dataset -- its
    auto-converted parquet preview drops the mask/question files entirely;
    the real content is only available via the raw files).
    """

    @staticmethod
    def load_dataset(split: str = "train"):
        from huggingface_hub import snapshot_download
        import json
        import os

        snapshot_dir = snapshot_download(
            repo_id="wentao-yuan/where2place", repo_type="dataset"
        )
        questions_path = os.path.join(snapshot_dir, "point_questions.jsonl")
        examples = []
        with open(questions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # point_questions.jsonl has no "mask" key -- masks live under
                # masks/<same-stem>.png regardless of the source image's own
                # extension (confirmed against the live repo layout).
                image_stem = os.path.splitext(record["image"])[0]
                examples.append(
                    {
                        "image_path": os.path.join(
                            snapshot_dir, "images", record["image"]
                        ),
                        "mask_path": os.path.join(
                            snapshot_dir, "masks", f"{image_stem}.png"
                        ),
                        "question": record["text"],
                    }
                )
        return examples

    @classmethod
    def load_encoded_dataset(
        cls, processor, context_length, split="train", image_size=None
    ):
        examples = cls.load_dataset(split)
        return LazyWhere2PlaceDataset(
            examples, processor, context_length, image_size=image_size
        )


# ---------------------------------------------------------------------------
# Audio / ASR
# ---------------------------------------------------------------------------

#: Target sampling rate for every audio dataset here. Qwen3-ASR's feature
#: extractor is fixed at 16 kHz; LibriSpeech is natively 16 kHz.
AUDIO_SAMPLING_RATE = 16000


def decode_audio_column(audio, target_sampling_rate: int = AUDIO_SAMPLING_RATE):
    """Return a mono float32 waveform for one HuggingFace ``Audio`` column value.

    Handles every shape ``datasets`` returns: ``{"array", "sampling_rate"}``,
    undecoded ``{"bytes", "path"}`` (via soundfile, falling back to torchcodec),
    a torchcodec ``AudioDecoder``, or a bare array.

    Multi-channel is averaged to mono. The sampling rate is checked, not
    resampled -- a mismatch is a pipeline bug; cast the column instead.
    """
    sampling_rate = None
    array = None

    if isinstance(audio, dict):
        if audio.get("array") is not None:
            array = audio["array"]
            sampling_rate = audio.get("sampling_rate", target_sampling_rate)
        else:
            array, sampling_rate = _decode_audio_bytes(
                audio.get("bytes"), audio.get("path")
            )
    elif hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        array = samples.data
        sampling_rate = int(samples.sample_rate)
    elif audio is not None:
        array = audio
        sampling_rate = target_sampling_rate

    if array is None:
        raise ValueError(f"Could not extract a waveform from audio value {audio!r}")

    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:
        # (channels, samples) -> mono
        array = array.mean(axis=0 if array.shape[0] < array.shape[-1] else -1)
    array = array.reshape(-1)

    if int(sampling_rate) != int(target_sampling_rate):
        raise ValueError(
            f"Audio sampling rate {sampling_rate} != required "
            f"{target_sampling_rate}; cast the column with "
            f"Audio(sampling_rate={target_sampling_rate}) before use."
        )
    return array, int(target_sampling_rate)


def _decode_audio_bytes(raw_bytes, path):
    """Decode encoded audio (e.g. FLAC) to ``(array, sampling_rate)``."""
    if raw_bytes is None and path is None:
        raise ValueError("Undecoded audio has neither 'bytes' nor 'path'.")

    try:
        import soundfile
    except ImportError:
        soundfile = None

    if soundfile is not None:
        import io

        source = io.BytesIO(raw_bytes) if raw_bytes is not None else path
        array, sampling_rate = soundfile.read(source, dtype="float32", always_2d=False)
        return array, int(sampling_rate)

    try:
        from torchcodec.decoders import AudioDecoder
    except ImportError as exc:
        raise ImportError(
            "Decoding audio requires an audio backend; install 'soundfile' "
            "(preferred) or 'torchcodec'."
        ) from exc

    decoder = AudioDecoder(raw_bytes if raw_bytes is not None else path)
    samples = decoder.get_all_samples()
    return samples.data, int(samples.sample_rate)


class LazyLibriSpeechDataset(torch.utils.data.Dataset):
    """Lazy LibriSpeech dataset for ASR calibration and WER/CER evaluation.

    Each item is one utterance through the processor, giving the fused
    ``input_ids`` / ``attention_mask`` / ``input_features`` / ``input_features_mask``.

    Qwen3-ASR offers ``apply_transcription_request`` (forced-language prefill,
    ``n_window`` chunking); general processors like Gemma4's have only
    ``apply_chat_template``. ``_build_transcription_inputs`` picks whichever
    exists.

    Where ``n_window`` applies, the mel axis must be a multiple of ``2 *
    n_window``. The feature extractor pads it; this class asserts the invariant
    so a mismatch fails here rather than inside the encoder.

    ``include_reference`` attaches the reference transcript. Evaluation needs it;
    calibration wants pure tensors, so it defaults off.
    """

    #: Mel-frame padding multiple = ``2 * n_window``.
    DEFAULT_N_WINDOW = 50

    def __init__(
        self,
        raw_dataset,
        processor,
        context_length,
        *,
        language: str | None = "English",
        n_window: int | None = None,
        audio_frames: int | None = None,
        model=None,
        include_reference: bool = False,
        sampling_rate: int = AUDIO_SAMPLING_RATE,
    ):
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.context_length = context_length
        self.language = language
        self.include_reference = include_reference
        self.sampling_rate = sampling_rate
        if n_window is None:
            feature_extractor = getattr(processor, "feature_extractor", None)
            n_window = getattr(feature_extractor, "n_window", self.DEFAULT_N_WINDOW)
        self.n_window = int(n_window)
        # Set only for a statically-shaped (ONNX) export: pads every sample's
        # audio input to this fixed mel-frame count, via model.pad_audio_item
        # -- see Generator.pad_audio_item for why this is a model hook rather
        # than logic in this (model-family-generic) dataset class.
        self.audio_frames = audio_frames
        self.model = model
        self._warned_over_context = False

    def __len__(self):
        return len(self.raw_dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def mel_frame_multiple(self) -> int:
        """Mel frames must be an integer multiple of this."""
        return 2 * self.n_window

    def _fit_waveform(self, waveform):
        """Pad or truncate the raw waveform to fit ``audio_frames``, for a
        statically-shaped (ONNX) export. A no-op when ``audio_frames`` is
        unset (e.g. the torch backend).

        Delegated to ``Generator.fit_audio_waveform``: the fit depends on the
        processor's own framing, so it belongs with the model, not here.
        """
        if self.audio_frames is None:
            return waveform
        return self.model.fit_audio_waveform(waveform, self.audio_frames)

    def _build_transcription_inputs(self, waveform):
        """Run the processor generically across audio-chat model families.

        Only ``apply_chat_template`` with an audio content block is universal.
        ``apply_transcription_request`` (Qwen3-ASR) adds a forced-language prefill
        and ``n_window``, so it is used only where it exists; other processors get
        a plain text instruction instead, having no forced-language convention.
        """
        if hasattr(self.processor, "apply_transcription_request"):
            # apply_transcription_request builds the ASR chat template (optional
            # system prompt + audio user turn + assistant turn prefilled with
            # "language <NAME><asr_text>" so the model emits only the transcript)
            # and runs the processor, returning the fused inputs.
            return self.processor.apply_transcription_request(
                audio=[waveform],
                language=self.language,
                # Nested: a flat n_window is not in the typed kwargs schema, so
                # transformers drops it and silently uses its own default.
                processor_kwargs={
                    "audio_kwargs": {
                        "sampling_rate": self.sampling_rate,
                        "n_window": self.n_window,
                    }
                },
            )

        content = [
            {"type": "audio", "audio": waveform},
            {"type": "text", "text": "Transcribe the audio."},
        ]
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            return_dict=True,
            # Without this the model sees a completed user turn and emits <eos>
            # at once, so every hypothesis is empty and WER reads a flat 100.
            add_generation_prompt=True,
            processor_kwargs={"audio_kwargs": {"sampling_rate": self.sampling_rate}},
            return_tensors="pt",
        )

    def _pad_audio_inputs(self, item):
        """Pad to a fixed ``audio_frames`` count for a static (ONNX) export; a
        no-op when unset, as on the torch backend.

        Delegated to ``Generator.pad_audio_item``: each family's exported encoder
        has its own fixed-shape contract, and this class stays generic.
        """
        if self.audio_frames is None:
            return item
        return self.model.pad_audio_item(item, self.audio_frames)

    def __getitem__(self, index):
        sample = self.raw_dataset[index]

        waveform, _ = decode_audio_column(sample["audio"], self.sampling_rate)
        waveform = self._fit_waveform(waveform)

        inputs = self._build_transcription_inputs(waveform)

        item = {
            key: inputs[key]
            for key in (
                "input_ids",
                "attention_mask",
                "input_features",
                "input_features_mask",
            )
        }

        # Chunked-attention encoders (Qwen3-ASR) require the mel axis to tile
        # evenly into 2*n_window blocks; processors without n_window have no
        # such encoder and skip the check.
        if hasattr(self.processor, "apply_transcription_request"):
            num_frames = item["input_features"].shape[-1]
            if num_frames % self.mel_frame_multiple:
                raise ValueError(
                    f"Mel frames ({num_frames}) is not a multiple of "
                    f"2 * n_window ({self.mel_frame_multiple}); the audio encoder "
                    f"cannot chunk it. Check that n_window matches the model config."
                )

        item = self._pad_audio_inputs(item)

        # Audio prompts must not be truncated: input_ids carry exactly as many
        # audio placeholder tokens as the encoder will emit, so trimming them
        # would desynchronise text and audio. Warn instead of silently cutting.
        num_tokens = item["input_ids"].shape[-1]
        if (
            self.context_length is not None
            and num_tokens > self.context_length
            and not self._warned_over_context
        ):
            self._warned_over_context = True
            warnings.warn(
                f"LibriSpeech sample produced {num_tokens} tokens, exceeding the "
                f"context length {self.context_length}. Audio prompts are not "
                f"truncated (that would break audio-token alignment); use a "
                f"larger context length or shorter utterances."
            )

        if self.include_reference:
            item["reference"] = sample["text"]

        return item


@YAMLConfigParser.register_dataset(LibriSpeechSpec)
class LibriSpeech(MultimodalDataset):
    """LibriSpeech ASR corpus (CC BY 4.0), read-speech English audio.

    Pinned to an exact revision and the ``all`` config, so split names are the
    familiar ``validation.clean`` / ``test.clean`` and results stay reproducible.

    YAML usage::

        dataset:
          name: LibriSpeech
          split: validation.clean
          num_samples: 128
    """

    REPO_ID = "openslr/librispeech_asr"
    #: Pinned so a repo-side reconversion cannot silently change scores.
    REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
    #: The 'all' config exposes '<split>.<clean|other>' split names.
    CONFIG = "all"
    CALIBRATION_SPLIT = "validation.clean"
    EVAL_SPLIT = "test.clean"

    @staticmethod
    def load_dataset(split: str = "test.clean", num_samples: int | None = None):
        # data_files scopes the download to this split's parquet shard(s) only --
        # passing `config=` instead makes `datasets` download_and_prepare() every
        # split of the 'all' config (~62 GB) on first use, regardless of `split`.
        # Shards live at all/<split>/<shard>.parquet at REVISION; a glob that
        # matches nothing raises FileNotFoundError here rather than loading empty.
        raw_dataset = load_dataset(
            LibriSpeech.REPO_ID,
            data_files={split: f"{LibriSpeech.CONFIG}/{split}/*.parquet"},
            split=split,
            revision=LibriSpeech.REVISION,
            # data_files only supplies this one split's shard(s), but the repo's
            # pinned dataset_infos.json still declares every split of the 'all'
            # config -- skip the expected-vs-recorded split check or `datasets`
            # raises ExpectedMoreSplitsError for the splits we deliberately didn't load.
            verification_mode="no_checks",
        )
        # no_checks skips every verification, not just the split count, so check
        # the columns used downstream here rather than fail far from the load.
        missing = {"audio", "text"} - set(raw_dataset.column_names)
        if missing:
            raise ValueError(
                f"{LibriSpeech.REPO_ID}@{LibriSpeech.REVISION} split {split!r} is "
                f"missing column(s) {sorted(missing)}; got {raw_dataset.column_names}."
            )
        # decode=False keeps the encoded FLAC bytes: decoding happens in
        # decode_audio_column, which does not require datasets' torchcodec
        # backend to be installed.
        raw_dataset = raw_dataset.cast_column("audio", Audio(decode=False))
        if num_samples is not None:
            raw_dataset = raw_dataset.select(range(min(num_samples, len(raw_dataset))))
        return raw_dataset

    @classmethod
    def load_encoded_dataset(
        cls,
        processor,
        context_length,
        split: str = "test.clean",
        num_samples: int | None = None,
        language: str | None = "English",
        n_window: int | None = None,
        audio_frames: int | None = None,
        model=None,
        include_reference: bool = False,
    ):
        # include_reference defaults to False so calibration items are pure
        # tensors; the WER/CER metric asks for the reference explicitly.
        raw_dataset = cls.load_dataset(split, num_samples)
        return LazyLibriSpeechDataset(
            raw_dataset,
            processor,
            context_length,
            language=language,
            n_window=n_window,
            audio_frames=audio_frames,
            model=model,
            include_reference=include_reference,
        )


@YAMLConfigParser.register_dataset(GeneratedDatasetSpec)
class GeneratedDataset(TextDataset):
    """Calibrate on text the float model generates from seed prompts. Generation
    runs through the standard HuggingFace ``AutoModelForCausalLM`` / ``generate``
    API in bfloat16, independent of the quantsim being calibrated.

    YAML usage::

        dataset:
          name: GeneratedDataset
          num_inputs: 5
          max_new_tokens: 512
          seed: 42

    ``model_id`` selects the float model to generate from. It defaults to the
    run's top-level ``model:`` (injected by the recipe chain) so calibration
    matches the quantized model, and need not appear in the dataset config; set
    it explicitly only to generate from a different model.
    """

    @staticmethod
    def load_dataset(split: str):
        raise NotImplementedError(
            "GeneratedDataset does not load from disk; it generates text from "
            "the float model. Use load_encoded_dataset."
        )

    @classmethod
    def load_encoded_dataset(
        cls,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        *,
        model_id: str,
        num_inputs: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 40,
        seed: int = 42,
    ):
        prompts = load_text_prompts(prompts_file=CALIBRATION_PROMPTS_FILE)
        target_tokens = num_inputs * context_length

        encoded = cls._generate_encoded(
            tokenizer=tokenizer,
            model_id=model_id,
            prompts=prompts,
            target_tokens=target_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )

        return cls._build_chunked_dataset(encoded, context_length)

    @staticmethod
    def _build_chunked_dataset(encoded, context_length):
        num_tokens = encoded["input_ids"].shape[-1]
        chunk_size = min(context_length, num_tokens)
        return ChunkedDataset(encoded, chunk_size)

    @classmethod
    def _generate_encoded(
        cls,
        *,
        tokenizer,
        model_id,
        prompts,
        target_tokens,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        seed,
    ) -> dict[str, torch.Tensor]:
        """Load the float model and generate until ``target_tokens`` is reached.

        Seed prompts are cycled in order; generation continues prompt-by-prompt
        until the concatenated, re-tokenized text has at least ``target_tokens``
        tokens, then it is truncated to exactly ``target_tokens``.
        """
        set_seed(seed)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        gen_config = build_generation_config(
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )
        # The pipeline applies the chat template and generation prompt, runs
        # generate() under no-grad, and returns the prompt plus the generated
        # text (return_full_text=True), matching the original prompt+completion
        # calibration samples. Device placement comes from the model's
        # device_map, so no explicit device is passed.
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        join_token = "\n\n" if tokenizer.bos_token is None else tokenizer.bos_token

        generated_texts = []
        total_tokens = 0
        prompt_idx = 0
        # Cycle through the prompt list until we have enough tokens. A hard cap
        # (2x the prompts needed in the ideal case) guards against pathological
        # early-EOS loops that never reach target.
        max_generations = 2 * (target_tokens // max(max_new_tokens, 1) + len(prompts))
        progress = tqdm(
            total=target_tokens,
            unit="tok",
            desc=f"Generating calibration text ({model_id})",
        )
        try:
            while (
                total_tokens < target_tokens and len(generated_texts) < max_generations
            ):
                prompt = prompts[prompt_idx % len(prompts)]
                prompt_idx += 1
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ]
                # With chat input, return_full_text=True yields the full
                # conversation as a list of message dicts; render it back to a
                # single prompt+completion string via the chat template.
                full_messages = generator(
                    messages,
                    generation_config=gen_config,
                    return_full_text=True,
                )[0]["generated_text"]
                text = tokenizer.apply_chat_template(full_messages, tokenize=False)
                generated_texts.append(text)
                # Approximate running total; the authoritative count comes
                # from the final joined re-tokenization below.
                total_tokens = len(
                    tokenizer(
                        join_token.join(generated_texts),
                        add_special_tokens=True,
                    )["input_ids"]
                )
                # Clamp the bar to its total so a final overshoot doesn't
                # render as >100%.
                progress.n = min(total_tokens, target_tokens)
                progress.refresh()
        finally:
            progress.close()
            # Free the float weights before sim calibration runs.
            del generator
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        encoded = tokenizer(
            join_token.join(generated_texts),
            return_tensors="pt",
            add_special_tokens=True,
        )
        # Truncate to exactly target_tokens so the dataset is a whole number of
        # context-length chunks (when target_tokens is a multiple of CL).
        if encoded["input_ids"].shape[-1] > target_tokens:
            encoded = {
                "input_ids": encoded["input_ids"][:, :target_tokens],
                "attention_mask": encoded["attention_mask"][:, :target_tokens],
            }
        return encoded
