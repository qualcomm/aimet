# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Metrics for GenAI testing"""

from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, GenerationConfig, TextStreamer

from GenAITests.shared.helpers.yaml_config_parser import YAMLConfigParser
from GenAITests.shared.models.generator import Generator
from .datasets import (
    Wikitext,
    TinyMMLU as TinyMMLUDataset,
    MMLU as MMLUDataset,
    MMMLU as MMMLUDataset,
)


class EvaluationMetric(ABC):
    """Generic GenAI evaluation metric"""

    @classmethod
    @abstractmethod
    def evaluate(
        cls, model: Generator, tokenizer: PreTrainedTokenizer, context_length: int
    ) -> float | list[str]:
        """Perform evaluation on provided model"""


@YAMLConfigParser.register_metric
class PPL(EvaluationMetric):
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


class GenericMMLU(EvaluationMetric):
    """Generic MMLU evaluation metric. Should work with any MMLU dataset."""

    @staticmethod
    @abstractmethod
    def get_dataloader(
        tokenizer: PreTrainedTokenizer, context_length: int
    ) -> DataLoader:
        """Get the dataloader associated with this MMLU evaluator."""

    @classmethod
    @torch.no_grad()
    def evaluate(
        cls,
        model: Generator,
        tokenizer: PreTrainedTokenizer,
        context_length: int,
        **kwargs,
    ) -> float:
        dataloader = cls.get_dataloader(tokenizer, context_length, **kwargs)

        def tokenize_letter(letter: str):
            return torch.Tensor(
                tokenizer(letter, add_special_tokens=False)["input_ids"]
            ).to(dtype=torch.int)

        choices = tuple(tokenize_letter(letter) for letter in ("A", "B", "C", "D"))

        correct_predictions = 0

        for batch in tqdm(
            dataloader, total=len(dataloader), desc=f"Evaluating {cls.__name__}"
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
            last_logit = torch.nn.functional.log_softmax(last_logit, dim=-1)

            scores = tuple(last_logit[choice] for choice in choices)
            index = scores.index(max(scores))
            prediction = choices[index]
            label = torch.Tensor(batch["label"]).to(dtype=torch.int)

            if prediction == label:
                correct_predictions += 1

        return float(correct_predictions / len(dataloader)) * 100


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
class Interactive(EvaluationMetric):
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
            else GenerationConfig(
                max_new_tokens=1000,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                top_k=40,
                top_p=0.95,
                temperature=0.8,
            )
        )

        print(formatted_prompt, end="")
        if highlight_output:
            print("\033[0;31m", end="")  # Start red color for output

        streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)
        outputs = model.generate(
            inputs=tokenized_user_input["input_ids"],
            attention_mask=tokenized_user_input["attention_mask"],
            generation_config=model.generation_config,
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
        cls, model: Generator, tokenizer: PreTrainedTokenizer, context_length: int
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
        cls, model: Generator, tokenizer: PreTrainedTokenizer, context_length: int
    ) -> list[str]:
        generated_text = []
        for prompt in TrickyPrompts.prompts.get(model.config.model_type, []):
            print("===============================")
            generated_text.append(
                cls.generate_output(
                    model,
                    tokenizer,
                    formatted_prompt=prompt,
                    generation_config=GenerationConfig(
                        max_new_tokens=2,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=False,
                    ),
                    highlight_output=True,
                )
            )
        print("===============================")
        return generated_text


@YAMLConfigParser.register_metric
class Prompts(Interactive):
    prompts = ["What is gravity?", "What is a llama?"]

    @classmethod
    def evaluate(
        cls, model: Generator, tokenizer: PreTrainedTokenizer, context_length: int
    ) -> list[str]:
        generated_text = []
        for prompt in Prompts.prompts:
            print("===============================")
            generated_text.append(
                cls.generate_output(model, tokenizer, unformatted_prompt=prompt)
            )
        print("===============================")
        return generated_text
