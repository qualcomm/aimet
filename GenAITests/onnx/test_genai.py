# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""GenAI test runner"""

import pytest
import torch
import gc
import os
from pathlib import Path
from transformers import AutoConfig

from GenAITests.shared.helpers.profiler import ResourceProfiler, write_stats_to_disk
from GenAITests.shared.models.generator import Generator

from GenAITests.shared.helpers import datasets, metrics
from GenAITests.onnx import models
from GenAITests.onnx.helpers import quant_recipes

from GenAITests.onnx.models.utils.torch_onnx_interface import TorchONNXInterface
from GenAITests.onnx.models.utils.torch_onnx_export_utils import (
    get_model_checkpoint_path,
)


def test_llm_quantization(test_parameters):
    if test_parameters is None:
        pytest.skip("No GenAI test parameters provided.")

    print(test_parameters)
    model_kwargs = test_parameters.pop("model")
    model_cls = model_kwargs.pop("class")
    context_length = model_kwargs.pop("context_length")
    sequence_length = model_kwargs.pop("sequence_length")
    model_id = model_kwargs.pop("model_id", None)

    dataset_kwargs = test_parameters.pop("dataset")
    dataset_cls = dataset_kwargs.pop("class")

    recipe_kwargs = test_parameters.pop("recipe")
    recipe_cls = recipe_kwargs.pop("class")

    profiler_kwargs = test_parameters.pop("profiler")

    metrics = test_parameters.pop("metrics")

    gc.collect()
    torch.cuda.empty_cache()

    test_statistics = {}

    quantsim = model_cls.instantiate_quantsim(
        model_id, context_length, sequence_length, **model_kwargs
    )
    tokenizer = model_cls.instantiate_tokenizer(model_id)
    config = AutoConfig.from_pretrained(
        get_model_checkpoint_path(
            model_id if model_id is not None else model_cls.DEFAULT_MODEL_ID
        )
    )

    quantsim_with_torch_interface = TorchONNXInterface(quantsim, config)
    generator = Generator(
        quantsim_with_torch_interface, tokenizer, sequence_length, context_length
    )

    with ResourceProfiler(**profiler_kwargs) as profiler:
        train_dataset = dataset_cls.load_encoded_dataset(
            tokenizer, context_length, **dataset_kwargs
        )
        recipe_cls.apply(quantsim, generator, train_dataset, **recipe_kwargs)
    test_statistics[f"{recipe_cls.__name__}+{dataset_cls.__name__}"] = (
        profiler.as_dict()
    )

    gc.collect()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for metric_kwargs in metrics:
            metric_cls = metric_kwargs.pop("class")
            with ResourceProfiler(
                **profiler_kwargs, disable_constant_sampling=True
            ) as profiler:
                result = metric_cls.evaluate(
                    generator, tokenizer, context_length, **metric_kwargs
                )
                print(f"{metric_cls.__name__} result: {result}")
            test_statistics[f"{metric_cls.__name__}"] = {
                "result": result
            } | profiler.as_dict()

    model_kwargs["context_length"] = context_length
    model_kwargs["sequence_length"] = sequence_length
    model_kwargs["model_id"] = (
        model_id if model_id is not None else model_cls.DEFAULT_MODEL_ID
    )

    output_folder = Path(os.getcwd()) / "genai_test_artifacts"
    output_folder.mkdir(parents=True, exist_ok=True)
    write_stats_to_disk(
        str(output_folder / "profiling_data.json"),
        model_cls,
        model_kwargs,
        recipe_cls,
        recipe_kwargs,
        test_statistics,
    )
