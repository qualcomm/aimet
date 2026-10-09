# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
Calibration data utilities.

QAI Hub Models separate the dataset used for calibration
(``get_calibration_dataset_cls``) from the datasets used for evaluation
(``get_eval_dataset_classes``). The calibration dataset may be unlabeled and may
only support a subset of splits (e.g. OpenImagesV7Dataset supports only TRAIN),
so calibration must never go through the accuracy evaluation helpers.
"""

from __future__ import annotations

from typing import Any, Optional

import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from qai_hub_models.datasets import BaseDataset, DatasetSplit, instantiate_dataset
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.onnx.helpers import extract_io_types_from_onnx_model
from qai_hub_models.utils.onnx.torch_wrapper import OnnxSessionTorchWrapper

__all__ = [
    "instantiate_calibration_dataset",
    "get_calibration_dataloader",
    "run_onnx_calibration",
    "run_torch_calibration",
]


def _candidate_splits(model: Any, calib_cls: type[BaseDataset]) -> list[DatasetSplit]:
    # A dataset that doubles as an eval dataset (e.g. ImageNet) must keep using
    # VAL: its TRAIN split is a huge, unnecessary download.
    eval_classes = list(model.get_eval_dataset_classes() or [])
    if calib_cls in eval_classes:
        return [DatasetSplit.VAL]
    return [DatasetSplit.TRAIN, DatasetSplit.VAL]


def instantiate_calibration_dataset(
    model: Any,
    calib_cls: type[BaseDataset],
    input_spec: Optional[InputSpec] = None,
) -> BaseDataset:
    """Instantiate a calibration dataset using the first split its class supports."""
    if input_spec is None:
        input_spec = model.get_input_spec()

    errors = []
    for split in _candidate_splits(model, calib_cls):
        try:
            return instantiate_dataset(calib_cls, split, input_spec)
        except ValueError as e:
            errors.append(f"{split.name}: {e}")
    raise RuntimeError(
        f"{calib_cls.__name__} supports none of the candidate calibration splits. "
        + " | ".join(errors)
    )


def get_calibration_dataloader(
    model: Any,
    calib_cls: type[BaseDataset],
    num_samples: int,
    samples_per_batch: Optional[int] = 16,
) -> DataLoader:
    """Deterministic calibration DataLoader, with num_samples clamped to dataset size."""
    dataset = instantiate_calibration_dataset(model, calib_cls)
    num_samples = max(1, min(num_samples, len(dataset)))
    return dataset.get_dataloader(num_samples, samples_per_batch)


def _per_sample_inputs(model_inputs: Any):
    """Yield one batch-of-1 input (tensor, or tuple of tensors) per sample."""
    if isinstance(model_inputs, torch.Tensor):
        for i in range(model_inputs.shape[0]):
            yield model_inputs[i : i + 1]
    else:
        for per_sample in zip(*[t for t in model_inputs]):
            yield tuple(t.unsqueeze(0) for t in per_sample)


def run_onnx_calibration(
    sess: ort.InferenceSession,
    model: Any,
    calib_cls: type[BaseDataset],
    num_samples: int,
) -> None:
    """Forward calibration samples through an ORT session. Computes no accuracy."""
    dataloader = get_calibration_dataloader(model, calib_cls, num_samples)
    inputs, outputs = extract_io_types_from_onnx_model(sess)
    wrapper = OnnxSessionTorchWrapper(sess, inputs, outputs)

    print(f"Calibrating on {calib_cls.__name__} ({num_samples} samples requested).")
    for batch in dataloader:
        model_inputs = batch[0]
        for sample in _per_sample_inputs(model_inputs):
            if isinstance(sample, tuple):
                wrapper(*sample)
            else:
                wrapper(sample)


def run_torch_calibration(
    model_to_calibrate: torch.nn.Module,
    dataloader: DataLoader,
) -> None:
    """Forward batches from a calibration dataloader through a torch module."""
    try:
        device = next(model_to_calibrate.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0]
            if isinstance(inputs, (list, tuple)):
                model_to_calibrate(*[x.to(device) for x in inputs])
            else:
                model_to_calibrate(inputs.to(device))
