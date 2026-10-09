# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
QAI Hub Models Loader

This module provides utilities for loading models from the Qualcomm AI Hub Models
repository along with their associated datasets and input specifications.

Key Functionality:
- Dynamic model loading from qai_hub_models package
- Automatic dataset resolution from model metadata
- Input specification extraction for pipeline compatibility
- Graceful handling of various model attribute formats
- CI/CD compatibility (auto-accepts git clones for external repos)

Design Philosophy:
QAI Hub Models have varying structures and metadata formats. This module
provides a unified interface that handles these variations, ensuring
consistent access to models regardless of their specific implementation.
"""

import os
import importlib
from typing import Tuple, Any, Optional, Union, List

from qai_hub_models.datasets import BaseDataset, DatasetSplit, instantiate_dataset
from qai_hub_models.utils.evaluate.helpers import get_deterministic_sample
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.asset_loaders import always_answer_prompts


# ==================== Utility Functions ====================


def _pick_model_cls(module) -> type:
    """
    Find and return the appropriate model class from a module.

    QAI Hub Models modules may contain the model class under various names.
    This function searches for the correct class using multiple strategies:
    1. Look for common names (Model, BaseModel)
    2. Search all module attributes for BaseModel subclasses

    Args:
        module: Python module containing the model class

    Returns:
        Model class (subclass of BaseModel)

    Raises:
        RuntimeError: If no suitable model class is found

    Search Strategy:
        1. Check for 'Model' attribute
        2. Check for 'BaseModel' attribute (if not the base class itself)
        3. Scan all module attributes for BaseModel subclasses
    """
    # Strategy 1: Try common explicit names
    for class_name in ("Model", "BaseModel"):
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            # Ensure it's a BaseModel subclass (not BaseModel itself)
            if (
                isinstance(cls, type)
                and issubclass(cls, BaseModel)
                and cls is not BaseModel
            ):
                return cls

    # Strategy 2: Search all attributes for BaseModel subclasses
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        try:
            # Check if it's a class and subclass of BaseModel
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseModel)
                and obj is not BaseModel
            ):
                return obj
        except TypeError:
            # issubclass raises TypeError if obj is not a class
            pass

    # No suitable class found
    raise RuntimeError(
        f"No BaseModel subclass found in module {module.__name__}. "
        f"Ensure the model module exports a class inheriting from BaseModel."
    )


def resolve_eval_dataset_cls(model: BaseModel) -> type[BaseDataset]:
    """
    Resolve the labeled dataset class used to measure model accuracy.

    Uses the first entry of get_eval_dataset_classes() (QAIHM v0.55+). The
    calibration dataset is deliberately not considered: it may be unlabeled
    (e.g. OpenImagesV7Dataset) and does not support the VAL split.

    Raises:
        RuntimeError: If the model declares no evaluation dataset class.
    """
    eval_classes = model.get_eval_dataset_classes()
    if eval_classes:
        return eval_classes[0]

    raise RuntimeError(
        f"Unable to resolve an eval dataset class from model {type(model).__name__}. "
        f"Model must declare 'get_eval_dataset_classes'."
    )


def resolve_calibration_dataset_cls(model: BaseModel) -> type[BaseDataset]:
    """
    Resolve the dataset class used to calibrate quantization encodings.

    Uses get_calibration_dataset_cls() when the model declares one, otherwise
    falls back to the eval dataset class.
    """
    calibration_cls = model.get_calibration_dataset_cls()
    if calibration_cls is not None:
        return calibration_cls
    return resolve_eval_dataset_cls(model)


# ==================== Main API ====================


def load_model_data(model_name: str) -> Tuple[BaseModel, Any, dict, Any]:
    """
    Load a model from QAI Hub Models with its associated dataset and specifications.

    This is the main entry point for loading models in the AIMET regression pipeline.
    It handles the complete setup process:
    1. Dynamically import the model module
    2. Find and instantiate the model class
    3. Load pretrained weights
    4. Set up the dataset and dataloader
    5. Extract input specifications

    Args:
        model_name: Name of the model in qai_hub_models
                   (e.g., "resnet50", "mobilenet_v2", "yolov5s")

    Returns:
        Tuple containing:
            - model: Instantiated and pretrained model object
            - dataset: Dataset object for evaluation
            - input_spec: Input specification dictionary
            - dataloader: Dataloader with deterministic sampling

    Raises:
        ModuleNotFoundError: If model module doesn't exist
        RuntimeError: If model class cannot be found or dataset cannot be resolved

    Example:
        >>> model, dataset, input_spec, dataloader = load_model_data("resnet50")
        >>> print(f"Model loaded: {type(model).__name__}")
        >>> print(f"Dataset: {dataset.name}")
        >>> print(f"Input shape: {input_spec['image'].shape}")

    Technical Notes:
        - Models are loaded with pretrained weights via from_pretrained()
        - Dataset split is always VALIDATION for evaluation consistency
        - Dataloader uses deterministic sampling for reproducibility
        - Default to 100 samples for initial testing (can be overridden)
        - Git clone prompts are automatically accepted in CI/CD environments
    """
    # Wrap the entire model loading process to auto-accept git clone prompts
    # This is critical for CI/CD where interactive prompts would block execution
    with always_answer_prompts(True):
        # ============ Step 1: Import Model Module ============
        try:
            module = importlib.import_module(f"qai_hub_models.models.{model_name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Model '{model_name}' not found in qai_hub_models. "
                f"Ensure the model name is correct and qai_hub_models is installed. "
                f"Available models can be found at: https://github.com/qualcomm/ai-hub-models"
            ) from e

        # ============ Step 2: Find Model Class ============
        model_cls = _pick_model_cls(module)

        # ============ Step 3: Instantiate Model ============
        # Load with pretrained weights (standard for evaluation)
        model: BaseModel = model_cls.from_pretrained()

        # ============ Step 4: Resolve Dataset ============
        dataset_cls = resolve_eval_dataset_cls(model)

        # ============ Step 5: Get Input Specification ============
        # Input spec defines the expected input format (shape, dtype, etc.).
        # Resolved before dataset construction so the dataset sizes images
        # to the model instance's actual shape rather than a class-level default.
        input_spec = model.get_input_spec()

        # Load the validation split of the dataset
        # We use validation split for all evaluation to avoid training data leakage
        dataset = instantiate_dataset(
            dataset_cls,
            DatasetSplit.VAL,  # Always use validation split for evaluation
            input_spec,
        )

        # ============ Step 6: Create Dataloader ============
        # Deterministic sampling ensures reproducible results
        dataloader = get_deterministic_sample(
            dataset,
            num_samples=100,  # Default sample size for initial testing
            samples_per_job=100,  # Process all samples in one batch
        )

    return model, dataset, input_spec, dataloader
