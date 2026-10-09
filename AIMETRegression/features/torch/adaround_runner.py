# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring


"""
AdaRound Feature Runner for AIMET Torch

This module implements AIMET's Adaptive Rounding (AdaRound) post-training
quantization optimization technique for PyTorch models.

Algorithm Overview:
AdaRound optimizes the rounding of weights during quantization to minimize
reconstruction error. Traditional quantization uses "round-to-nearest" which
may not be optimal. AdaRound learns whether each weight should be rounded
up or down to minimize the layer-wise reconstruction error.

Process:
1. Prepare model for AIMET
2. Create DataLoader for AdaRound optimization
3. Apply AdaRound to learn optimal weight rounding
4. Build QuantSim with optimized model
5. Load saved encodings and calibrate remaining
6. Evaluate and export
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from qai_hub_models.datasets import BaseDataset

from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.model_preparer import prepare_model

from AIMETRegression.evaluation.calibration import (
    get_calibration_dataloader,
    run_torch_calibration,
)
from AIMETRegression.evaluation.eval_torch import eval_pytorch_model, load_torch_dataset
from AIMETRegression.evaluation.metrics_utils import measure_inference_metrics
from AIMETRegression.features.torch._common import (
    bitwidth_from_token,
    build_quantsim_torch,
    export_torch_qdq,
    create_dummy_input,
    map_quant_scheme,
    create_calibration_dataloader,
    parse_output_names_from_qnn_options,
)

_ARTIFACTS_DIR = Path("./AIMETRegression/artifacts")
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def run_adaround(
    *,
    model: Any,
    input_spec: Dict,
    dataset_cls: type[BaseDataset],
    calib_dataset_cls: type[BaseDataset] | None = None,
    config: Dict[str, Any],
    export_dir: Optional[Path] = None,
) -> Tuple[Path, float, Dict[str, str], str]:
    """
    Apply AdaRound optimization for improved weight quantization on PyTorch models.

    AdaRound optimizes the rounding of weights during quantization by learning
    whether each weight should be rounded up or down to minimize reconstruction
    error. This typically improves accuracy compared to round-to-nearest.

    Args:
        model: QAI Hub model object
        input_spec: Input specification for dummy input creation
        dataset_cls: Dataset class for evaluation (e.g., ImagenetDataset)
        config: Configuration dictionary containing:
            Required:
                - model_name: Name for output files
            Quantization:
                - quant_scheme: Quantization scheme (default: "min_max")
                - param_type: Weight precision (default: "int8")
                - activation_type: Activation precision (default: "int8")
                - config_file: Optional AIMET config file
            AdaRound specific:
                - adaround_samples: Samples for optimization (default: 64)
                - adaround_iters: Optimization iterations (default: 15000)
            Evaluation:
                - calib_samples: Calibration samples (default: 256)
                - eval_samples: Evaluation samples (default: 256)
                - metrics_samples: Samples for performance measurement (default: 64)
                - metrics_runs: Number of timing runs (default: 1)
                - metrics_warmup: Warmup runs (default: 0)
        export_dir: Optional export directory

    Returns:
        Tuple containing:
            - exported_onnx_path: Path to optimized ONNX model
            - feature_accuracy: Accuracy after AdaRound optimization
            - stats: Dict with "techniques", "runtime", and "memory"

    Raises:
        Exception: If AdaRound optimization fails
    """
    # ============ Extract Configuration ============
    model_name = config["model_name"]

    if export_dir is None:
        export_dir = config.get("_export_dir")
        if export_dir:
            export_dir = Path(export_dir)
        else:
            export_dir = Path("./AIMETRegression/artifacts") / model_name
            export_dir.mkdir(parents=True, exist_ok=True)
    else:
        export_dir = Path(export_dir)

    # Quantization parameters
    quant_scheme = str(config.get("quant_scheme", "min_max"))
    param_type = str(config.get("param_type", "int8"))
    activation_type = str(config.get("activation_type", "int8"))
    aimet_cfg_file = config.get("config_file", None)

    # Parse bitwidths
    default_param_bw = bitwidth_from_token(param_type, 8)
    default_output_bw = bitwidth_from_token(activation_type, 8)

    # Evaluation parameters
    calib_samples = int(config.get("calib_samples", 256))
    eval_samples = int(config.get("eval_samples", 256))

    # AdaRound specific parameters
    adaround_samples = int(config.get("adaround_samples", min(64, calib_samples)))
    adaround_iters = int(config.get("adaround_iters", 15000))

    # Performance measurement
    metrics_samples = int(config.get("metrics_samples", 64))
    metrics_runs = int(config.get("metrics_runs", 1))
    metrics_warmup = int(config.get("metrics_warmup", 0))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"[AdaRound Torch] Configuration:")
    print(f"  Quantization: W{default_param_bw}/A{default_output_bw}")
    print(f"  Scheme: {quant_scheme}")
    print(f"  AdaRound iterations: {adaround_iters}")
    print(f"  AdaRound samples: {adaround_samples}")
    print(f"  CUDA acceleration: {'Enabled' if use_cuda else 'Disabled'}")

    # ============ Step 1: Extract and Prepare Model ============
    print(f"[AdaRound Torch] Extracting PyTorch model...")

    if hasattr(model, "to_torch_model"):
        torch_model = model.to_torch_model()
    else:
        torch_model = model

    torch_model = torch_model.to(device).eval()

    # Ensure all buffers are on the correct device
    for buffer in torch_model.buffers():
        buffer.data = buffer.data.to(device)

    print(f"[AdaRound Torch] Creating dummy input...")
    dummy_input = create_dummy_input(input_spec, device)

    apply_prepare = config.get("apply_prepare_model", False)
    if apply_prepare:
        print(f"[AdaRound Torch] Preparing model for AIMET...")
        torch_model = prepare_model(torch_model).to(device)
        torch_model.eval()

    # Fold BN before AdaRound so both AdaRound's internal QuantSim and our
    # post-AdaRound QuantSim see the same folded module names.
    print(f"[AdaRound Torch] Folding BatchNorm layers...")
    fold_all_batch_norms(torch_model, input_shapes=(tuple(dummy_input.shape),))

    prepared_model = torch_model

    # ============ Step 2: Create DataLoader for AdaRound ============
    print(f"[AdaRound Torch] Creating dataloader with {adaround_samples} samples...")

    adaround_dataloader = create_calibration_dataloader(
        model,
        calib_dataset_cls or dataset_cls,
        num_samples=adaround_samples,
        batch_size=1,
    )

    num_batches = len(adaround_dataloader)
    print(f"[AdaRound Torch] Created dataloader with {num_batches} batches")

    # ============ Step 3: Create AdaRound Parameters ============
    print(f"[AdaRound Torch] Configuring AdaRound parameters...")

    # Custom forward function that handles both (inputs, labels) and just inputs
    def adaround_forward_fn(model, batch):
        if isinstance(batch, (list, tuple)):
            # Labeled batch: (inputs, labels) - use only inputs
            inputs = batch[0]
        else:
            # Unlabeled batch: just inputs
            inputs = batch

        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        return model(inputs)

    adaround_params = AdaroundParameters(
        data_loader=adaround_dataloader,
        num_batches=num_batches,
        default_num_iterations=adaround_iters,
        default_reg_param=0.01,
        default_beta_range=(20, 2),
        default_warm_start=0.2,
        forward_fn=adaround_forward_fn,  # Custom forward function for compatibility
    )

    # Resolve quant scheme
    scheme_enum = map_quant_scheme(quant_scheme)

    # ============ Step 4: Apply AdaRound Optimization ============
    print(f"[AdaRound Torch] Starting optimization ({adaround_iters} iterations)...")
    print(f"[AdaRound Torch] This may take 5-10 minutes depending on model size...")

    # Create output directory for encodings
    adaround_output_dir = export_dir / "adaround_output"
    adaround_output_dir.mkdir(parents=True, exist_ok=True)

    # Apply AdaRound - this returns the model with optimized weights
    # and saves encodings to the specified path
    adarounded_model = Adaround.apply_adaround(
        model=prepared_model,
        dummy_input=dummy_input,
        params=adaround_params,
        path=str(adaround_output_dir),
        filename_prefix=model_name,
        default_param_bw=default_param_bw,
        param_bw_override_list=None,
        ignore_quant_ops_list=None,
        default_quant_scheme=scheme_enum,
        default_config_file=str(aimet_cfg_file),
    )

    print(f"[AdaRound Torch] Optimization complete")

    # Check for saved encodings file
    encodings_file = adaround_output_dir / f"{model_name}.encodings"
    if not encodings_file.exists():
        # Try alternative extension
        encodings_file = adaround_output_dir / f"{model_name}.encodings.json"

    if encodings_file.exists():
        print(f"[AdaRound Torch] Encodings saved to: {encodings_file}")
    else:
        print(f"[AdaRound Torch] Warning: Encodings file not found at expected path")

    # ============ Step 5: Build QuantSim with AdaRounded Model ============
    print(f"[AdaRound Torch] Building QuantSim with optimized model...")

    # Build QuantSim using the adarounded model
    # BN already folded before AdaRound — skip here to keep module names
    # consistent with the saved encodings.
    sim = build_quantsim_torch(
        model=adarounded_model,
        dummy_input=dummy_input,
        quant_scheme=quant_scheme,
        default_param_bw=default_param_bw,
        default_output_bw=default_output_bw,
        config_file=str(aimet_cfg_file),
        apply_prepare_model=False,
        apply_bn_fold=False,
        use_cuda=use_cuda,
    )

    # ============ Step 6: Load Saved Encodings and Calibrate ============
    print(f"[AdaRound Torch] Loading saved encodings and calibrating...")

    if not encodings_file.exists():
        raise FileNotFoundError(f"AdaRound encodings not found: {encodings_file}")

    sim.set_and_freeze_param_encodings(str(encodings_file))
    print(f"[AdaRound Torch] Loaded parameter encodings from: {encodings_file}")

    # Load dataset once — reused by calibration, eval, and metrics calls
    _dataset = load_torch_dataset(model, dataset_cls)

    # Compute activation encodings (and param encodings if not loaded)
    calib_loader_for_encodings = get_calibration_dataloader(
        model, calib_dataset_cls or dataset_cls, calib_samples
    )

    def calibration_callback(model_to_calibrate: torch.nn.Module, args):
        """Forward pass callback for encoding calibration."""
        model_to_calibrate.eval()
        run_torch_calibration(model_to_calibrate, calib_loader_for_encodings)

    sim.model.eval()
    sim.compute_encodings(
        forward_pass_callback=calibration_callback,
        forward_pass_callback_args=calib_samples,
    )

    print(f"[AdaRound Torch] Calibration complete")

    # ============ Step 7: Evaluate Optimized Model ============
    print(f"[AdaRound Torch] Evaluating accuracy with {eval_samples} samples...")

    feature_acc = eval_pytorch_model(
        sim.model,
        model,
        dataset_cls,
        num_samples=eval_samples,
        dataset=_dataset,
    )

    print(f"[AdaRound Torch] Optimized accuracy: {feature_acc:.4f}")

    # ============ Step 8: Measure Performance ============
    print(f"[AdaRound Torch] Measuring inference performance...")

    def eval_for_metrics():
        sim.model.eval()
        with torch.no_grad():
            return eval_pytorch_model(
                sim.model,
                model,
                dataset_cls,
                num_samples=metrics_samples,
                dataset=_dataset,
            )

    runtime_str, memory_str = measure_inference_metrics(
        eval_for_metrics,
        runs=metrics_runs,
        warmup=metrics_warmup,
    )

    print(f"[AdaRound Torch] Runtime: {runtime_str}, Memory: {memory_str}")

    # ============ Step 9: Export and Bundle ============
    print(f"[AdaRound Torch] Exporting optimized model...")
    print(f"[AdaRound Torch] Moving model and dummy_input to CPU for export...")

    sim.model.cpu()
    dummy_input_cpu = dummy_input.cpu()

    output_names = parse_output_names_from_qnn_options(config.get("qnn_options", ""))
    qdq_path = export_torch_qdq(
        sim=sim,
        dummy_input=dummy_input_cpu,
        export_dir=export_dir,
        model_name=model_name,
        input_spec=input_spec,
        output_names=output_names,
    )

    # ============ Step 10: Prepare Results ============
    stats = {
        "techniques": f"quantsim(W{default_param_bw}A{default_output_bw}, {quant_scheme}) + adaround({adaround_iters})",
        "runtime": runtime_str,
        "memory": memory_str,
    }

    return qdq_path, float(feature_acc), stats
