# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
AIMET Torch Mixed-Precision Feature Runner

Applies AIMET's mixed-precision optimization to PyTorch models.
Supports two modes controlled by the `mode` config option:

1. AUTO MODE (mode: "auto" or not specified):
   Uses choose_mixed_precision() API for automatic per-layer precision selection.
   - Limited to W8A8 and W8A16 candidates (AIMET Torch limitation)
   - Uses SQNR-based evaluation for fast candidate selection

2. MANUAL MODE (mode: "manual"):
   Uses MixedPrecisionConfigurator for explicit precision control.
   - Supports int4, int8, int16, fp16
   - Set precision per module type (e.g., all Conv2d to int8, all Linear to fp16)

Configuration in model YAML:
    - name: amp_int8_fp16_fast
      feature: mixed_precision
      mode: manual  # or "auto"
      precision_config:  # For manual mode
        Conv2d: {activation: int8, param: int8}
        Linear: {activation: fp16, param: fp16}
      phase1_samples: 64
      phase2_samples: 256
      allowed_accuracy_drop: 0.01
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from qai_hub_models.datasets import BaseDataset

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
    create_calibration_dataloader,
    parse_output_names_from_qnn_options,
)

_ARTIFACTS_DIR = Path("./AIMETRegression/artifacts")
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Precision Mapping ====================

MODULE_TYPE_MAP = {
    "Conv2d": nn.Conv2d,
    "Conv1d": nn.Conv1d,
    "Conv3d": nn.Conv3d,
    "Linear": nn.Linear,
    "BatchNorm2d": nn.BatchNorm2d,
    "BatchNorm1d": nn.BatchNorm1d,
    "LayerNorm": nn.LayerNorm,
    "Embedding": nn.Embedding,
    "MultiheadAttention": nn.MultiheadAttention,
}

PRECISION_MAP = {
    "int4": "int4",
    "int8": "int8",
    "int16": "int16",
    "fp16": "fp16",
    "float16": "fp16",
}


def _normalize_precision(precision: str) -> str:
    """Normalize precision string to AIMET-compatible format."""
    p = precision.lower().strip()
    return PRECISION_MAP.get(p, p)


def _get_module_type(type_str: str) -> type:
    """Get torch.nn module type from string."""
    if type_str in MODULE_TYPE_MAP:
        return MODULE_TYPE_MAP[type_str]
    if hasattr(nn, type_str):
        return getattr(nn, type_str)
    raise ValueError(f"Unknown module type: {type_str}")


# ==================== Manual Mode ====================


def _run_manual_mode(
    sim,
    config: Dict[str, Any],
) -> None:
    """
    Apply manual mixed precision using MixedPrecisionConfigurator.

    Config example:
        precision_config:
          Conv2d: {activation: int8, param: int8}
          Linear: {activation: fp16, param: fp16}
        model_input_precision: int8
        model_output_precision: fp16
    """
    # NOTE: MixedPrecisionConfigurator is only available in aimet_torch.v2 subpackage.
    # Importing from aimet_torch.mixed_precision raises ImportError.
    # TODO: Remove '.v2' once AIMET exposes this class at the top-level API.
    from aimet_torch.v2.mixed_precision import MixedPrecisionConfigurator

    print(f"[MP-Manual] Configuring mixed precision...")

    mp_config = MixedPrecisionConfigurator(sim)

    precision_config = config.get("precision_config", {})

    if not precision_config:
        print(f"[MP-Manual] Using default config: Conv2d=int8, Linear=fp16")
        precision_config = {
            "Conv2d": {"activation": "int8", "param": "int8"},
            "Linear": {"activation": "fp16", "param": "fp16"},
        }

    for module_type_str, settings in precision_config.items():
        try:
            module_type = _get_module_type(module_type_str)

            activation = _normalize_precision(settings.get("activation", "int8"))

            param = settings.get("param", None)
            if isinstance(param, str):
                param = {"weight": _normalize_precision(param)}
            elif isinstance(param, dict):
                param = {k: _normalize_precision(v) for k, v in param.items()}

            print(
                f"[MP-Manual] Setting {module_type_str}: activation={activation}, param={param}"
            )
            mp_config.set_precision(module_type, activation=activation, param=param)

        except Exception as e:
            print(
                f"[MP-Manual] Warning: Failed to set precision for {module_type_str}: {e}"
            )

    model_input_precision = config.get("model_input_precision")
    if model_input_precision:
        precision = _normalize_precision(model_input_precision)
        print(f"[MP-Manual] Setting model input precision: {precision}")
        mp_config.set_model_input_precision(precision)

    model_output_precision = config.get("model_output_precision")
    if model_output_precision:
        precision = _normalize_precision(model_output_precision)
        print(f"[MP-Manual] Setting model output precision: {precision}")
        mp_config.set_model_output_precision(precision)

    log_file = str(_ARTIFACTS_DIR / "mmp_log.txt")
    strict = config.get("strict", False)

    print(f"[MP-Manual] Applying mixed precision configuration...")
    mp_config.apply(log_file=log_file, strict=strict)
    print(f"[MP-Manual] Configuration applied. Log: {log_file}")


# ==================== Auto Mode ====================


def _run_auto_mode(
    sim,
    qai_model: Any,
    dataset_cls: type[BaseDataset],
    config: Dict[str, Any],
    device: torch.device,
    calib_loader,
    dummy_input: torch.Tensor,
) -> None:
    """
    Run automatic mixed precision using choose_mixed_precision().

    Note: AIMET Torch's choose_mixed_precision currently only supports
    W8A8 and W8A16 candidates (not fp16).
    """
    from aimet_torch.mixed_precision import choose_mixed_precision
    from aimet_torch.common.defs import CallbackFunc, QuantizationDataType

    phase1_samples = int(config.get("phase1_samples", 64))
    phase2_samples = int(config.get("phase2_samples", 256))
    allowed_accuracy_drop = float(config.get("allowed_accuracy_drop", 0.01))

    # Check if user requested unsupported candidates
    user_candidates = config.get("candidates", "int8_int16")
    if isinstance(user_candidates, str):
        user_candidates_lower = user_candidates.lower()
        if "fp16" in user_candidates_lower or "float16" in user_candidates_lower:
            print(
                f"[MP-Auto] ⚠️  WARNING: AIMET Torch auto mode does NOT support fp16 candidates!"
            )
            print(f"[MP-Auto]    Requested: {user_candidates}")
            print(
                f"[MP-Auto]    Available: W8A8 (int8/int8) and W8A16 (int8/int16) only"
            )
            print(
                f"[MP-Auto]    💡 TIP: Use 'mode: manual' with 'precision_config' for fp16 support"
            )
            print(f"[MP-Auto]    Falling back to W8A8 + W8A16 candidates...")

    print(f"[MP-Auto] Running automatic mixed precision search...")
    print(f"[MP-Auto] Candidates: W8A8, W8A16 (AIMET Torch limitation)")
    print(f"[MP-Auto] Allowed accuracy drop: {allowed_accuracy_drop:.2%}")

    # Forward pass callback for calibration (required by choose_mixed_precision)
    def forward_pass_callback_fn(mdl: nn.Module, args):
        """Forward pass callback for AIMET calibration."""
        num_batches = args if args else len(calib_loader)
        mdl.eval()
        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if i >= num_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                inputs = inputs.to(device)
                mdl(inputs)

    # Wrap in CallbackFunc as AIMET expects
    forward_pass_callback = CallbackFunc(forward_pass_callback_fn, len(calib_loader))

    # Eval callback for phase 1 - SQNR-based (fast)
    def eval_callback_phase1_fn(mdl):
        """SQNR-based evaluation for fast candidate pruning."""
        mdl.eval()
        total_sqnr = 0.0
        count = 0

        with torch.no_grad():
            for i, batch in enumerate(calib_loader):
                if count >= phase1_samples:
                    break
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                inputs = inputs.to(device)
                outputs = mdl(inputs)
                # Simple proxy metric - use output magnitude as SQNR proxy
                total_sqnr += outputs.abs().mean().item()
                count += inputs.shape[0]

        return total_sqnr / max(count, 1)

    # Eval callback for phase 2 - accuracy-based
    def eval_callback_phase2_fn(mdl):
        """Accuracy-based evaluation for final selection."""
        mdl.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in calib_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs, labels = (
                        batch,
                        torch.zeros(batch.shape[0], dtype=torch.long),
                    )

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = mdl(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if total >= phase2_samples:
                    break

        return correct / total if total > 0 else 0.0

    # Candidates: W8A8 and W8A16 only (AIMET Torch limitation)
    candidates = [
        ((8, QuantizationDataType.int), (8, QuantizationDataType.int)),  # W8A8
        ((16, QuantizationDataType.int), (8, QuantizationDataType.int)),  # W8A16
    ]

    # Ensure dummy_input is on the correct device
    dummy_input_device = dummy_input.to(device)

    try:
        choose_mixed_precision(
            sim=sim,
            dummy_input=dummy_input_device,
            candidates=candidates,
            eval_callback_for_phase1=eval_callback_phase1_fn,
            eval_callback_for_phase2=eval_callback_phase2_fn,
            allowed_accuracy_drop=allowed_accuracy_drop,
            results_dir=str(_ARTIFACTS_DIR / "amp_results"),
            clean_start=True,
            forward_pass_callback=forward_pass_callback,
        )
        print(f"[MP-Auto] Mixed precision search complete")
    except Exception as e:
        print(f"[MP-Auto] Warning: choose_mixed_precision failed: {e}")
        import traceback

        traceback.print_exc()
        print(f"[MP-Auto] Falling back to default quantization")


# ==================== Main Entry Point ====================


def run_mixed_precision(
    *,
    model: Any,
    input_spec: Dict,
    dataset_cls: type[BaseDataset],
    calib_dataset_cls: type[BaseDataset] | None = None,
    config: Dict[str, Any],
    export_dir: Path = None,
) -> Tuple[Path, float, Dict[str, str], str]:
    """
    Apply AIMET Torch mixed-precision optimization to a PyTorch model.

    Supports two modes:
    - auto: Uses choose_mixed_precision() for automatic selection (W8A8/W8A16 only)
    - manual: Uses MixedPrecisionConfigurator for explicit per-type precision

    Args:
        model: QAI Hub model object
        input_spec: Input specification for dummy input creation
        dataset_cls: Dataset class for evaluation (e.g., ImagenetDataset)
        config: Configuration dictionary
        export_dir: Optional export directory

    Returns:
        Tuple of (qdq_path, accuracy, stats)
    """
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

    # Determine mode
    mode = config.get("mode", "auto").lower()

    # Extract quantization parameters using bitwidth_from_token
    quant_scheme = str(config.get("quant_scheme", "tf_enhanced"))
    default_param_bw = bitwidth_from_token(config.get("param_type", "int8"), 8)
    default_output_bw = bitwidth_from_token(config.get("activation_type", "int8"), 8)

    aimet_cfg_file = config.get("config_file", None)

    calib_samples = int(config.get("calib_samples", 256))
    eval_samples = int(config.get("eval_samples", 256))
    batch_size = int(config.get("batch_size", 32))
    metrics_samples = int(config.get("metrics_samples", 64))
    metrics_runs = int(config.get("metrics_runs", 1))
    metrics_warmup = int(config.get("metrics_warmup", 0))

    use_cuda = torch.cuda.is_available()

    print(f"[AIMET Torch MP] Configuration:")
    print(f"  Mode: {mode}")
    print(f"  Scheme: {quant_scheme}")
    print(f"  Base Precision: W{default_param_bw}/A{default_output_bw}")
    print(f"  CUDA: {'Yes' if use_cuda else 'No'}")

    # ============ Extract PyTorch Model ============
    print(f"[AIMET Torch MP] Extracting PyTorch model...")

    if hasattr(model, "to_torch_model"):
        torch_model = model.to_torch_model()
    else:
        torch_model = model

    torch_model.eval()

    # ============ Create Dummy Input ============
    print(f"[AIMET Torch MP] Creating dummy input...")
    device = torch.device("cuda" if use_cuda else "cpu")
    dummy_input = create_dummy_input(input_spec, device)

    # ============ Move Model to Device ============
    print(f"[AIMET Torch MP] Moving model to device...")
    torch_model = torch_model.to(device)

    for buffer in torch_model.buffers():
        buffer.data = buffer.data.to(device)

    dummy_input = dummy_input.to(device)

    # ============ Build QuantSim ============
    print(f"[AIMET Torch MP] Building QuantizationSimModel...")
    sim = build_quantsim_torch(
        model=torch_model,
        dummy_input=dummy_input,
        quant_scheme=quant_scheme,
        default_param_bw=default_param_bw,
        default_output_bw=default_output_bw,
        config_file=aimet_cfg_file,
        use_cuda=use_cuda,
        apply_prepare_model=True,
    )

    # ============ Build Calibration DataLoader ============
    print(
        f"[AIMET Torch MP] Building calibration dataloader ({calib_samples} samples)..."
    )
    calib_loader = create_calibration_dataloader(
        model, calib_dataset_cls or dataset_cls, calib_samples, batch_size
    )

    # Load dataset once — reused by calibration, eval, and metrics calls
    _dataset = load_torch_dataset(model, dataset_cls)

    # ============ Initial Calibration ============
    print(f"[AIMET Torch MP] Calibrating encodings with {calib_samples} samples...")

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

    print(f"[AIMET Torch MP] Calibration complete")

    # ============ Apply Mixed Precision ============
    if mode == "manual":
        _run_manual_mode(sim, config)
        # Recompute encodings after MP configuration
        print(f"[AIMET Torch MP] Recomputing encodings after MP configuration...")
        sim.compute_encodings(
            forward_pass_callback=calibration_callback,
            forward_pass_callback_args=calib_samples,
        )
    else:
        # Auto mode
        _run_auto_mode(
            sim, model, dataset_cls, config, device, calib_loader, dummy_input
        )

    # ============ Evaluate Final Accuracy ============
    print(f"[AIMET Torch MP] Evaluating accuracy with {eval_samples} samples...")

    feature_acc = eval_pytorch_model(
        sim.model,
        model,
        dataset_cls,
        num_samples=eval_samples,
        dataset=_dataset,
    )

    print(f"[AIMET Torch MP] Mixed-precision accuracy: {feature_acc:.4f}")

    # ============ Measure Performance ============
    print(f"[AIMET Torch MP] Measuring runtime and memory...")

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

    print(f"[AIMET Torch MP] Runtime: {runtime_str}, Memory: {memory_str}")

    # ============ Export ============
    print(f"[AIMET Torch MP] Exporting QDQ ONNX...")
    print(f"[AIMET Torch MP] Moving model and dummy_input to CPU for export...")

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

    # ============ Prepare Results ============
    if mode == "manual":
        precision_config = config.get("precision_config", {})
        config_summary = (
            ", ".join(
                [
                    f"{k}={v.get('activation', 'int8')}"
                    for k, v in precision_config.items()
                ]
            )
            if precision_config
            else "default"
        )
        technique_str = f"mp_manual({config_summary})"
    else:
        candidates_config = config.get("candidates", "int8_int16")
        technique_str = f"mp_auto({candidates_config})"

    stats = {
        "techniques": technique_str,
        "runtime": runtime_str,
        "memory": memory_str,
    }

    return qdq_path, float(feature_acc), stats
