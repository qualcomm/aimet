# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
Lite Mixed-Precision (Lite-MP) Feature Runner for AIMET ONNX

This module implements AIMET's Lite Mixed-Precision quantization technique, which
intelligently balances accuracy and performance by selectively applying higher
precision to sensitive layers.

Algorithm Overview:
1. Create a standard INT8 quantized model using QuantSim
2. Analyze per-layer sensitivity to quantization
3. Identify the most sensitive layers (those causing highest accuracy loss)
4. Flip these sensitive layers to higher precision (FP16)
5. Keep remaining layers at INT8 for efficiency

This approach typically achieves better accuracy than pure INT8 quantization
with minimal performance impact, as only a small percentage of layers are
promoted to higher precision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List

import onnx
import onnxruntime as ort
from qai_hub_models.datasets import BaseDataset
from qai_hub_models.utils.evaluate.helpers import evaluate_session_on_dataset

# AIMET imports - these symbols are required objects, not strings
from aimet_onnx import analyze_per_layer_sensitivity, int4, int8, int16, float16
from aimet_onnx.lite_mp import flip_layers_to_higher_precision
from aimet_onnx.utils import make_psnr_eval_fn

from AIMETRegression.evaluation.calibration import run_onnx_calibration
from AIMETRegression.evaluation.metrics_utils import measure_inference_metrics
from AIMETRegression.features.onnx._common import (
    build_quantsim,
    export_onnx_qdq,
    format_quantsim_technique,
    pick_providers,
)

_ARTIFACTS_DIR = Path("./AIMETRegression/artifacts")
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _collect_inputs(
    sess: ort.InferenceSession, model, dataset_cls: type[BaseDataset], num_samples: int
) -> List[Dict]:
    inputs = []
    run = sess.run

    def wrapper(output_names, input_feed, *args, **kwargs):
        inputs.append(input_feed)
        return run(output_names, input_feed, *args, **kwargs)

    sess.run = wrapper
    run_onnx_calibration(sess, model, dataset_cls, num_samples)
    sess.run = run
    return inputs


def _resolve_override_precision(val: Any):
    """
    Convert configuration value to AIMET's required precision symbol.

    AIMET's flip_layers_to_higher_precision API requires specific symbol
    objects rather than strings. This function handles conversion from
    various user-friendly input formats.

    Args:
        val: Input value which can be:
            - AIMET symbol object (int4, int8, int16, float16) - returned unchanged
            - String variants: "int4"/"i4"/"4", "int8"/"i8"/"8",
                              "int16"/"i16"/"16", "float16"/"fp16"/"f16"

    Returns:
        AIMET precision symbol (aimet_onnx.int4/int8/int16/float16)

    Raises:
        TypeError: If the value cannot be mapped to a valid precision symbol
    """
    # Check if already a valid symbol
    if val is int4 or val is int8 or val is int16 or val is float16:
        return val

    # Parse string representations
    if isinstance(val, str):
        normalized = val.strip().lower()

        # INT4 variants
        if normalized in ("int4", "i4", "4"):
            return int4

        # INT8 variants
        if normalized in ("int8", "i8", "8"):
            return int8

        # INT16 variants
        if normalized in ("int16", "i16", "16"):
            return int16

        # FP16 variants
        if normalized in ("float16", "fp16", "f16"):
            return float16

    # Invalid input
    raise TypeError(
        f"override_precision must be one of: int4, int8, int16, float16 "
        f"(or string 'int4'/'int8'/'int16'/'float16'). Got {type(val).__name__}: {val!r}"
    )


def run_lite_mp(
    *,
    fp32_onnx_path: str,
    model: Any,
    dataset_cls: type[BaseDataset],
    calib_dataset_cls: type[BaseDataset] | None = None,
    config: Dict[str, Any],
    export_dir: Optional[Path] = None,
) -> Tuple[str, float, Dict[str, str], str]:
    """
    Apply Lite Mixed-Precision quantization to an FP32 ONNX model.

    This technique provides an optimal accuracy-performance trade-off by using
    mixed precision: INT8 for most layers, FP16 for sensitive layers. The
    sensitivity analysis identifies which layers contribute most to accuracy
    loss when quantized.

    Args:
        fp32_onnx_path: Path to the FP32 ONNX model from AI Hub
        model: QAI Hub model object (provides pre/post-processing)
        dataset_cls: Dataset class for evaluation (e.g., ImagenetDataset)
        config: Configuration dictionary containing:
            Required:
                - model_name: Name for output files
            Quantization:
                - quant_scheme: Base quantization scheme (default: "tf_enhanced")
                - param_type: Base weight precision (default: "int8")
                - activation_type: Base activation precision (default: "int8")
                - config_file: Optional AIMET config file path
            Lite-MP specific:
                - percent_flip: Percentage of layers to flip (default: 10)
                - override_precision: Target precision for sensitive layers (default: "float16")
                - lite_mp: Dict with above options (alternate location for backward compatibility)
            Evaluation:
                - calib_samples: Calibration samples (default: 256)
                - eval_samples: Evaluation samples (default: 256)
                - metrics_samples: Samples for performance measurement (default: 64)
                - metrics_runs: Number of timing runs (default: 1)
                - metrics_warmup: Warmup runs before timing (default: 0)

    Returns:
        Tuple containing:
            - exported_onnx_path: Path to the exported mixed-precision ONNX model
            - feature_accuracy: Model accuracy after Lite-MP quantization
            - stats: Dict with "techniques", "runtime", and "memory" keys

    Raises:
        TypeError: If override_precision cannot be resolved to a valid symbol
        Exception: If AIMET processing fails
    """
    # ============ Extract Configuration ============
    model_name = config["model_name"]

    # Use provided export_dir or extract from config (passed by runner.py)
    if export_dir is None:
        export_dir = config.get("_export_dir")
        if export_dir:
            export_dir = Path(export_dir)
        else:
            # Fallback to default location with model subdirectory
            export_dir = Path("./AIMETRegression/artifacts") / model_name
            export_dir.mkdir(parents=True, exist_ok=True)
    else:
        export_dir = Path(export_dir)

    # Base quantization parameters
    quant_scheme = str(config.get("quant_scheme", "tf_enhanced"))
    param_type = str(config.get("param_type", "int8"))
    activation_type = str(config.get("activation_type", "int8"))
    aimet_cfg_file = config.get("config_file", None)

    # Lite-MP specific parameters
    # Support both top-level and nested configuration for flexibility
    percent_to_flip = int(
        config.get("percent_flip", config.get("lite_mp", {}).get("percent_flip", 10))
    )
    override_cfg = config.get(
        "override_precision",
        config.get("lite_mp", {}).get("override_precision", "float16"),
    )
    lite_mp_samples = int(config.get("lite_mp", {}).get("eval_samples", 5))

    # Convert precision string to AIMET symbol
    override_sym = _resolve_override_precision(override_cfg)

    # Evaluation parameters
    calib_samples = int(config.get("calib_samples", 256))
    eval_samples = int(config.get("eval_samples", 256))
    metrics_samples = int(config.get("metrics_samples", 64))
    metrics_runs = int(config.get("metrics_runs", 1))
    metrics_warmup = int(config.get("metrics_warmup", 0))

    providers = pick_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])

    # Get precision name for display
    precision_display = {
        int4: "INT4",
        int8: "INT8",
        int16: "INT16",
        float16: "FP16",
    }.get(override_sym, "UNKNOWN")

    print(f"[Lite-MP] Configuration:")
    print(f"  Base quantization: W{param_type}/A{activation_type}")
    print(f"  Override precision: {precision_display}")
    print(f"  Layers to flip: {percent_to_flip}%")
    print(
        f"  CUDA acceleration: "
        f"{'Enabled' if 'CUDAExecutionProvider' in providers else 'Disabled'}"
    )

    # ============ Step 1: Build Base QuantSim Model ============
    print(f"[Lite-MP] Building base QuantSim model...")

    sim = build_quantsim(
        fp32_or_fpN_onnx_path=fp32_onnx_path,
        scheme=quant_scheme,
        param_type=param_type,
        activation_type=activation_type,
        config_file=aimet_cfg_file,
        providers=providers,
    )

    # Also load FP32 model for comparison (optional, for debugging)
    fp32_model = onnx.load(fp32_onnx_path)
    fp32_sess = ort.InferenceSession(
        fp32_model.SerializeToString(), providers=sim.session.get_providers()
    )

    # ============ Step 2: Initial Calibration ============
    print(f"[Lite-MP] Calibrating base model with {calib_samples} samples...")

    def calibration_callback(sess: ort.InferenceSession, _unused=None):
        """Forward pass callback for AIMET calibration."""
        run_onnx_calibration(
            sess, model, calib_dataset_cls or dataset_cls, calib_samples
        )

    # Compute initial encodings for INT8 quantization
    sim.compute_encodings(forward_pass_callback=calibration_callback)

    # ============ Step 3: Sensitivity Analysis ============
    print(f"[Lite-MP] Analyzing per-layer sensitivity...")
    inputs = _collect_inputs(
        sim.session,
        model,
        calib_dataset_cls or dataset_cls,
        num_samples=lite_mp_samples,
    )
    accuracy_evaluator = make_psnr_eval_fn(fp32_sess, inputs, output_indices=None)

    # Analyze which layers contribute most to quantization error
    layer_sensitivity = analyze_per_layer_sensitivity(sim, eval_fn=accuracy_evaluator)

    # ============ Step 4: Apply Mixed Precision ============
    precision_name = {int4: "INT4", int8: "INT8", int16: "INT16", float16: "FP16"}.get(
        override_sym, "UNKNOWN"
    )

    print(
        f"[Lite-MP] Promoting top {percent_to_flip}% sensitive layers to {precision_name}..."
    )

    # Flip most sensitive layers to higher precision
    flip_layers_to_higher_precision(
        sim,
        layer_sensitivity_dict=layer_sensitivity,
        percent_to_flip=percent_to_flip,
        override_precision=override_sym,  # Must be AIMET symbol, not string
    )

    # ============ Step 5: Re-calibrate After Precision Changes ============
    print(f"[Lite-MP] Re-calibrating after precision changes...")

    # Re-compute encodings with mixed precision configuration
    sim.compute_encodings(forward_pass_callback=calibration_callback)

    # ============ Step 6: Evaluate Mixed-Precision Model ============
    print(f"[Lite-MP] Evaluating mixed-precision model with {eval_samples} samples...")

    feature_acc, *_ = evaluate_session_on_dataset(
        sim.session, model, dataset_cls, num_samples=eval_samples
    )
    feature_acc = float(feature_acc)

    print(f"[Lite-MP] Mixed-precision accuracy: {feature_acc:.4f}")

    # ============ Step 7: Measure Performance ============
    print(f"[Lite-MP] Measuring inference performance...")

    runtime_str, memory_str = measure_inference_metrics(
        lambda: evaluate_session_on_dataset(
            sim.session, model, dataset_cls, num_samples=metrics_samples
        ),
        runs=metrics_runs,
        warmup=metrics_warmup,
    )

    print(f"[Lite-MP] Runtime: {runtime_str}, Memory: {memory_str}")

    # ============ Step 8: Export and Bundle ============
    print(f"[Lite-MP] Exporting mixed-precision model...")

    qdq_path = export_onnx_qdq(sim, export_dir, model_name)

    # ============ Step 9: Prepare Results ============
    # Map precision symbol to string name
    precision_map = {int4: "int4", int8: "int8", int16: "int16", float16: "float16"}
    precision_str = precision_map.get(override_sym, "unknown")

    quantsim_desc = format_quantsim_technique(param_type, activation_type, quant_scheme)
    technique_desc = f"{quantsim_desc} + lite_mp({precision_str}, {percent_to_flip}%)"

    stats = {
        "techniques": technique_desc,
        "runtime": runtime_str,
        "memory": memory_str,
    }

    return qdq_path, feature_acc, stats
