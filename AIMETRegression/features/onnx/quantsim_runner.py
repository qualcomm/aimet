# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=missing-module-docstring

"""
QuantSim Feature Runner

This module applies AIMET's QuantizationSimModel to an FP32 ONNX model.
QuantSim simulates INT8 quantization effects without actually quantizing the model,
allowing us to evaluate accuracy degradation before deployment.

The process:
1. Build QuantSim from FP32 ONNX with specified quantization parameters
2. Calibrate quantization encodings using representative data
3. Evaluate accuracy with simulated quantization
4. Export QDQ (Quantize-Dequantize) ONNX and encodings
5. Bundle for QNN compilation

"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple

import onnxruntime as ort
from qai_hub_models.datasets import BaseDataset
from qai_hub_models.utils.evaluate.helpers import evaluate_session_on_dataset

from AIMETRegression.evaluation.calibration import run_onnx_calibration
from AIMETRegression.evaluation.metrics_utils import measure_inference_metrics
from AIMETRegression.features.onnx._common import (
    build_quantsim,
    export_onnx_qdq,
    format_quantsim_technique,
    pick_providers,
)

# Output directory for artifacts
_ARTIFACTS_DIR = Path("./AIMETRegression/artifacts")
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def run_quantsim(
    *,
    fp32_onnx_path: str,
    model: Any,
    dataset_cls: type[BaseDataset],
    calib_dataset_cls: type[BaseDataset] | None = None,
    config: Dict[str, Any],
    export_dir: Optional[Path] = None,
) -> Tuple[str, float, Dict[str, str], str]:
    """
    Apply AIMET QuantSim to simulate INT8 quantization on an FP32 ONNX model.

    QuantSim creates a simulation model that mimics quantization behavior during
    inference without actually quantizing the weights. This allows accurate
    evaluation of quantization-induced accuracy loss.

    Args:
        fp32_onnx_path: Path to the FP32 ONNX model from AI Hub compilation
        model: QAI Hub model object (provides preprocessing/postprocessing)
        dataset_cls: Dataset class for evaluation (e.g., ImagenetDataset)
        config: Configuration dictionary containing:
            - model_name: Name of the model
            - quant_scheme: Quantization scheme (default: "tf_enhanced")
            - param_type: Weight precision (default: "int8")
            - activation_type: Activation precision (default: "int8")
            - config_file: Optional AIMET config file
            - calib_samples: Number of calibration samples (default: 256)
            - eval_samples: Number of evaluation samples (default: 256)
            - metrics_samples: Samples for performance measurement (default: 64)
            - metrics_runs: Number of timing runs (default: 1)
            - metrics_warmup: Warmup runs before timing (default: 0)

    Returns:
        Tuple containing:
            - exported_onnx_path: Path to the exported QDQ ONNX model
            - feature_accuracy: Accuracy after quantization simulation
            - stats: Dictionary with runtime and memory statistics

    Raises:
        Exception: If QuantSim creation or evaluation fails
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

    # Quantization parameters with defaults
    quant_scheme = str(config.get("quant_scheme", "tf_enhanced"))
    param_type = str(config.get("param_type", "int8"))
    activation_type = str(config.get("activation_type", "int8"))
    aimet_cfg_file = config.get("config_file", None)  # Optional AIMET config

    # Sample sizes for different stages
    calib_samples = int(config.get("calib_samples", 256))
    eval_samples = int(config.get("eval_samples", 256))
    metrics_samples = int(config.get("metrics_samples", 64))
    metrics_runs = int(config.get("metrics_runs", 1))
    metrics_warmup = int(config.get("metrics_warmup", 0))

    providers = pick_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])

    print(f"[QuantSim] Configuration:")
    print(f"  Scheme: {quant_scheme}")
    print(f"  Precision: W{param_type}/A{activation_type}")
    print(f"  CUDA: {'Yes' if 'CUDAExecutionProvider' in providers else 'No'}")

    # ============ Build QuantSim Model ============
    print(f"[QuantSim] Building QuantizationSimModel...")

    sim = build_quantsim(
        fp32_or_fpN_onnx_path=fp32_onnx_path,
        scheme=quant_scheme,
        param_type=param_type,
        activation_type=activation_type,
        config_file=aimet_cfg_file,
        providers=providers,
    )

    # ============ Calibrate Encodings ============
    print(f"[QuantSim] Calibrating encodings with {calib_samples} samples...")

    def calibration_callback(sess: ort.InferenceSession, _unused=None):
        """
        Callback function for AIMET to calibrate quantization encodings.
        Runs representative data through the model to determine optimal
        quantization parameters (scale/zero-point) for each layer.
        """
        run_onnx_calibration(
            sess, model, calib_dataset_cls or dataset_cls, calib_samples
        )

    # Compute optimal quantization encodings
    sim.compute_encodings(forward_pass_callback=calibration_callback)

    print(f"[QuantSim] Calibration complete")

    # ============ Evaluate Quantized Accuracy ============
    print(f"[QuantSim] Evaluating accuracy with {eval_samples} samples...")

    feature_acc, *_ = evaluate_session_on_dataset(
        sim.session, model, dataset_cls, num_samples=eval_samples
    )
    feature_acc = float(feature_acc)

    print(f"[QuantSim] Accuracy after quantization: {feature_acc:.4f}")

    # ============ Measure Performance ============
    print(f"[QuantSim] Measuring runtime and memory...")

    runtime_str, memory_str = measure_inference_metrics(
        lambda: evaluate_session_on_dataset(
            sim.session, model, dataset_cls, num_samples=metrics_samples
        ),
        runs=metrics_runs,
        warmup=metrics_warmup,
    )

    print(f"[QuantSim] Runtime: {runtime_str}, Memory: {memory_str}")

    # ============ Export AIMET Artifacts ============
    print(f"[QuantSim] Exporting QDQ ONNX and encodings...")

    # Export QDQ ONNX (encodings embedded) for QNN
    qdq_path = export_onnx_qdq(sim, export_dir, model_name)

    # ============ Prepare Results ============
    technique_str = format_quantsim_technique(param_type, activation_type, quant_scheme)
    stats = {
        "techniques": technique_str,
        "runtime": runtime_str,
        "memory": memory_str,
    }

    return qdq_path, feature_acc, stats
